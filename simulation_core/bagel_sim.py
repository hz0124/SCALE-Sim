import os
import json
import csv
import math
from datetime import datetime
from scalesim.scale_sim import scalesim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Bagel_sim():
    def __init__(self, hardware_type="ours", task="GenEdit"):
        """
        Args:
            hardware_type: 'ours', 'sdma', 'flightvgm', 'figna', 'base' and so on
            task: 'GenEdit', 'GenEval', 'MM'
            config_file: if you want to use a custom config file, please specify the path
        """
        self.hardware_type = hardware_type
        self.task = task

        self.kv_cache_init = None
        self.kv_cache_without_img = None
        self.kv_cache_without_text = None
        self.gen_text_len = None
        self.gen_image_step = None
        self.config_comp0 = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        self.config_comp1 = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        self.config_comm0 = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        self.config_comm1 = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        # Text-stage FFN slot. For ARGUS this points at the weight-INT8-
        # storage cfg (ours_ffn_w8.cfg: 32x32 FP-FP compute, Bandwidth 16
        # to model the x2 weight traffic); for every other hardware it
        # equals the projection cfg, so behaviour is unchanged.
        self.config_ffn_text = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        self.log_path = os.path.join(PROJECT_ROOT, "results")
        self.result_path = os.path.join(PROJECT_ROOT, "results", "bagel")
        self.num_layer = 28
        self.text_input_len = 1
        self.image_input_len = 1378
        self.text_attn = 2
        self.vae_attn = 1376
        self.num_head_kv = 4
        self.num_head_q = 28
        self.dim = 3584
        self.head_dim = 128
        self.upshape = 18944
        self.tile = 64
        # Need-6 batch-size sweep. Decode batching model (REBUTTAL_FRAMEWORK
        # need 6): weight-shared projections/FFN run as one batched GEMM
        # (M=batch_size -> sub-linear cycle growth while batch <= array_height),
        # attention is per-request (independent KV) so it scales x batch_size,
        # and the image stage is already array-saturated (M=1378) so it simply
        # scales x batch_size. Default 1 reproduces the unbatched numbers.
        self.batch_size = 1
        self.total_cycles_all = 0
        # Per-operator cycle breakdown (req 3): label -> total cycle
        # contribution, with the SAME multiplicity used for total_cycles_all
        # (num_layer × steps × heads × branches). Sum equals total_cycles_all.
        self.cycle_breakdown = {}
        self.sample_rate = 10
        self.ifmapbufsz = 0
        self.filterbufsz = 0
        self.ofmapbufsz = 0
        self.ifmapbw = 0
        self.filterbw = 0
        self.ofmapbw = 0
        self.active_rate = 0.5
        self.array_height = 0
        self.array_width = 0
        self.array_width_half = 0
        self.array_width_fp = 0
        self.array_width_ffn = 0
        self.array_width_proj = 0
        # need-9 disagg: physical lane budget for the utilization denominator
        # and idle-leakage split (default = array_height × array_width).
        self.pe_total = None
        # need-9 half-bandwidth study: override the SCALE-Sim cfg `Bandwidth`
        # at run time (text-stage off-chip BW). None = use the cfg value
        # untouched -> every other hw / the full-BW disagg group is unchanged.
        self.scalesim_bw_override = None
        # Peak off-chip DRAM bandwidth (GB/s) for the per-stage memory-util
        # denominator. Defaults to the aligned 32 GB/s every accelerator gets.
        self.peak_dram_gbps = 32.0
        # ARGUS projection-quantization extension (framework §15), default off.
        self.quant_proj = False
        # ARGUS technique switches (JSON keys, default all-on; consulted
        # only for hardware_type == 'ours'). See REBUTTAL_FRAMEWORK.md §4.5.
        # t2_hard is a placeholder: SAU overhead is not modelled yet.
        self.t1_soft = True
        self.t1_hard = True
        self.t2_soft = True
        self.t2_hard = True
        self.t3_soft = True
        self.sparsity_kv = 0.5
        self.sparsity_cross_attn = 1
        self.low_precise_self_attn = 0
        self.image_only_sim = 0
        self.text_only_sim = 0
        self.text_gen_finished_flag = False
        self.text_gen_cycles = 0

        # Energy accounting (Phase F). Disabled by default; opt-in via
        # CLI --energy or JSON "energy_enabled": true. See
        # simulation_core/energy_accounting/.
        self.energy_enabled = False
        self.energy = None              # set in setup_energy()
        self.dram_type = None           # JSON "dram_type", default in coefficients.py
        self.coef_overrides = None      # JSON "energy_coefficients"
        self.energy_validate = False    # set by --validate; runs accelergy CLI
                                        # cross-check on top-N sub-runs at end


    def setup_energy(self, hw_type=None):
        """
        Instantiate self.energy (EnergyAccountant) using coefficients keyed
        by hw_type (defaults to self.hardware_type). Idempotent: calling
        twice is harmless.
        """
        if self.energy is not None:
            return
        from simulation_core.energy_accounting.coefficients import get_coefficients
        from simulation_core.energy_accounting.accountant import EnergyAccountant
        coef = get_coefficients(
            hw_type or self.hardware_type,
            json_overrides=self.coef_overrides,
            dram_type=self.dram_type,
        )
        self.energy = EnergyAccountant(coef)
        self.energy_enabled = True


    def _record_energy(self, M, N, K, precision="fp16", scale=1.0,
                       multiplicity=1, label=None, hw_cfg_path=None,
                       weight_precision=None, mac_split=None):
        """
        Add one sub-run's MAC + SRAM + DRAM access to self.energy.

        Access counts come from the closed-form geometric estimate. (Sourcing
        them from the tiled SCALE-Sim DETAILED_ACCESS_REPORT was tried and
        removed: its DRAM counts are dominated by fixed-size prefetch-buffer
        fills on the sub-array tile, not real traffic — see DEVLOG #4.)

        Args:
            M, N, K:    real GEMM dims (M×K · K×N → M×N), post dim/tile blowup.
            precision:  fp16 / int8 / int4
            scale:      multiplier applied to access counts (used when M, N,
                        K describe the per-tile sub-run and the caller wants
                        to amortize the same scale ARGUS uses for cycles).
            multiplicity: outer multiplier for "this sub-run gets executed
                        N times across the full simulation" — e.g. once-only
                        text projections multiply by num_layer × gen_text_len,
                        per-step attention by num_layer × step_cnts × num_head_q.
                        ARGUS's cycle accumulation already applies the same
                        factor to cycles; multiplicity makes energy parallel.
            label:      Short human-readable tag (e.g. 'qkv', 'attn_qk',
                        'ffn_up'). Used in F7 cross-validation output.
            hw_cfg_path: SCALE-Sim cfg path for this sub-run (e.g.
                        'configs/bagel/ours_int4.cfg'). Needed by F7
                        validate to spawn accelergy with matching hw.
            weight_precision: optional storage precision for the filter
                        operand only (its SRAM/DRAM bytes). Use when low
                        precision is a *storage* format, not a compute
                        one — e.g. ARGUS text-stage FFN weights live as
                        INT8 in memory (1B traffic) but are computed at
                        fp16 after CRU dequant. Defaults to `precision`.
            mac_split:  optional {precision: fraction} dict splitting the
                        MAC count across precisions (fractions sum to 1).
                        E.g. ARGUS image-stage FFN on the INT4+BF16 dual
                        array bills {"fp16": .5, "w4a16": .5}; figna's W4A16
                        FP-INT multiplier bills {"w4a16": 1.0} while keeping
                        fp16-activation / int4-weight bytes. The `w4a16` MAC
                        coefficient (int4 weight x 16-bit activation) is NOT
                        pure int4 — see coefficients.py. Defaults to all MACs
                        at `precision`.
        """
        if not self.energy_enabled or self.energy is None:
            return
        if multiplicity <= 0:
            return
        from simulation_core.energy_accounting.extractors import from_geometric_estimate

        # Combined factor: per-sub-run scale × outer multiplicity.
        total_factor = float(scale) * float(multiplicity)

        # Snapshot energy before adding so we can attribute this sub-run's
        # contribution to the F7 audit trail.
        before_pJ = self.energy.total_pJ() if (label or hw_cfg_path) else 0.0

        # MAC ops: always M*N*K * total_factor (more reliable than the
        # SCALE-Sim CSV PE-level Read Count, which is biased by col_fold).
        total_macs = int(M * N * K * total_factor)
        if mac_split:
            for mac_prec, frac in mac_split.items():
                self.energy.add_mac_ops(int(total_macs * frac), precision=mac_prec)
        else:
            self.energy.add_mac_ops(total_macs, precision=precision)

        counts = from_geometric_estimate(
            M, N, K, self.array_height or 32, self.array_width or 32
        )
        counts = {k: int(v * total_factor) for k, v in counts.items()}

        # Translate precision to bytes-per-element. Activations (ifmap/
        # ofmap) follow `precision`; the filter operand may have its own
        # storage precision (weight_precision).
        _bytes = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
        act_bytes = _bytes.get(precision, 2.0)
        w_bytes = _bytes.get(weight_precision, act_bytes) if weight_precision else act_bytes

        dram_bytes_this = 0.0   # off-chip bytes this sub-run (for per-op util)
        for kind in ("ifmap", "filter", "ofmap"):
            word_bytes = w_bytes if kind == "filter" else act_bytes
            if kind == "ofmap":
                self.energy.add_sram_access(
                    kind, words=counts["sram_ofmap_writes"],
                    word_bytes=word_bytes, op="write")
                self.energy.add_dram_access(
                    kind, words=counts["dram_ofmap_writes"],
                    word_bytes=word_bytes, op="write")
                dram_bytes_this += counts["dram_ofmap_writes"] * word_bytes
            else:
                self.energy.add_sram_access(
                    kind, words=counts[f"sram_{kind}_reads"],
                    word_bytes=word_bytes, op="read")
                self.energy.add_dram_access(
                    kind, words=counts[f"dram_{kind}_reads"],
                    word_bytes=word_bytes, op="read")
                dram_bytes_this += counts[f"dram_{kind}_reads"] * word_bytes

        # Per-operator breakdown for the detailed utilization report. Derive an
        # op tag from the label: stage (text/img) × optype (attn/ffn/proj).
        if label:
            stage = "text" if label.startswith("text/") else "img"
            optype = "attn" if "attn" in label else ("ffn" if "ffn" in label else "proj")
            self.energy.add_op_breakdown(f"{stage}/{optype}", total_macs, dram_bytes_this)

        # Audit trail entry: subtotal_pJ = (after - before) gives this call's
        # net contribution, regardless of which buckets it touched.
        if label or hw_cfg_path:
            self.energy.record_subrun(
                label=label or f"{M}x{N}x{K}",
                M=M, N=N, K=K, precision=precision,
                multiplicity=multiplicity, scale=scale,
                hw_cfg_path=hw_cfg_path,
                subtotal_pJ=self.energy.total_pJ() - before_pJ,
            )


    def read_from_json(self, cfg_path=None):
        if cfg_path is None:
            base = os.path.dirname(__file__)
            cfg_path = os.path.join(PROJECT_ROOT, "topologies", "bagel", "config.json")
        
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            cfg = {}
        except FileNotFoundError:
            print(f"Config file not found: {cfg_path}")
            cfg = {}
        
        self.kv_cache_init = cfg.get("kv_cache_init")
        self.gen_text_len = cfg.get("gen_text_len")
        self.gen_image_step = cfg.get("gen_image_step")

        # ARGUS technique switches (REBUTTAL_FRAMEWORK.md §4.5). Read before
        # the cfg-slot dispatch because t3_soft decides the FFN slot.
        self.t1_soft = bool(cfg.get("t1_soft", True))
        self.t1_hard = bool(cfg.get("t1_hard", True))
        self.t2_soft = bool(cfg.get("t2_soft", True))
        self.t2_hard = bool(cfg.get("t2_hard", True))
        self.t3_soft = bool(cfg.get("t3_soft", True))
        # Optional ARGUS extension (framework §15): quantize qkv/omap projection
        # weights to INT4 so the projections use the FULL dual array (and W4A16
        # billing) instead of the FP-FP half array. Default off — the shipped
        # ours numbers keep unquantized projections. Only affects ours.
        self.quant_proj = bool(cfg.get("quant_proj", False))

        if self.hardware_type == 'base' or self.hardware_type == 'flightvgm' or self.hardware_type == 'sdma':
            self.config_comp0 = cfg.get("config", self.config_comp0)
            self.config_comp1 = cfg.get("config", self.config_comp1)
            self.config_comm0 = cfg.get("config", self.config_comm0)
            self.config_comm1 = cfg.get("config", self.config_comm1)
            self.config_ffn_text = cfg.get("config", self.config_ffn_text)
        elif self.hardware_type == 'ours':
            # No FP-INT8 compute unit exists: the text stage runs entirely
            # on the 32x32 FP-FP array (the FP-INT4 array idles). INT8 is a
            # storage format for FFN weights only, so qkv/omap/attention all
            # take the plain fp16 cfg and only the FFN slot gets the
            # weight-bandwidth-doubled cfg (when T3 is on).
            self.config_comp0 = cfg.get("config_fp16", self.config_comp0)
            self.config_comp1 = cfg.get("config_int4", self.config_comp1)
            self.config_comm0 = cfg.get("config_fp16", self.config_comm0)
            self.config_comm1 = cfg.get("config_fp16", self.config_comm1)
            if self.quant_proj:
                # qkv/omap (config_comm0, drives both text and image text-side
                # projections) routed to the full-array INT4 projection cfg.
                self.config_comm0 = cfg.get(
                    "config_proj_int4", cfg.get("config_int4", self.config_comm0))
            if self.t3_soft:
                if self.task == 'MM':
                    # MM is pure-text decode: there is no diffusion stage to keep
                    # the FP-INT4 array busy, so the AR-stage FFN runs on it at
                    # W4A16 (INT4 weights), matching figna — instead of the
                    # INT8-storage FP-FP path used when an image stage follows.
                    self.config_ffn_text = cfg.get(
                        "config_int4", cfg.get("config_ffn_text", self.config_ffn_text))
                else:
                    self.config_ffn_text = cfg.get(
                        "config_ffn_text", cfg.get("config_fp16", self.config_ffn_text))
            else:
                self.config_ffn_text = cfg.get("config_fp16", self.config_ffn_text)
        elif self.hardware_type in ('figna', 'axcore'):
            self.config_comp0 = cfg.get("config_fp16", self.config_comp0)
            self.config_comp1 = cfg.get("config_int4", self.config_comp1)
            self.config_comm0 = cfg.get("config_int4", self.config_comm0)
            self.config_comm1 = cfg.get("config_fp16", self.config_comm1)
            self.config_ffn_text = cfg.get("config_int4", self.config_ffn_text)
        elif self.hardware_type == 'disagg':
            # need-9 disaggregated baseline. TEXT stage runs on the narrow
            # FIGNA-style text-accel (W4A16 proj/FFN on the int4 cfg, fp16
            # attention) via SCALE-Sim; these text-accel cfgs encode the
            # text-accel array width (e.g. 32x4). The IMAGE stage runs on the
            # wide fp16 vision-accel (config_vision); its array width comes
            # from JSON array_width and run_gen_image switches config_comm0
            # over to config_vision at its start. Gets T1+T2 (sparse cross-
            # attn + FFN reuse) but NOT T3 (no unified stage-adaptive array).
            self.config_comm0 = cfg.get("config_text_int4", self.config_comm0)
            self.config_comm1 = cfg.get("config_text_fp16", self.config_comm1)
            self.config_ffn_text = cfg.get("config_text_int4", self.config_ffn_text)
            self.config_comp0 = cfg.get("config_vision", self.config_comp0)
            self.config_comp1 = cfg.get("config_vision", self.config_comp1)

        self.log_path = cfg.get("log_path", self.log_path)
        self.result_path = cfg.get("result_path", self.result_path)
        image_len = cfg.get("image_len", self.image_input_len)
        text_len = cfg.get("text_len", self.text_input_len)
        self.kv_cache_without_img = self.kv_cache_init - image_len
        self.kv_cache_without_text = self.kv_cache_init - text_len
        self.sample_rate = cfg.get("sample_rate", self.sample_rate)
        self.ifmapbufsz = cfg.get("ifmapbufsz", self.ifmapbufsz)
        self.filterbufsz = cfg.get("filterbufsz", self.filterbufsz)
        self.ofmapbufsz = cfg.get("ofmapbufsz", self.ofmapbufsz)
        self.ifmapbw = cfg.get("ifmapbw", self.ifmapbw)
        self.filterbw = cfg.get("filterbw", self.filterbw)
        self.ofmapbw = cfg.get("ofmapbw", self.ofmapbw)
        self.active_rate = cfg.get("activate_rate", self.active_rate)
        self.array_height = cfg.get("array_height", self.array_height)
        self.array_width = cfg.get("array_width", self.array_width)
        self.array_width_half = cfg.get("array_width_half", self.array_width)
        # FP-FP-only width for image-stage qkv/omap: their weights are not
        # quantized, so only the FP array computes them. Defaults to the
        # full array_width — non-ARGUS hardware is unaffected.
        self.array_width_fp = cfg.get("array_width_fp", self.array_width)
        # Effective width for image-stage qkv/omap projections. With quant_proj
        # the INT4 projection weights let both sub-arrays compute them (full
        # array_width); otherwise FP-FP only (array_width_fp). Non-ARGUS hw and
        # quant_proj-off ours fall back to array_width_fp.
        if self.hardware_type == 'ours' and self.quant_proj:
            self.array_width_proj = self.array_width
        else:
            self.array_width_proj = self.array_width_fp
        # Effective width for image-stage FFN. The INT4+BF16 dual-array
        # merge (width 64) is a T3 feature; with T3 off the FFN also runs
        # FP-only.
        if self.hardware_type == 'ours' and not self.t3_soft:
            self.array_width_ffn = self.array_width_fp
        else:
            self.array_width_ffn = self.array_width

        # Physical lane budget. For disagg the JSON array_width is the
        # vision-accel width only; pe_total (=2048) covers both accelerators
        # so the utilization denominator and idle-leakage split see the whole
        # chip. Defaults to the single-array size for every other hardware.
        self.pe_total = cfg.get("pe_total", self.array_height * self.array_width)
        # need-9 half-bandwidth group: optional SCALE-Sim Bandwidth override
        # (text stage) + peak-BW basis for the memory-util denominator.
        self.scalesim_bw_override = cfg.get("scalesim_bw_override", self.scalesim_bw_override)
        self.peak_dram_gbps = cfg.get("peak_dram_gbps", self.peak_dram_gbps)

        self.sparsity_kv = cfg.get("sparsity", self.sparsity_kv)
        self.sparsity_cross_attn = cfg.get("sparsity_cross_attn", self.sparsity_cross_attn)
        self.low_precise_self_attn = cfg.get("low_precise_self_attn", self.low_precise_self_attn)
        self.image_only_sim = cfg.get("image_only_sim", self.image_only_sim)
        self.text_only_sim = cfg.get("text_only_sim", self.text_only_sim)

        text_flag = cfg.get("text_gen_finished_flag", 'False')
        if text_flag == "True":
            self.text_gen_finished_flag = True
        else:
            self.text_gen_finished_flag = False

        self.text_gen_cycles = cfg.get("text_gen_cycles", 0)
        self.tile = cfg.get("tile", self.tile)
        self.batch_size = cfg.get("batch_size", self.batch_size)

        # Energy accounting (Phase F): JSON can opt-in independent of CLI.
        # CLI --energy in run_bagel.py also flips self.energy_enabled but goes
        # through setup_energy() to actually instantiate the accountant.
        if cfg.get("energy_enabled", False):
            self.energy_enabled = True
        self.dram_type = cfg.get("dram_type", self.dram_type)
        self.coef_overrides = cfg.get("energy_coefficients", self.coef_overrides)

    def build_topologies(self, kv_len=0, is_gen_text=True, part='all', m=None):
        """
        生成Bagel模型单层拓扑结构，并保存到相应的csv文件中

        Args:
        kv_len: KV cache长度，影响attention矩阵大小
        is_gen_text: True为文本生成，False为图像生成
        part: 指定生成哪个部分
        m: 覆盖 M 维(行数)。用于 batch-size sweep —— 把逐 token 的 M=1
           投影改成 batched GEMM 的 M=batch_size。None 时退回默认 layer_input。
        """
        # 创建输出目录
        output_dir = os.path.join(PROJECT_ROOT, "topologies", "bagel")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "layer.csv")

        # 生成矩阵操作列表
        operations = []
        layer_input = m if m is not None else (self.text_input_len if is_gen_text else self.image_input_len)

        # 如果文件存在，先删除以确保全新生成
        if os.path.exists(output_path):
            os.remove(output_path)

        # Q, K, V 映射
        if is_gen_text:
            if part in ['all', 'qkv']:
                operations.extend([
                    ("QKVmap", layer_input, self.tile, self.dim)
                ])
        
            # Multi-head attention操作
            if part in ['all', 'attn_qk']:
                operations.extend([
                    (f"Head_QKT", layer_input, self.tile, self.head_dim)
                ])
            if part in ['all', 'attn_sfmxv']:
                operations.extend([
                    (f"Head_SFMXxV", layer_input, self.tile, kv_len)
                ])

            if part in ['all', 'omap']:
                # 输出映射
                operations.append(("Omap", layer_input, self.tile, self.dim))
        
            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.tile, self.dim))

            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.tile, self.upshape))
        else:
            if part in ['all', 'qkv']:
                operations.extend([
                    ("Qmap_text", self.text_attn, self.dim, self.dim),
                    ("Kmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                ])

            # N=tile column strips of the image-stage text-side qkv, for the
            # tile-scaling fast path (OS dataflow is linear in N — same trick
            # the text-stage projections already use). Replaces the full-N
            # qkv call that iterated 400k+ SCALE-Sim steps; caller scales each
            # strip by (real_N / tile). See run_gen_image.
            if part == 'qkv_q_strip':
                operations.append(("Qmap_text", self.text_attn, self.tile, self.dim))
            if part == 'qkv_kv_strip':
                operations.append(("Kmap_text", self.text_attn, self.tile, self.dim))

                """operations.extend([
                    ("Qmap_vae", self.vae_attn, self.dim, self.dim),
                    ("Kmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                ])"""

            # Multi-head attention操作
            if part in ['all', 'attn']:
                operations.extend([
                    (f"Head_QKT", layer_input, kv_len, self.head_dim),
                    (f"Head_SFMXxV", layer_input, self.head_dim, kv_len),
                ])

            if part in ['all', 'omap']:
                # 输出映射
                operations.append(("Omap", layer_input, self.dim, self.dim))
        

            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.upshape, self.dim))

            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.dim, self.upshape))
        
        # 写入CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Layer", "M", "N", "K", ""])  # 表头
            for op in operations:
                writer.writerow([op[0], op[1], op[2], op[3], ""])
        
        print(f"Topology saved to: {output_path}")
        return output_path

    def run_sim_once(self, kv_length=0, is_gen_text=True, part='all', config="./configs/scale.cfg", m=None):
        topology_path = self.build_topologies(kv_len=kv_length, is_gen_text=is_gen_text, part=part, m=m)

        s = scalesim(
            save_disk_space=True,
            verbose=True,   # NB: upstream couples verbose into the result —
                            # verbose=False makes run_scale return 0 cycles.
            config=config,
            topology=topology_path,
            layout=os.path.join(PROJECT_ROOT, "layouts", "GEMM_mnk", "vit_l_KM_KN.csv"),
            input_type_gemm=True
        )

        # need-9 half-bandwidth group: statically halve the off-chip pipe by
        # overriding the cfg `Bandwidth` (USER mode) before the run. The cfg is
        # already parsed in scalesim.__init__ -> set_params, so config.bandwidths
        # is populated; run_scale reads it live. None for every other run.
        if self.scalesim_bw_override is not None:
            s.config.bandwidths = [int(self.scalesim_bw_override)]

        results = s.run_scale(top_path=self.log_path)

        # 解包结果，获取总周期数
        if isinstance(results, (tuple, list)) and len(results) >= 1:
            return results[0]  # 返回总周期数
        else:
            return results
    
    def _add_op(self, label, cycles):
        """Accumulate per-operator cycle contribution (req 3 breakdown)."""
        self.cycle_breakdown[label] = self.cycle_breakdown.get(label, 0.0) + cycles

    def _format_cycle_breakdown(self):
        """Per-operator cycle breakdown + % of end-to-end (req 3).
        Machine-parseable: 'OPBREAKDOWN <label> <cycles> <pct>' lines."""
        cb = self.cycle_breakdown
        total = sum(cb.values())
        if total <= 0:
            return "CYCLE BREAKDOWN: none recorded"
        lines = ["CYCLE BREAKDOWN (per-operator cycles / % of end-to-end):"]
        # stage subtotals first
        for stage in ("text", "img"):
            sub = sum(v for k, v in cb.items() if k.startswith(stage + "/"))
            if sub > 0:
                lines.append(f"  [{stage}] subtotal: {sub:,.0f}  ({100.0*sub/total:5.1f}%)")
                for k in sorted(cb):
                    if k.startswith(stage + "/"):
                        lines.append(f"      OPBREAKDOWN {k} {cb[k]:.0f} {100.0*cb[k]/total:.3f}")
        lines.append(f"  breakdown sum: {total:,.0f}  (total_cycles={self.total_cycles_all:,.0f})")
        return "\n".join(lines)

    def _append_to_log(self, message):
        """追加信息到日志文件"""
        # 确保结果目录存在
        log_file = self.result_path
        log_dir = os.path.dirname(log_file)

        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # 追加到日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(log_entry.strip())

    # --- need-9 / 2026-06-17: analytical memory-bound text-decode model -------
    # SCALE-Sim mis-models M=1 decode: its os mapping wastes ~31/32 of the array
    # (overall util ~3%) so the inflated compute time hides ALL memory traffic and
    # no bandwidth knob (off-chip or SRAM) ever binds (verified: stall=0 even with
    # weights >> SRAM, and across os/ws/is). Physically a well-mapped M=1 GEMV fills
    # the array (map N,K to the array, stream the single M) and is then bounded by
    # streaming its weights / KV from DRAM. We model each text-stage GEMM as
    # max(ideal-fill compute, operand stream), so bandwidth genuinely binds and
    # weight precision (int4 vs fp16) matters.
    _PREC_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}

    def _text_array_macs(self):
        """MACs/cycle the text stage can use (array assumed fully mapped for M=1)."""
        h = self.array_height or 32
        if self.hardware_type == 'disagg':
            # text-accel = whole chip minus the vision-accel lanes.
            return max(self.pe_total - h * self.array_width, h)
        if self.hardware_type == 'ours':
            # text runs on the FP-FP sub-array only (FP-INT4 array idle).
            return h * (self.array_width_fp or self.array_width)
        return h * self.array_width

    def _text_bw_bpc(self):
        """Off-chip bytes/cycle for the text stage (peak_dram_gbps @ 500 MHz).
        Half-bandwidth disagg sets peak_dram_gbps=16 -> this halves, doubling the
        memory-bound text time."""
        return self.peak_dram_gbps * 1e9 / 500_000_000.0

    def _text_gemm_cycles(self, M, N, K, w_bytes, bw_bpc, array_macs, act_bytes=None):
        """One M-row text GEMM: max(ideal-fill compute, operand-stream).
        compute = M*N*K / array_macs  (array fully mapped);
        stream  = (weights N*K*w_bytes + ifmap M*K*act + ofmap M*N*act) / bw_bpc
        — all off-chip operands loaded once (weights/KV dominate for M=1; the
        activation terms keep memory-util ≤100%). Stream is M-light on weights so
        batching amortizes the weight term (decode-batch win)."""
        if act_bytes is None:
            act_bytes = w_bytes
        compute = (M * N * K) / array_macs
        stream = (N * K * w_bytes + M * K * act_bytes + M * N * act_bytes) / bw_bpc
        return max(compute, stream)

    def run_gen_text(self):
        self._append_to_log(f"=========== Text Generation Started (total steps: {self.gen_text_len}) ===========")
        text_start_cycles = self.total_cycles_all

        # Energy accounting: per-operand precision (REBUTTAL_FRAMEWORK.md §5).
        #   ours : everything computes on the FP-FP array at fp16; with T3
        #          the FFN weights are merely *stored* as INT8 (1B traffic).
        #   figna: W4A16 — FP-INT4 MACs, int4 weight bytes, fp16 activations.
        #   else : uniform precision sniffed from the cfg file name.
        from simulation_core.energy_accounting.coefficients import precision_from_config_path
        prec_comm0 = precision_from_config_path(self.config_comm0)
        prec_comm1 = precision_from_config_path(self.config_comm1)
        if self.hardware_type == 'ours':
            if self.quant_proj:
                # INT4 projection weights × fp16 activations → W4A16.
                proj_kwargs = {"precision": "fp16", "weight_precision": "int4",
                               "mac_split": {"w4a16": 1.0}}
            else:
                proj_kwargs = {"precision": "fp16"}
            ffn_kwargs = {"precision": "fp16"}
            if self.t3_soft:
                if self.task == 'MM':
                    # figna-aligned W4A16 FFN for pure-text MM (see read_from_json):
                    # INT4 weights × fp16 activations, MACs billed as w4a16.
                    ffn_kwargs["weight_precision"] = "int4"
                    ffn_kwargs["mac_split"] = {"w4a16": 1.0}
                else:
                    ffn_kwargs["weight_precision"] = "int8"
        elif self.hardware_type in ('figna', 'disagg', 'axcore'):
            # disagg text-accel is FIGNA-style W4A16 (int4 weights × fp16 acts).
            proj_kwargs = {"precision": "fp16", "weight_precision": "int4",
                           "mac_split": {"w4a16": 1.0}}
            ffn_kwargs = dict(proj_kwargs)
        else:
            proj_kwargs = {"precision": prec_comm0}
            ffn_kwargs = {"precision": precision_from_config_path(self.config_ffn_text)}

        # Once-only sub-runs (qkv/omap/ffn_up/ffn_down) get reused across
        # every (layer, token); their multiplicity is num_layer * gen_text_len.
        once_mult = self.num_layer * self.gen_text_len

        # Real GEMM dims (M=1 for token-by-token decode):
        #   qkv      : N = dim + 2 * num_head_kv * head_dim, K = dim
        #   omap     : N = dim,       K = dim
        #   ffn_up   : N = upshape,   K = dim
        #   ffn_down : N = dim,       K = upshape
        qkv_N = self.dim + 2 * self.num_head_kv * self.head_dim

        # Analytical memory-bound text model (2026-06-17): cycles per GEMM =
        # max(ideal-fill compute, operand-stream/BW). Replaces the SCALE-Sim os
        # runs, which mis-modelled M=1 as compute-bound and hid all bandwidth.
        bw_bpc = self._text_bw_bpc()
        tmacs = self._text_array_macs()
        proj_wb = self._PREC_BYTES.get(
            proj_kwargs.get("weight_precision") or proj_kwargs.get("precision", "fp16"), 2.0)
        ffn_wb = self._PREC_BYTES.get(
            ffn_kwargs.get("weight_precision") or ffn_kwargs.get("precision", "fp16"), 2.0)
        kv_wb = self._PREC_BYTES.get(prec_comm1, 2.0)   # KV cache stored at attn precision
        proj_ab = self._PREC_BYTES.get(proj_kwargs.get("precision", "fp16"), 2.0)  # activation bytes
        ffn_ab = self._PREC_BYTES.get(ffn_kwargs.get("precision", "fp16"), 2.0)

        self._append_to_log(f"Text generation - Start qkv mapping")
        input_cycles = self._text_gemm_cycles(self.batch_size, qkv_N, self.dim, proj_wb, bw_bpc, tmacs, proj_ab)
        self._record_energy(M=self.batch_size, N=qkv_N, K=self.dim,
                            multiplicity=once_mult, label="text/qkv",
                            hw_cfg_path=self.config_comm0, **proj_kwargs)
        self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")

        self._append_to_log(f"Text generation - Start output mapping")
        output_cycles = self._text_gemm_cycles(self.batch_size, self.dim, self.dim, proj_wb, bw_bpc, tmacs, proj_ab)
        self._record_energy(M=self.batch_size, N=self.dim, K=self.dim,
                            multiplicity=once_mult, label="text/omap",
                            hw_cfg_path=self.config_comm0, **proj_kwargs)
        self._append_to_log(f"Text generation - End output mapping, total cycles:{output_cycles}")

        self._append_to_log(f"Text generation - Start FFN up")
        # Gated FFN (Bagel): gate + up are two same-shape projections, so ×2.
        ffn_up_cycles = self._text_gemm_cycles(self.batch_size, self.upshape, self.dim, ffn_wb, bw_bpc, tmacs, ffn_ab) * 2
        self._record_energy(M=self.batch_size, N=self.upshape, K=self.dim, scale=2.0,
                            multiplicity=once_mult, label="text/ffn_up_gate+up",
                            hw_cfg_path=self.config_ffn_text, **ffn_kwargs)
        self._append_to_log(f"Text generation - End FFN up, total cycles:{ffn_up_cycles}")

        self._append_to_log(f"Text generation - Start FFN down")
        ffn_down_cycles = self._text_gemm_cycles(self.batch_size, self.dim, self.upshape, ffn_wb, bw_bpc, tmacs, ffn_ab)
        self._record_energy(M=self.batch_size, N=self.dim, K=self.upshape,
                            multiplicity=once_mult, label="text/ffn_down",
                            hw_cfg_path=self.config_ffn_text, **ffn_kwargs)
        self._append_to_log(f"Text generation - End FFN down, total cycles:{ffn_down_cycles}")

        for step in range(0, self.gen_text_len, self.sample_rate):
            if step + self.sample_rate > self.gen_text_len:
                step_end = self.gen_text_len - 1
            else:
                step_end = step + self.sample_rate - 1

            len_average = (step + step_end) // 2
            kv_len = self.kv_cache_init + step + len_average + 1

            self._append_to_log(f"Text generation - Step {step} to Step {step_end} started (kv_len in averagr: {kv_len})")

            results_this_iter = 0
            results_this_iter += input_cycles
            # Attention (M=1 per query): reads the KV cache from DRAM — genuinely
            # KV-bandwidth-bound. attn_qk streams K-cache (kv_len×head_dim),
            # attn_sfmxv streams V-cache (kv_len×head_dim), both at kv precision.
            attn_qk_single_cycle = self._text_gemm_cycles(1, kv_len, self.head_dim, kv_wb, bw_bpc, tmacs, kv_wb)
            attn_sfmxv_single_cycle = self._text_gemm_cycles(1, self.head_dim, kv_len, kv_wb, bw_bpc, tmacs, kv_wb)
            attn_single_cycle = attn_qk_single_cycle + attn_sfmxv_single_cycle
            # Attention is per-request (each request has its own KV cache) -> it
            # cannot be merged into one batched GEMM the way the projections are,
            # so it scales linearly with batch_size. (Need-6 batch model.)
            attn_cycles = attn_single_cycle * self.num_head_q * self.batch_size
            results_this_iter += attn_cycles
            results_this_iter += output_cycles
            results_this_iter += ffn_up_cycles
            results_this_iter += ffn_down_cycles

            step_cycles = results_this_iter * self.num_layer

            step_cnts = step_end - step + 1
            step_cycles_total = step_cycles * step_cnts
            self.total_cycles_all += step_cycles_total

            # Per-operator cycle breakdown (same num_layer × step_cnts factor;
            # qkv/omap/ffn constant across steps, attn varies with kv_len).
            opmul = self.num_layer * step_cnts
            self._add_op("text/qkv", input_cycles * opmul)
            self._add_op("text/attn", attn_cycles * opmul)
            self._add_op("text/omap", output_cycles * opmul)
            self._add_op("text/ffn_up", ffn_up_cycles * opmul)
            self._add_op("text/ffn_down", ffn_down_cycles * opmul)

            # Energy: per-step attention. Real dims:
            #   attn_qk    : M=1, N=kv_len,   K=head_dim
            #   attn_sfmxv : M=1, N=head_dim, K=kv_len
            # Multiplicity per attn sub-run = num_layer * step_cnts * num_head_q.
            # Attention is per-request -> x batch_size (Need-6 batch model).
            attn_mult = self.num_layer * step_cnts * self.num_head_q * self.batch_size
            self._record_energy(M=1, N=kv_len, K=self.head_dim,
                                precision=prec_comm1, multiplicity=attn_mult,
                                label=f"text/attn_qk(kv={kv_len})", hw_cfg_path=self.config_comm1)
            self._record_energy(M=1, N=self.head_dim, K=kv_len,
                                precision=prec_comm1, multiplicity=attn_mult,
                                label=f"text/attn_sfmxv(kv={kv_len})", hw_cfg_path=self.config_comm1)


        text_total_cycles = self.total_cycles_all - text_start_cycles
        self._append_to_log(f"=========== Text Generation Completed, total cycles: {text_total_cycles}, total seconds: {text_total_cycles/500000000} ===========")

    def run_sim_once_comp(self, kv_len=0, is_gen_text=False, part='all', which_ffn = 'full_cache'):
        cycle_result = 0
        layer_input = self.text_input_len if is_gen_text else self.image_input_len
        if part in ['all', 'prefetch']:
            size_a = self.text_attn * self.dim
            size_b = self.dim * self.dim
            if size_a > self.active_rate * self.ifmapbufsz * 1024:
                prefetch_cycles_a = self.active_rate * self.ifmapbufsz * 1024 / self.ifmapbw
            else:
                prefetch_cycles_a = size_a / self.ifmapbw

            if size_b > self.active_rate * self.filterbufsz * 1024:
                prefetch_cycles_b = self.active_rate * self.filterbufsz * 1024 / self.filterbw
            else:
                prefetch_cycles_b = size_b / self.filterbw
            
            prefetch_cycles = max(prefetch_cycles_a, prefetch_cycles_b)
            cycle_result += prefetch_cycles

        # qkv/omap width = array_width_proj (set in read_from_json): with
        # quant_proj the INT4 projection weights let both sub-arrays compute
        # them at the merged array_width (W4A16); without quant_proj they fall
        # back to the FP-FP-only array_width_fp. The FFN uses array_width_ffn
        # (the INT4+BF16 dual-array merge under T3). For non-ARGUS hw all three
        # default to array_width.
        if part in ['all', 'qkv']:
            Qmap_row_fold = math.ceil(self.vae_attn/self.array_height)
            Qmap_col_fold = math.ceil(self.dim/self.array_width_proj)
            Qmap_cycle_each_fold = self.dim + self.array_height + self.array_width_proj - 2
            Qmap_cycles = Qmap_cycle_each_fold * Qmap_col_fold * Qmap_row_fold

            cycle_result += Qmap_cycles

            KVmap_row_fold = math.ceil(self.vae_attn/self.array_height)
            KVmap_col_fold = math.ceil(self.num_head_kv * self.head_dim/self.array_width_proj)
            KVmap_cycle_each_fold = self.dim + self.array_height + self.array_width_proj - 2
            KVmap_cycles = KVmap_cycle_each_fold * KVmap_col_fold * KVmap_row_fold

            cycle_result += 2 * KVmap_cycles


        # Multi-head attention操作
        if part in ['all', 'attn']:
            QKT_row_fold = math.ceil(layer_input/self.array_height)
            QKT_col_fold = math.ceil(kv_len/self.array_width_half)
            QKT_cycle_each_fold = self.head_dim + self.array_height + self.array_width_half - 2
            QKT_cycles = QKT_cycle_each_fold * QKT_col_fold * QKT_row_fold

            cycle_result += QKT_cycles

            SFMXxV_row_fold = math.ceil(layer_input/self.array_height)
            SFMXxV_col_fold = math.ceil(self.head_dim/self.array_width_half)
            SFMXxV_cycle_each_fold = kv_len + self.array_height + self.array_width_half - 2
            SFMXxV_cycles = SFMXxV_cycle_each_fold * SFMXxV_col_fold * SFMXxV_row_fold

            cycle_result += SFMXxV_cycles

        if part in ['all', 'omap']:
            # 输出映射
            Omap_row_fold = math.ceil(layer_input/self.array_height)
            Omap_col_fold = math.ceil(self.dim/self.array_width_proj)
            Omap_cycle_each_fold = self.dim + self.array_height + self.array_width_proj - 2
            Omap_cycles = Omap_cycle_each_fold * Omap_col_fold * Omap_row_fold

            cycle_result += Omap_cycles


        if part in ['all', 'ffn_up']:
            if which_ffn == 'without_img':
                layer = layer_input * (1 - self.text_only_sim)
            elif which_ffn == 'without_text':
                layer = layer_input * (1 - self.image_only_sim)
            else:
                layer = layer_input
            FFN_up_row_fold = math.ceil(layer/self.array_height)
            FFN_up_col_fold = math.ceil(self.upshape/self.array_width_ffn)
            FFN_up_cycle_each_fold = self.dim + self.array_height + self.array_width_ffn - 2
            FFN_up_cycles = FFN_up_cycle_each_fold * FFN_up_col_fold * FFN_up_row_fold

            cycle_result += FFN_up_cycles * 2

        if part in ['all', 'ffn_down']:
            FFN_down_row_fold = math.ceil(layer_input/self.array_height)
            FFN_down_col_fold = math.ceil(self.upshape/self.array_width_ffn)
            FFN_down_cycle_each_fold = self.dim + self.array_height + self.array_width_ffn - 2
            FFN_down_cycles = FFN_down_cycle_each_fold * FFN_down_col_fold * FFN_down_row_fold

            cycle_result += FFN_down_cycles

        if part in ['all', 'drain']:

            drain_cycles = self.active_rate * self.ofmapbufsz * 1024 / self.filterbw
            cycle_result += drain_cycles

        return cycle_result


    def run_gen_image(self):
        self._append_to_log(f"=========== Image Generation Started (total steps: {self.gen_image_step}) ===========")
        image_start_cycles = self.total_cycles_all

        # need-9 disagg: the whole image stage runs on the vision-accel. The
        # text stage already used config_comm0 = text-accel cfg; now route the
        # image-stage text-side qkv strip (and its precision sniff) to the
        # vision-accel cfg so it doesn't run on the tiny text array. Image
        # array widths already come from JSON array_width (vision width).
        if self.hardware_type == 'disagg':
            self.config_comm0 = self.config_comp0

        # Energy: per-operand precision for the image stage (§5).
        #   qkv/omap weights are NOT quantized on any hw (fp16);
        #   ours with T3 splits FFN weights INT4+BF16 across the dual array
        #   (weights 0.5B, MACs billed 50/50 fp16+int4);
        #   figna keeps W4A16 in the image stage too (int4 weights+MACs,
        #   fp16 activations).
        from simulation_core.energy_accounting.coefficients import precision_from_config_path
        prec_comm0 = precision_from_config_path(self.config_comm0)  # qkv text part
        prec_comp0 = precision_from_config_path(self.config_comp0)  # image / attn part
        if self.hardware_type == 'ours':
            if self.quant_proj:
                img_proj_kwargs = {"precision": "fp16", "weight_precision": "int4",
                                   "mac_split": {"w4a16": 1.0}}
            else:
                img_proj_kwargs = {"precision": "fp16"}
            if self.t3_soft:
                img_ffn_kwargs = {"precision": "fp16", "weight_precision": "int4",
                                  "mac_split": {"fp16": 0.5, "w4a16": 0.5}}
            else:
                img_ffn_kwargs = {"precision": "fp16"}
        elif self.hardware_type in ('figna', 'axcore'):
            img_proj_kwargs = {"precision": "fp16", "weight_precision": "int4",
                               "mac_split": {"w4a16": 1.0}}
            img_ffn_kwargs = dict(img_proj_kwargs)
        else:
            img_proj_kwargs = {"precision": prec_comp0}
            img_ffn_kwargs = {"precision": prec_comp0}
        # Text-side qkv runs through SCALE-Sim on comm0; its weights follow
        # the same per-operand story as the image-side projections for
        # ours/figna, and the comm0-sniffed precision otherwise.
        if self.hardware_type in ('ours', 'figna', 'axcore'):
            txt_qkv_kwargs = dict(img_proj_kwargs)
        else:
            txt_qkv_kwargs = {"precision": prec_comm0}

        kv_normal_full = self.kv_cache_init + self.gen_text_len + self.image_input_len
        kv_without_img_full = self.kv_cache_without_img + self.gen_text_len + self.image_input_len
        kv_without_text_full = self.kv_cache_without_text + self.gen_text_len + self.image_input_len

        if self.hardware_type == "ours":
            if not self.t1_soft:
                # T1 off: single full-precision path over the full KV on the
                # FP-FP array (the per-path width array_width_half still
                # applies — high-precision attention can't use the INT4 array).
                kv_normal = kv_normal_full
                kv_without_img = kv_without_img_full
                kv_without_text = kv_without_text_full
            elif self.t1_hard:
                # T1 hard on: HSD load-balances the surviving attention work
                # evenly across the two arrays (formerly the 'ours_balenced'
                # variant) -> each path sees half the total kv.
                kv_cross_attn = self.kv_cache_init + self.gen_text_len
                kv_remain_cross_attn = int(kv_cross_attn * self.sparsity_cross_attn)
                kv_normal = int((kv_remain_cross_attn + self.image_input_len) / 2)
                kv_without_img = int(((self.kv_cache_without_img + self.gen_text_len) * self.sparsity_cross_attn + self.image_input_len) / 2)
                kv_without_text = int(((self.kv_cache_without_text + self.gen_text_len) * self.sparsity_cross_attn + self.image_input_len) / 2)
            else:
                # T1 hard off: dispatch only — low- and high-precision paths
                # each run on their own array, makespan = max of the two.
                kv_cross_attn = self.kv_cache_init + self.gen_text_len
                kv_remain_cross_attn = int(kv_cross_attn * self.sparsity_cross_attn)
                kv_low_prec_self_attn = int(self.image_input_len * self.low_precise_self_attn)
                kv_low_prec_attn = kv_remain_cross_attn + kv_low_prec_self_attn
                kv_high_prec_attn = self.image_input_len - kv_low_prec_self_attn
                kv_normal = max(kv_low_prec_attn, kv_high_prec_attn)
                kv_without_img = max(int((self.kv_cache_without_img + self.gen_text_len) * self.sparsity_cross_attn) + kv_low_prec_self_attn, kv_high_prec_attn)
                kv_without_text = max(int((self.kv_cache_without_text + self.gen_text_len) * self.sparsity_cross_attn) + kv_low_prec_self_attn, kv_high_prec_attn)

        elif self.hardware_type == "sdma":
            kv_normal = int(kv_normal_full * self.sparsity_kv)
            kv_without_img = int(kv_without_img_full * self.sparsity_kv)
            kv_without_text = int(kv_without_text_full * self.sparsity_kv)

        elif self.hardware_type == "disagg":
            # T1 (sparse cross-attn) on the SINGLE vision-accel array. The
            # surviving cross-attn KV (× sparsity_cross_attn) plus the full
            # self-attn (image_input_len) all run on one fp16 array — there is
            # no T3 dual-array /2 load-balance (that array sharing IS T3).
            kv_normal = int((self.kv_cache_init + self.gen_text_len) * self.sparsity_cross_attn) + self.image_input_len
            kv_without_img = int((self.kv_cache_without_img + self.gen_text_len) * self.sparsity_cross_attn) + self.image_input_len
            kv_without_text = int((self.kv_cache_without_text + self.gen_text_len) * self.sparsity_cross_attn) + self.image_input_len

        else:
            kv_normal = kv_normal_full
            kv_without_img = kv_without_img_full
            kv_without_text = kv_without_text_full

        if self.task == "GenEdit":
            # 三个不同KV配置的图像生成
            kv_configs = [
                ("full_cache", kv_normal),
                ("without_img", kv_without_img),
                ("without_text", kv_without_text)
            ]
        
        elif self.task == "GenEval":
            kv_configs = [
                ("full_cache", kv_normal),
                ("without_text", kv_without_text)
            ]
        
        else:
            kv_configs = [
                ("full_cache", kv_normal),
            ]

        total_image_cycles = 0

        # Energy bookkeeping (Phase F): set up multiplicities once.
        # Per-image-gen-step contributions:
        #   prefetch / drain      : ×1
        #   per-config blocks     : ×num_layer (inside the for loop, then config_cycles_iter * num_layer)
        #   attn (per config)     : ×num_layer × num_head_q
        # And the whole image_total then gets multiplied by gen_image_step at
        # the end.  We pass that combined multiplicity into _record_energy.
        n_kv_cfg = len(kv_configs)
        n_steps = self.gen_image_step
        # input/output/ffn_down are reused across every kv_config in the loop.
        # Image stage scales x batch_size (Need-6 batch model: already
        # array-saturated, no batching utilization benefit).
        per_image_mult = self.num_layer * n_kv_cfg * n_steps * self.batch_size

        self._append_to_log(f"Image generation - Start prefetching")
        prefetch_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='prefetch')
        self._append_to_log(f"Image generation - End prefetching, total cycles:{prefetch_cycles}")
        # Prefetch is data movement, not compute; we don't model it for energy
        # (it's a throughput estimate, not a real workload). Skip.

        self._append_to_log(f"Image generation - Start qkv mapping")
        # Text-side QKV: 3 GEMMs (Q/K/V). M=text_attn, K=dim, N=dim (Q) or
        # num_head_kv*head_dim (K/V). Tile-scaled (N=64 strip × real_N/tile)
        # instead of full-N — OS dataflow is linear in N, so this is exact but
        # ~56x faster than the full-N SCALE-Sim call.
        kv_proj_N = self.num_head_kv * self.head_dim
        q_strip = self.run_sim_once(kv_length=0, is_gen_text=False, part='qkv_q_strip', config=self.config_comm0)
        kv_strip = self.run_sim_once(kv_length=0, is_gen_text=False, part='qkv_kv_strip', config=self.config_comm0)
        input_cycles_text = (q_strip * (self.dim / self.tile)
                             + 2 * kv_strip * (kv_proj_N / self.tile))
        self._record_energy(M=self.text_attn, N=self.dim, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_text_Q",
                            hw_cfg_path=self.config_comm0, **txt_qkv_kwargs)
        self._record_energy(M=self.text_attn, N=kv_proj_N, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_text_K",
                            hw_cfg_path=self.config_comm0, **txt_qkv_kwargs)
        self._record_energy(M=self.text_attn, N=kv_proj_N, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_text_V",
                            hw_cfg_path=self.config_comm0, **txt_qkv_kwargs)

        input_cycles_image = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='qkv')
        # Image-side QKV from run_sim_once_comp: Q (vae_attn × dim × dim) + 2*KV
        # (vae_attn × num_head_kv*head_dim × dim).
        self._record_energy(M=self.vae_attn, N=self.dim, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_image_Q",
                            hw_cfg_path=self.config_comp0, **img_proj_kwargs)
        self._record_energy(M=self.vae_attn, N=kv_proj_N, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_image_K",
                            hw_cfg_path=self.config_comp0, **img_proj_kwargs)
        self._record_energy(M=self.vae_attn, N=kv_proj_N, K=self.dim,
                            multiplicity=per_image_mult, label="img/qkv_image_V",
                            hw_cfg_path=self.config_comp0, **img_proj_kwargs)

        input_cycles = input_cycles_text + input_cycles_image
        self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")

        self._append_to_log(f"Image generation - Start output mapping")
        output_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='omap')
        # Omap: M=image_input_len, N=dim, K=dim
        self._record_energy(M=self.image_input_len, N=self.dim, K=self.dim,
                            multiplicity=per_image_mult, label="img/omap",
                            hw_cfg_path=self.config_comp0, **img_proj_kwargs)
        self._append_to_log(f"Image generation - End output mapping, total cycles:{output_cycles}")

        self._append_to_log(f"Image generation - Start FFN down")
        ffn_down_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='ffn_down')
        # FFN-down: M=image_input_len, N=dim, K=upshape
        self._record_energy(M=self.image_input_len, N=self.dim, K=self.upshape,
                            multiplicity=per_image_mult, label="img/ffn_down",
                            hw_cfg_path=self.config_comp0, **img_ffn_kwargs)
        self._append_to_log(f"Image generation - End FFN down, total cycles:{ffn_down_cycles}")

        self._append_to_log(f"Image generation - Start draining")
        drain_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='drain')
        self._append_to_log(f"Image generation - End draining, total cycles:{drain_cycles}")
        # Drain is a throughput proxy; no energy model for it.

        total_image_cycles += prefetch_cycles
        self._add_op("img/prefetch", prefetch_cycles * n_steps)

        for config_name, kv_len in kv_configs:
            self._append_to_log(f"Image generation - {config_name} config started (kv_len: {kv_len})")

            # Which sim ratio shrinks this branch's ffn_up. On 3-branch tasks
            # (GenEdit) the full_cache branch also reuses FFN results, billed
            # at the text_only_sim ratio (same recipe as without_img) — user
            # decision 2026-06-12, applies to every hw with T2 (ours,
            # flightvgm; base/sdma/figna have *_sim=0 so this is a no-op).
            # t2_soft off (ARGUS ablation) disables all reuse.
            ffn_branch = config_name
            if config_name == "full_cache" and len(kv_configs) == 3:
                ffn_branch = "without_img"
            if not self.t2_soft:
                ffn_branch = "no_reuse"

            config_cycles_iter = 0
            config_cycles_iter += input_cycles
            attn_single_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='attn')
            attn_cycles = attn_single_cycles * self.num_head_q
            config_cycles_iter += attn_cycles
            config_cycles_iter += output_cycles
            ffn_up_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='ffn_up',which_ffn=ffn_branch)
            config_cycles_iter += ffn_up_cycles
            config_cycles_iter += ffn_down_cycles
            config_cycles = config_cycles_iter * self.num_layer
            total_image_cycles += config_cycles

            # Per-operator breakdown (× num_layer × n_steps; final image total
            # is × gen_image_step = n_steps).
            opmul = self.num_layer * n_steps
            self._add_op("img/qkv", input_cycles * opmul)
            self._add_op("img/attn", attn_cycles * opmul)
            self._add_op("img/omap", output_cycles * opmul)
            self._add_op("img/ffn_up", ffn_up_cycles * opmul)
            self._add_op("img/ffn_down", ffn_down_cycles * opmul)

            # Energy: per-config attention + ffn_up.
            #   attn: M=image_input_len, two GEMMs (kv_len, head_dim) and
            #         (head_dim, kv_len). Per-config multiplicity =
            #         num_layer × num_head_q × gen_image_step.
            #   ffn_up: depends on which_ffn (image_only/text_only mask).
            attn_mult = self.num_layer * self.num_head_q * n_steps * self.batch_size
            self._record_energy(M=self.image_input_len, N=kv_len, K=self.head_dim,
                                precision=prec_comp0, multiplicity=attn_mult,
                                label=f"img/attn_qk({config_name},kv={kv_len})",
                                hw_cfg_path=self.config_comp0)
            self._record_energy(M=self.image_input_len, N=self.head_dim, K=kv_len,
                                precision=prec_comp0, multiplicity=attn_mult,
                                label=f"img/attn_sfmxv({config_name},kv={kv_len})",
                                hw_cfg_path=self.config_comp0)

            # FFN-up has variable layer_input (ffn_branch dampens it via
            # image_only_sim / text_only_sim). We mirror the cycle calc.
            if ffn_branch == "without_img":
                layer_eff = self.image_input_len * (1 - self.text_only_sim)
            elif ffn_branch == "without_text":
                layer_eff = self.image_input_len * (1 - self.image_only_sim)
            else:
                layer_eff = self.image_input_len
            ffn_up_mult = self.num_layer * n_steps * self.batch_size  # per-config, no num_head
            self._record_energy(M=int(layer_eff), N=self.upshape, K=self.dim,
                                multiplicity=ffn_up_mult,
                                label=f"img/ffn_up_gate({config_name})",
                                hw_cfg_path=self.config_comp0, **img_ffn_kwargs)
            # ARGUS multiplies ffn_up_cycles by 2 in the cycle calc (gate +
            # up projections), so we record ffn_up's energy twice.
            self._record_energy(M=int(layer_eff), N=self.upshape, K=self.dim,
                                multiplicity=ffn_up_mult,
                                label=f"img/ffn_up_proj({config_name})",
                                hw_cfg_path=self.config_comp0, **img_ffn_kwargs)

            self._append_to_log(f"Image generation - {config_name} config completed, cycles: {config_cycles}")

        total_image_cycles += drain_cycles
        self._add_op("img/drain", drain_cycles * n_steps)

        # Need-6 batch model: the image stage is already array-saturated
        # (M=image_input_len=1378 >> array_height), so batching gives no
        # utilization benefit -> the whole stage simply scales x batch_size.
        # Scale the per-operator breakdown too so it stays consistent with
        # total_cycles_all.
        if self.batch_size != 1:
            for k in list(self.cycle_breakdown):
                if k.startswith("img/"):
                    self.cycle_breakdown[k] *= self.batch_size

        # 乘以生成步数
        final_image_cycles = total_image_cycles * self.gen_image_step * self.batch_size
        self.total_cycles_all += final_image_cycles
        
        image_total_cycles = self.total_cycles_all - image_start_cycles
        self._append_to_log(f"=========== Image Generation Completed, total cycles: {image_total_cycles}, total seconds: {image_total_cycles/500000000} ===========")


    def run_model(self):
        """运行完整的Bagel模型仿真"""
        # 清空之前的日志文件。result_path 是文件路径（与 _append_to_log 一致），不是目录——
        # 历史上这里写成 os.path.join(result_path, "results.log") 得到一个永不存在的路径，
        # os.remove 从不执行 → 日志被 append 污染。直接用 result_path 即可。
        log_file = self.result_path
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # 记录仿真开始
        self._append_to_log("======= Bagel Model Simulation Started =======")
        self._append_to_log(f"Configuration: text_len={self.gen_text_len}, image_steps={self.gen_image_step}, layers={self.num_layer}, batch_size={self.batch_size}")

        start_time = datetime.now()

        # 重置总周期数
        self.total_cycles_all = 0
        self.cycle_breakdown = {}

        # 运行文本生成
        if self.text_gen_finished_flag:
            self._append_to_log(f"=========== Text Generation Started (total steps: {self.gen_text_len}) ===========")
            text_total_cycles = self.text_gen_cycles
            self.total_cycles_all += text_total_cycles
            self._append_to_log(f"=========== Text Generation Completed, total cycles: {text_total_cycles}, total seconds: {text_total_cycles/500000000} ===========")
        else:
            # 运行文本生成
            self.run_gen_text()

        # Per-stage cycle split (total_cycles_all started at 0). Needed by the
        # need-9 disagg idle-leakage charge below.
        text_cycles_total = self.total_cycles_all
        image_cycles_total = 0

        # need-9 per-stage utilization: snapshot the energy ledger at the
        # text->image boundary so MAC ops / DRAM bytes can be attributed to
        # each stage by differencing. Read-only; no effect when energy is off.
        snap_text = self.energy.snapshot() if self.energy is not None else None

        # 运行图像生成 (GenEval 也是生成类任务，含图像阶段；run_gen_image
        # 已为 GenEval 准备了 full_cache/without_text 两分支。2026-06-12 修
        # 核对项7：此前 GenEval 漏跑图像阶段。MM 是纯理解任务，无图像阶段。)
        if self.task in ("GenEdit", "GenImage", "GenEval"):
            self.run_gen_image()
            image_cycles_total = self.total_cycles_all - text_cycles_total
        snap_all = self.energy.snapshot() if self.energy is not None else None
        
        # 记录最终结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self._append_to_log("=" * 50)
        self._append_to_log(f"FINAL RESULT - Total Cycles: {self.total_cycles_all}, total seconds: {self.total_cycles_all/500000000}")
        self._append_to_log(f"Simulation Duration: {duration:.2f} seconds")
        self._append_to_log(self._format_cycle_breakdown())

        # Energy report (Phase F). Only printed when --energy / energy_enabled
        # was set; otherwise self.energy is None and we skip silently.
        if self.energy_enabled and self.energy is not None:
            self.energy.add_cycles(int(self.total_cycles_all))

            # need-9 per-stage utilization (whole-chip 2048 basis, user-chosen
            # 2026-06-17). Compute = MAC / (PE_total × cycles); Mem = achieved
            # off-chip BW / peak. Per-stage MAC and DRAM bytes come from
            # differencing the two ledger snapshots taken around run_gen_image.
            # The cycle-weighted sum of the two compute utils equals the legacy
            # aggregate util, so this only refines the existing number.
            if snap_text is not None and snap_all is not None:
                pe = self.pe_total or (self.array_height * self.array_width)
                freq = self.energy.coef["freq_hz"]
                peak_bpc = self.peak_dram_gbps * 1e9 / freq   # bytes / cycle
                text_mac = snap_text["mac"]
                image_mac = snap_all["mac"] - snap_text["mac"]
                text_bytes = snap_text["dram_bytes"]
                image_bytes = snap_all["dram_bytes"] - snap_text["dram_bytes"]

                def _cmp_util(mac, cyc):
                    return 100.0 * mac / (pe * cyc) if (pe and cyc) else 0.0

                def _mem_util(byts, cyc):
                    return 100.0 * (byts / cyc) / peak_bpc if (cyc and peak_bpc) else 0.0

                tc = _cmp_util(text_mac, text_cycles_total)
                im_c = _cmp_util(image_mac, image_cycles_total)
                tm = _mem_util(text_bytes, text_cycles_total)
                im_m = _mem_util(image_bytes, image_cycles_total)
                self._append_to_log(
                    f"STAGE UTIL (PE_total={pe}, peak={self.peak_dram_gbps}GB/s={peak_bpc:.1f}B/cyc): "
                    f"text_compute={tc:.2f}% text_mem={tm:.2f}% "
                    f"image_compute={im_c:.2f}% image_mem={im_m:.2f}%")
                self._append_to_log(
                    f"STAGE RAW: text_cyc={text_cycles_total} image_cyc={image_cycles_total} "
                    f"text_mac={text_mac} image_mac={image_mac} "
                    f"text_dram_bytes={text_bytes:.0f} image_dram_bytes={image_bytes:.0f}")

                # Detailed per-operator utilization (paper Fig.14 style):
                # text stage attn/ffn/avg MEMORY util; image stage attn/ffn/avg
                # COMPUTE util; whole-run compute + memory util.
                cb = self.cycle_breakdown
                om = self.energy.op_mac
                ob = self.energy.op_dram_bytes
                def cyc_of(*keys):
                    return sum(cb.get(k, 0) for k in keys)
                def cu(mac, cyc):
                    return 100.0 * mac / (pe * cyc) if (pe and cyc) else 0.0
                def mu(byts, cyc):
                    return 100.0 * (byts / cyc) / peak_bpc if (cyc and peak_bpc) else 0.0
                t_attn_c = cyc_of("text/attn")
                t_ffn_c  = cyc_of("text/ffn_up", "text/ffn_down")
                i_attn_c = cyc_of("img/attn")
                i_ffn_c  = cyc_of("img/ffn_up", "img/ffn_down")
                self._append_to_log(
                    "DETAIL UTIL (Fig14-style) -- "
                    f"TEXT mem%: attn={mu(ob.get('text/attn',0), t_attn_c):.2f} "
                    f"ffn={mu(ob.get('text/ffn',0), t_ffn_c):.2f} "
                    f"avg={mu(text_bytes, text_cycles_total):.2f} | "
                    f"IMAGE compute%: attn={cu(om.get('img/attn',0), i_attn_c):.2f} "
                    f"ffn={cu(om.get('img/ffn',0), i_ffn_c):.2f} "
                    f"avg={cu(image_mac, image_cycles_total):.2f} | "
                    f"TOTAL: compute={cu(snap_all['mac'], self.total_cycles_all):.2f} "
                    f"mem={mu(snap_all['dram_bytes'], self.total_cycles_all):.2f}")

            # Compute-array static power (leakage + clock) is now charged
            # uniformly for EVERY hardware as array_static_mw × wall-time inside
            # the accountant (compute_array_static_pJ) — a slower design pays
            # more because it holds the 2048-lane array longer. This subsumes
            # the old disagg-only idle-leakage special case (the whole chip
            # leaks regardless of which lanes are active), so no per-hw charge
            # is needed here. See REBUTTAL_FRAMEWORK.md §6.5 / coefficients.py.

            self._append_to_log(self.energy.format_report())

            # F7: optional accelergy cross-validation on top-N sub-runs.
            if self.energy_validate:
                self._append_to_log("Running F7 cross-validation (this may take ~10-30s)...")
                from simulation_core.energy_accounting.validate import (
                    cross_validate_top_records, format_validation_report
                )
                top = self.energy.top_records_by_energy(n=3)
                results = cross_validate_top_records(
                    records=top,
                    coefficients=self.energy.coef,
                    ours_log_dir=self.log_path,
                    argus_root=PROJECT_ROOT,
                )
                self._append_to_log(format_validation_report(results))
        self._append_to_log("======= Bagel Model Simulation Completed =======")
        
        return self.total_cycles_all


def main():
    """主函数"""
    print("Bagel Model Simulation Starting...")
    
    # 实例化Bagel仿真类
    bagel = Bagel_sim()
    
    # 读取JSON配置
    bagel.read_from_json()
    
    # 运行完整模型仿真
    total_cycles = bagel.run_model()
    
    print(f"\nSimulation completed successfully!")
    print(f"Total cycles: {total_cycles}")
    print(f"Results saved to: {bagel.result_path}/results.log")


if __name__ == "__main__":
    main()