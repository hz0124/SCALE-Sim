import os
import json
import csv
import math
from datetime import datetime
from scalesim.scale_sim import scalesim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Janus_sim():
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
        # Text-stage FFN slot (see Bagel_sim): ARGUS points it at the
        # weight-INT8-storage cfg; other hardware reuses its projection cfg.
        self.config_ffn_text = os.path.join(PROJECT_ROOT, "configs", "scale.cfg")
        self.log_path = os.path.join(PROJECT_ROOT, "results")
        self.result_path = os.path.join(PROJECT_ROOT, "results", "bagel")
        self.num_layer = 24
        self.text_input_len = 1
        self.image_input_len = 1378
        self.text_attn = 2
        self.vae_attn = 1376
        self.num_head_kv = 16
        self.num_head_q = 16
        self.dim = 2048
        self.head_dim = 128
        self.upshape = 5632
        self.tile = 64
        self.total_cycles_all = 0
        # Per-operator cycle breakdown (req 3); see Bagel_sim.
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
        # Peak off-chip DRAM bandwidth (GB/s) for the analytical memory-bound
        # text model (mirrors Bagel_sim). Default = aligned 32 GB/s.
        self.peak_dram_gbps = 32.0
        self.quant_proj = False
        # ARGUS technique switches (REBUTTAL_FRAMEWORK.md §4.5); mirrors
        # Bagel_sim. t2_hard is a placeholder (SAU overhead unmodelled).
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
        # CLI --energy or JSON "energy_enabled": true. Mirrors Bagel_sim;
        # see simulation_core/energy_accounting/.
        self.energy_enabled = False
        self.energy = None
        self.dram_type = None
        self.coef_overrides = None
        self.energy_validate = False    # set by --validate


    def setup_energy(self, hw_type=None):
        """Instantiate self.energy. Idempotent."""
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
        """Add one sub-run's MAC + SRAM + DRAM access to self.energy.
        See Bagel_sim._record_energy for full docstring; identical semantics
        (incl. weight_precision for storage-only low precision and mac_split
        for splitting/forcing MAC precision). Access counts come from the
        closed-form geometric estimate (the tiled SCALE-Sim report path was
        removed — its DRAM counts are prefetch-buffer artifacts, see DEVLOG #4).
        """
        if not self.energy_enabled or self.energy is None:
            return
        if multiplicity <= 0:
            return
        from simulation_core.energy_accounting.extractors import from_geometric_estimate

        total_factor = float(scale) * float(multiplicity)
        before_pJ = self.energy.total_pJ() if (label or hw_cfg_path) else 0.0
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

        _bytes = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
        act_bytes = _bytes.get(precision, 2.0)
        w_bytes = _bytes.get(weight_precision, act_bytes) if weight_precision else act_bytes

        for kind in ("ifmap", "filter", "ofmap"):
            word_bytes = w_bytes if kind == "filter" else act_bytes
            if kind == "ofmap":
                self.energy.add_sram_access(
                    kind, words=counts["sram_ofmap_writes"],
                    word_bytes=word_bytes, op="write")
                self.energy.add_dram_access(
                    kind, words=counts["dram_ofmap_writes"],
                    word_bytes=word_bytes, op="write")
            else:
                self.energy.add_sram_access(
                    kind, words=counts[f"sram_{kind}_reads"],
                    word_bytes=word_bytes, op="read")
                self.energy.add_dram_access(
                    kind, words=counts[f"dram_{kind}_reads"],
                    word_bytes=word_bytes, op="read")

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
            cfg_path = os.path.join(PROJECT_ROOT, "topologies", "janus", "config.json")
        
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
        # ARGUS projection-quantization extension (framework §15), default off.
        # See Bagel_sim. Only affects ours.
        self.quant_proj = bool(cfg.get("quant_proj", False))

        if self.hardware_type == 'base' or self.hardware_type == 'flightvgm' or self.hardware_type == 'sdma':
            self.config_comp0 = cfg.get("config", self.config_comp0)
            self.config_comp1 = cfg.get("config", self.config_comp1)
            self.config_comm0 = cfg.get("config", self.config_comm0)
            self.config_comm1 = cfg.get("config", self.config_comm1)
            self.config_ffn_text = cfg.get("config", self.config_ffn_text)
        elif self.hardware_type == 'ours':
            # No FP-INT8 compute unit: text stage all on the FP-FP array,
            # INT8 is a storage format for FFN weights only (see Bagel_sim).
            self.config_comp0 = cfg.get("config_fp16", self.config_comp0)
            self.config_comp1 = cfg.get("config_int4", self.config_comp1)
            self.config_comm0 = cfg.get("config_fp16", self.config_comm0)
            self.config_comm1 = cfg.get("config_fp16", self.config_comm1)
            if self.quant_proj:
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
        # FP-FP-only width for image-stage qkv/omap (weights unquantized);
        # FFN keeps the dual-array width only when T3 is on. Mirrors Bagel_sim.
        self.array_width_fp = cfg.get("array_width_fp", self.array_width)
        # Image-stage qkv/omap width: full dual array with quant_proj, else
        # FP-FP only. Mirrors Bagel_sim.
        if self.hardware_type == 'ours' and self.quant_proj:
            self.array_width_proj = self.array_width
        else:
            self.array_width_proj = self.array_width_fp
        if self.hardware_type == 'ours' and not self.t3_soft:
            self.array_width_ffn = self.array_width_fp
        else:
            self.array_width_ffn = self.array_width
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

        # Energy accounting (Phase F)
        if cfg.get("energy_enabled", False):
            self.energy_enabled = True
        self.dram_type = cfg.get("dram_type", self.dram_type)
        self.coef_overrides = cfg.get("energy_coefficients", self.coef_overrides)

    def build_topologies(self, kv_len=0, is_gen_text=True, part='all'):
        """
        生成Janus模型单层拓扑结构，并保存到相应的csv文件中

        Args:
        kv_len: KV cache长度，影响attention矩阵大小
        is_gen_text: True为文本生成，False为图像生成
        part: 指定生成哪个部分
        """
        # 创建输出目录
        output_dir = os.path.join(PROJECT_ROOT, "topologies", "bagel")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "layer.csv")
        
        # 生成矩阵操作列表
        operations = []
        layer_input = self.text_input_len if is_gen_text else self.image_input_len

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

    def run_sim_once(self, kv_length=0, is_gen_text=True, part='all', config="./configs/scale.cfg"):
        topology_path = self.build_topologies(kv_len=kv_length, is_gen_text=is_gen_text, part=part)

        s = scalesim(
            save_disk_space=True,
            verbose=True,   # NB: upstream couples verbose into the result —
                            # verbose=False makes run_scale return 0 cycles.
            config=config,
            topology=topology_path,
            layout=os.path.join(PROJECT_ROOT, "layouts", "GEMM_mnk", "vit_l_KM_KN.csv"),
            input_type_gemm=True
        )

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
        """Per-operator cycle breakdown + % of end-to-end (req 3). See Bagel_sim."""
        cb = self.cycle_breakdown
        total = sum(cb.values())
        if total <= 0:
            return "CYCLE BREAKDOWN: none recorded"
        lines = ["CYCLE BREAKDOWN (per-operator cycles / % of end-to-end):"]
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

    # --- analytical memory-bound text-decode model (mirrors Bagel_sim, 2026-06-17).
    # SCALE-Sim's os mapping wastes ~31/32 of the array for M=1 decode, hiding all
    # memory traffic so no bandwidth knob binds. Physically a well-mapped M=1 GEMV
    # fills the array and is then bounded by streaming its weights/KV from DRAM. We
    # model each text GEMM as max(ideal-fill compute, operand stream). See
    # devlog_rebuttal/docs/NEED9_DISAGG.md §6. Janus has no GQA (num_head_q==num_head_kv)
    # and no disagg hw type, so the KV-reuse caveat and disagg branch don't apply.
    _PREC_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}

    def _text_array_macs(self):
        """MACs/cycle the text stage can use (array assumed fully mapped for M=1)."""
        h = self.array_height or 32
        if self.hardware_type == 'ours':
            # text runs on the FP-FP sub-array only (FP-INT4 array idle).
            return h * (self.array_width_fp or self.array_width)
        return h * self.array_width

    def _text_bw_bpc(self):
        """Off-chip bytes/cycle for the text stage (peak_dram_gbps @ 500 MHz)."""
        return self.peak_dram_gbps * 1e9 / 500_000_000.0

    def _text_gemm_cycles(self, M, N, K, w_bytes, bw_bpc, array_macs, act_bytes=None):
        """One M-row text GEMM: max(ideal-fill compute, operand-stream).
        compute = M*N*K / array_macs;
        stream = (N*K*w_bytes + M*K*act + M*N*act)/bw_bpc — all off-chip operands
        once (weights/KV dominate for M=1; activation terms keep mem-util ≤100%)."""
        if act_bytes is None: act_bytes = w_bytes
        compute = (M * N * K) / array_macs
        stream = (N * K * w_bytes + M * K * act_bytes + M * N * act_bytes) / bw_bpc
        return max(compute, stream)

    def run_gen_text(self):
        self._append_to_log(f"=========== Text Generation Started (total steps: {self.gen_text_len}) ===========")
        text_start_cycles = self.total_cycles_all

        # Energy: per-operand precision + once-only multiplicities (mirrors
        # Bagel_sim.run_gen_text — see the comment there).
        from simulation_core.energy_accounting.coefficients import precision_from_config_path
        prec_comm0 = precision_from_config_path(self.config_comm0)
        prec_comm1 = precision_from_config_path(self.config_comm1)
        if self.hardware_type == 'ours':
            if self.quant_proj:
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
        elif self.hardware_type in ('figna', 'axcore'):
            proj_kwargs = {"precision": "fp16", "weight_precision": "int4",
                           "mac_split": {"w4a16": 1.0}}
            ffn_kwargs = dict(proj_kwargs)
        else:
            proj_kwargs = {"precision": prec_comm0}
            ffn_kwargs = {"precision": precision_from_config_path(self.config_ffn_text)}
        once_mult = self.num_layer * self.gen_text_len
        qkv_N = self.dim + 2 * self.num_head_kv * self.head_dim

        # Analytical memory-bound text model (mirrors Bagel_sim): cycles per GEMM =
        # max(ideal-fill compute, operand-stream/BW). Replaces the SCALE-Sim os runs.
        bw_bpc = self._text_bw_bpc()
        tmacs = self._text_array_macs()
        proj_wb = self._PREC_BYTES.get(
            proj_kwargs.get("weight_precision") or proj_kwargs.get("precision", "fp16"), 2.0)
        ffn_wb = self._PREC_BYTES.get(
            ffn_kwargs.get("weight_precision") or ffn_kwargs.get("precision", "fp16"), 2.0)
        kv_wb = self._PREC_BYTES.get(prec_comm1, 2.0)   # KV cache at attn precision
        proj_ab = self._PREC_BYTES.get(proj_kwargs.get("precision","fp16"),2.0)
        ffn_ab = self._PREC_BYTES.get(ffn_kwargs.get("precision","fp16"),2.0)

        self._append_to_log(f"Text generation - Start qkv mapping")
        input_cycles = self._text_gemm_cycles(1, qkv_N, self.dim, proj_wb, bw_bpc, tmacs, proj_ab)
        self._record_energy(M=1, N=qkv_N, K=self.dim,
                            multiplicity=once_mult, label="text/qkv",
                            hw_cfg_path=self.config_comm0, **proj_kwargs)
        self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")

        self._append_to_log(f"Text generation - Start output mapping")
        output_cycles = self._text_gemm_cycles(1, self.dim, self.dim, proj_wb, bw_bpc, tmacs, proj_ab)
        self._record_energy(M=1, N=self.dim, K=self.dim,
                            multiplicity=once_mult, label="text/omap",
                            hw_cfg_path=self.config_comm0, **proj_kwargs)
        self._append_to_log(f"Text generation - End output mapping, total cycles:{output_cycles}")

        self._append_to_log(f"Text generation - Start FFN up")
        # Gated FFN (Janus): gate + up are two same-shape projections, so ×2.
        ffn_up_cycles = self._text_gemm_cycles(1, self.upshape, self.dim, ffn_wb, bw_bpc, tmacs, ffn_ab) * 2
        self._record_energy(M=1, N=self.upshape, K=self.dim, scale=2.0,
                            multiplicity=once_mult, label="text/ffn_up_gate+up",
                            hw_cfg_path=self.config_ffn_text, **ffn_kwargs)
        self._append_to_log(f"Text generation - End FFN up, total cycles:{ffn_up_cycles}")

        self._append_to_log(f"Text generation - Start FFN down")
        ffn_down_cycles = self._text_gemm_cycles(1, self.dim, self.upshape, ffn_wb, bw_bpc, tmacs, ffn_ab)
        self._record_energy(M=1, N=self.dim, K=self.upshape,
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
            # Attention (M=1 per query): reads K/V cache from DRAM (kv_len×head_dim
            # each, at kv precision) — KV-bandwidth-bound. Mirrors Bagel_sim.
            attn_qk_single_cycle = self._text_gemm_cycles(1, kv_len, self.head_dim, kv_wb, bw_bpc, tmacs, kv_wb)
            attn_sfmxv_single_cycle = self._text_gemm_cycles(1, self.head_dim, kv_len, kv_wb, bw_bpc, tmacs, kv_wb)
            attn_single_cycle = attn_qk_single_cycle + attn_sfmxv_single_cycle
            attn_cycles = attn_single_cycle * self.num_head_q
            results_this_iter += attn_cycles
            results_this_iter += output_cycles
            results_this_iter += ffn_up_cycles
            results_this_iter += ffn_down_cycles

            step_cycles = results_this_iter * self.num_layer

            step_cnts = step_end - step + 1
            step_cycles_total = step_cycles * step_cnts
            self.total_cycles_all += step_cycles_total

            opmul = self.num_layer * step_cnts
            self._add_op("text/qkv", input_cycles * opmul)
            self._add_op("text/attn", attn_cycles * opmul)
            self._add_op("text/omap", output_cycles * opmul)
            self._add_op("text/ffn_up", ffn_up_cycles * opmul)
            self._add_op("text/ffn_down", ffn_down_cycles * opmul)

            # Energy: per-step attention (multiplicity = num_layer × step_cnts × num_head_q).
            attn_mult = self.num_layer * step_cnts * self.num_head_q
            self._record_energy(M=1, N=kv_len, K=self.head_dim,
                                precision=prec_comm1, multiplicity=attn_mult,
                                label=f"text/attn_qk(kv={kv_len})", hw_cfg_path=self.config_comm1)
            self._record_energy(M=1, N=self.head_dim, K=kv_len,
                                precision=prec_comm1, multiplicity=attn_mult,
                                label=f"text/attn_sfmxv(kv={kv_len})", hw_cfg_path=self.config_comm1)


        text_total_cycles = self.total_cycles_all - text_start_cycles
        self._append_to_log(f"=========== Text Generation Completed, total cycles: {text_total_cycles}, total seconds: {text_total_cycles/500000000} ===========")

    def run_sim_once_comp(self, kv_len=0, is_gen_text=False, part='all', which_ffn = 'full_cache', image_input=0):
        cycle_result = 0
        layer_input = self.text_input_len if is_gen_text else image_input
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
        # (the INT4+BF16 dual-array merge under T3). Mirrors
        # Bagel_sim.run_sim_once_comp.
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

        kv_normal_full = self.kv_cache_init + self.gen_text_len + self.image_input_len
        kv_without_text_full = self.image_input_len

        if self.hardware_type == "ours":
            if not self.t1_soft:
                # T1 off: single full-precision path over the full KV.
                kv_normal = kv_normal_full
                kv_without_text = kv_without_text_full
            elif self.t1_hard:
                # T1 hard on: HSD load-balances the surviving attention work
                # evenly across the two arrays (formerly 'ours_balenced').
                kv_cross_attn = self.kv_cache_init + self.gen_text_len
                kv_remain_cross_attn = int(kv_cross_attn * self.sparsity_cross_attn)
                kv_normal = int((kv_remain_cross_attn + self.image_input_len) / 2)
                kv_without_text = int(((self.kv_cache_without_text + self.gen_text_len) * self.sparsity_cross_attn + self.image_input_len) / 2)
            else:
                # T1 hard off: dispatch only, makespan = max(two paths).
                kv_cross_attn = self.kv_cache_init + self.gen_text_len
                kv_remain_cross_attn = int(kv_cross_attn * self.sparsity_cross_attn)
                kv_low_prec_self_attn = int(self.image_input_len * self.low_precise_self_attn)
                kv_low_prec_attn = kv_remain_cross_attn + kv_low_prec_self_attn
                kv_high_prec_attn = self.image_input_len - kv_low_prec_self_attn
                kv_normal = max(kv_low_prec_attn, kv_high_prec_attn)
                kv_without_text = max(int((self.kv_cache_without_text + self.gen_text_len) * self.sparsity_cross_attn) + kv_low_prec_self_attn, kv_high_prec_attn)

        elif self.hardware_type == "sdma":
            kv_normal = int(kv_normal_full * self.sparsity_kv)
            kv_without_text = int(kv_without_text_full * self.sparsity_kv)

        else:
            kv_normal = kv_normal_full
            kv_without_text = kv_without_text_full

        input_normal = self.kv_cache_init + self.gen_text_len + self.image_input_len
        input_img_only = self.image_input_len
        
        kv_configs = [
            ("full_cache", kv_normal, input_normal),
            ("without_text", kv_without_text, input_img_only)
        ]

        total_image_cycles = 0

        # Energy bookkeeping (Phase F): Janus's image gen has 2 kv_configs.
        # Each config has its own input_len (full_cache: kv_init + gen_text_len + image_input_len;
        # without_text: image_input_len). Sub-runs inside the for-loop run once per
        # config, so per-config multiplicity = num_layer × gen_image_step.
        from simulation_core.energy_accounting.coefficients import precision_from_config_path
        prec_comp0 = precision_from_config_path(self.config_comp0)
        # Per-operand precision for the image stage (mirrors Bagel_sim):
        # qkv/omap weights unquantized; ours' T3 bills the FFN as INT4+BF16
        # dual-array (weights 0.5B, MACs 50/50); figna stays W4A16.
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
        n_steps = self.gen_image_step
        kv_proj_N = self.num_head_kv * self.head_dim

        self._append_to_log(f"Image generation - Start prefetching")
        prefetch_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='prefetch', image_input=input_normal)
        self._append_to_log(f"Image generation - End prefetching, total cycles:{prefetch_cycles}")

        self._append_to_log(f"Image generation - Start draining")
        drain_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='drain')
        self._append_to_log(f"Image generation - End draining, total cycles:{drain_cycles}")

        total_image_cycles += prefetch_cycles
        self._add_op("img/prefetch", prefetch_cycles * n_steps)

        for config_name, kv_len, input_len in kv_configs:
            self._append_to_log(f"Image generation - {config_name} config started (kv_len: {kv_len})")
            per_cfg_mult = self.num_layer * n_steps
            attn_mult = self.num_layer * self.num_head_q * n_steps

            config_cycles_iter = 0
            self._append_to_log(f"Image generation - Start qkv mapping")
            input_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='qkv', image_input=input_len)
            # qkv: Q (input_len, dim, dim) + 2× KV (input_len, num_head_kv·head_dim, dim)
            self._record_energy(M=input_len, N=self.dim, K=self.dim,
                                multiplicity=per_cfg_mult, label=f"img/qkv_Q({config_name})",
                                hw_cfg_path=self.config_comp0, **img_proj_kwargs)
            self._record_energy(M=input_len, N=kv_proj_N, K=self.dim,
                                multiplicity=per_cfg_mult, label=f"img/qkv_K({config_name})",
                                hw_cfg_path=self.config_comp0, **img_proj_kwargs)
            self._record_energy(M=input_len, N=kv_proj_N, K=self.dim,
                                multiplicity=per_cfg_mult, label=f"img/qkv_V({config_name})",
                                hw_cfg_path=self.config_comp0, **img_proj_kwargs)
            self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")
            config_cycles_iter += input_cycles

            self._append_to_log(f"Image generation - Start attn")
            attn_single_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='attn', image_input=input_len)
            attn_cycles = attn_single_cycles * self.num_head_q
            # attn: 2 GEMMs per head, multiplied by num_head_q in cycle math
            self._record_energy(M=input_len, N=kv_len, K=self.head_dim,
                                precision=prec_comp0, multiplicity=attn_mult,
                                label=f"img/attn_qk({config_name},kv={kv_len})",
                                hw_cfg_path=self.config_comp0)
            self._record_energy(M=input_len, N=self.head_dim, K=kv_len,
                                precision=prec_comp0, multiplicity=attn_mult,
                                label=f"img/attn_sfmxv({config_name},kv={kv_len})",
                                hw_cfg_path=self.config_comp0)
            self._append_to_log(f"Image generation - End attn, total cycles:{attn_cycles}")
            config_cycles_iter += attn_cycles

            self._append_to_log(f"Image generation - Start output mapping")
            output_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='omap', image_input=input_len)
            self._record_energy(M=input_len, N=self.dim, K=self.dim,
                                multiplicity=per_cfg_mult, label=f"img/omap({config_name})",
                                hw_cfg_path=self.config_comp0, **img_proj_kwargs)
            self._append_to_log(f"Image generation - End output mapping, total cycles:{output_cycles}")
            config_cycles_iter += output_cycles

            self._append_to_log(f"Image generation - Start FFN up")
            # Janus only runs GenEval (2 branches: full_cache + without_text),
            # so there is no 3-branch full_cache reuse extension as in Bagel.
            # t2_soft off (ARGUS ablation): no reuse shrink on any branch.
            ffn_branch = config_name if self.t2_soft else "no_reuse"
            ffn_up_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='ffn_up',which_ffn=ffn_branch, image_input=input_len)
            # ffn_up: layer_input depends on which_ffn mask, recorded twice (gate + up).
            if ffn_branch == "without_img":
                layer_eff = input_len * (1 - self.text_only_sim)
            elif ffn_branch == "without_text":
                layer_eff = input_len * (1 - self.image_only_sim)
            else:
                layer_eff = input_len
            self._record_energy(M=int(layer_eff), N=self.upshape, K=self.dim,
                                multiplicity=per_cfg_mult,
                                label=f"img/ffn_up_gate({config_name})",
                                hw_cfg_path=self.config_comp0, **img_ffn_kwargs)
            self._record_energy(M=int(layer_eff), N=self.upshape, K=self.dim,
                                multiplicity=per_cfg_mult,
                                label=f"img/ffn_up_proj({config_name})",
                                hw_cfg_path=self.config_comp0, **img_ffn_kwargs)
            self._append_to_log(f"Image generation - End FFN up, total cycles:{ffn_up_cycles}")
            config_cycles_iter += ffn_up_cycles

            self._append_to_log(f"Image generation - Start FFN down")
            ffn_down_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='ffn_down', image_input=input_len)
            self._record_energy(M=input_len, N=self.dim, K=self.upshape,
                                multiplicity=per_cfg_mult,
                                label=f"img/ffn_down({config_name})",
                                hw_cfg_path=self.config_comp0, **img_ffn_kwargs)
            self._append_to_log(f"Image generation - End FFN down, total cycles:{ffn_down_cycles}")
            config_cycles_iter += ffn_down_cycles
            
            config_cycles = config_cycles_iter * self.num_layer
            total_image_cycles += config_cycles

            opmul = self.num_layer * n_steps
            self._add_op("img/qkv", input_cycles * opmul)
            self._add_op("img/attn", attn_cycles * opmul)
            self._add_op("img/omap", output_cycles * opmul)
            self._add_op("img/ffn_up", ffn_up_cycles * opmul)
            self._add_op("img/ffn_down", ffn_down_cycles * opmul)

            self._append_to_log(f"Image generation - {config_name} config completed, cycles: {config_cycles}")

        total_image_cycles += drain_cycles
        self._add_op("img/drain", drain_cycles * n_steps)

        # 乘以生成步数
        final_image_cycles = total_image_cycles * self.gen_image_step
        self.total_cycles_all += final_image_cycles
        
        image_total_cycles = self.total_cycles_all - image_start_cycles
        self._append_to_log(f"=========== Image Generation Completed, total cycles: {image_total_cycles}, total seconds: {image_total_cycles/500000000} ===========")


    def run_model(self):
        """运行完整的Janus模型仿真"""
        # 清空之前的日志文件。result_path 是文件路径（与 _append_to_log 一致），不是目录——
        # 历史上这里写成 os.path.join(result_path, "results.log") 得到一个永不存在的路径，
        # os.remove 从不执行 → 日志被 append 污染。直接用 result_path 即可。
        log_file = self.result_path
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # 记录仿真开始
        self._append_to_log("======= Bagel Model Simulation Started =======")
        self._append_to_log(f"Configuration: text_len={self.gen_text_len}, image_steps={self.gen_image_step}, layers={self.num_layer}")
        
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
        
        # GenEval 也含图像阶段 (Janus 无 GenEdit)。2026-06-12 修核对项7。
        if self.task in ("GenEdit", "GenImage", "GenEval"):
            # 运行图像生成
            self.run_gen_image()
        
        # 记录最终结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self._append_to_log("=" * 50)
        self._append_to_log(f"FINAL RESULT - Total Cycles: {self.total_cycles_all}, total seconds: {self.total_cycles_all/500000000}")
        self._append_to_log(f"Simulation Duration: {duration:.2f} seconds")
        self._append_to_log(self._format_cycle_breakdown())

        # Energy report (Phase F).
        if self.energy_enabled and self.energy is not None:
            self.energy.add_cycles(int(self.total_cycles_all))
            self._append_to_log(self.energy.format_report())

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
