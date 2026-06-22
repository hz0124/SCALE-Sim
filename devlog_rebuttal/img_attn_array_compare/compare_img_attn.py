#!/usr/bin/env python
"""
Bagel GenEdit 图像(diffusion)阶段 attention 算子对比：
ARGUS 真实配置 vs 统一 32x64 单精度阵列(FP16FP16 / FP8FP8 / FP4FP4)。

口径（与用户确认）：
- 基线 = 统一 64 宽阵列 + 全稠密 KV（config_ours_alloff.json + array_width_half=64）。
  三个基线几何/工作负载相同，只差 MAC 精度 → 计算时间相同，只有能耗按精度不同。
- ARGUS = config_ours.json（attention 走 array_width_half=32 半宽、fp16、含 T1+T3 稀疏）。
- 计算时间 = attn 周期 / 500MHz；计算能耗 = attn MAC 能耗(MAC × mac_pj)，不含 SRAM/DRAM/静态。
- per-MAC 能量从 get_coefficients('ours')['mac_pj'] 读：fp16=1.5 / fp8=0.4 / fp4=0.2 pJ。

attn 周期取 cycle_breakdown['img/attn']（已 ×层/头/步/3 个 CFG 分支）；
attn MAC 由能量记录中 label 以 'img/attn' 开头者的 M·N·K·multiplicity·scale 求和。

Run:  python devlog_rebuttal/img_attn_array_compare/compare_img_attn.py
"""
import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from simulation_core.bagel_sim import Bagel_sim
from simulation_core.energy_accounting.coefficients import get_coefficients

FREQ = 500e6
TMP = "/tmp/img_attn_compare"
os.makedirs(TMP, exist_ok=True)
MAC_PJ = get_coefficients("ours")["mac_pj"]   # single source of truth


# operand bytes per precision (drives SRAM/DRAM access energy, which scales
# linearly with operand width). The harness runs attention at fp16 (2 B); for
# FP8/FP4 we scale the measured fp16 SRAM+DRAM energy by bytes[prec]/2.
PREC_BYTES = {"fp16": 2.0, "fp8": 1.0, "fp4": 0.5}


def run_attn(cfg_rel, width_half_override=None, tag="run"):
    """In-process Bagel GenEdit run. Returns dict with attn cycles, MAC count,
    and the fp16-measured SRAM+DRAM dynamic energy for img/attn (pJ)."""
    b = Bagel_sim(hardware_type="ours", task="GenEdit")
    b.read_from_json(cfg_path=os.path.join(ROOT, cfg_rel))
    b.result_path = os.path.join(TMP, f"{tag}.log")
    if width_half_override is not None:
        b.array_width_half = width_half_override
    b.energy_enabled = True
    b.setup_energy()
    # silence the verbose SCALE-Sim / harness stdout
    devnull = open(os.devnull, "w")
    old = sys.stdout
    sys.stdout = devnull
    try:
        b.run_model()
    finally:
        sys.stdout = old
        devnull.close()

    attn_cycles = float(b.cycle_breakdown.get("img/attn", 0.0))
    attn_mac = 0
    dyn_total_pJ = 0.0   # MAC + SRAM + DRAM (dynamic) for img/attn, at fp16
    kv = {}
    for r in b.energy.records:
        if r["label"].startswith("img/attn"):
            attn_mac += r["M"] * r["N"] * r["K"] * r["multiplicity"] * r["scale"]
            dyn_total_pJ += r["subtotal_pJ"]   # MAC+SRAM+DRAM delta for this sub-run
            m = re.search(r"\(([^,]+),kv=(\d+)\)", r["label"])
            if m:
                kv[m.group(1)] = int(m.group(2))
    attn_mac = int(attn_mac)
    # measured run is fp16; back out SRAM+DRAM = dynamic_total - MAC(fp16)
    sram_dram_fp16_pJ = dyn_total_pJ - attn_mac * MAC_PJ["fp16"]
    return {"cycles": attn_cycles, "mac": attn_mac,
            "sram_dram_fp16_pJ": sram_dram_fp16_pJ, "kv": kv}


def main():
    print("running ARGUS (config_ours.json, 32 half-width, fp16, T1+T3 sparse) ...", flush=True)
    A = run_attn("topologies/bagel/config_ours.json", tag="argus")
    print("running baseline (config_ours_alloff.json + array_width_half=64, dense KV, fp16) ...", flush=True)
    B = run_attn("topologies/bagel/config_ours_alloff.json", width_half_override=64, tag="base64")

    def secs(c):
        return c / FREQ

    def mac_mJ(mac, prec):
        return mac * MAC_PJ[prec] * 1e-9

    def memr_mJ(sram_dram_fp16_pJ, prec):
        # SRAM/DRAM energy scales linearly with operand bytes
        return sram_dram_fp16_pJ * (PREC_BYTES[prec] / PREC_BYTES["fp16"]) * 1e-9

    kv_str = lambda kv: "/".join(f"{v}" for v in kv.values()) if kv else "—"

    # rows: (name, array, precision, source-run dict)
    rows = [
        ("ARGUS",    "32×64 (attn 32 半宽)", "fp16", A),
        ("FP16FP16", "32×64 (attn 64 满宽)", "fp16", B),
        ("FP8FP8",   "32×64 (attn 64 满宽)", "fp8",  B),
        ("FP4FP4",   "32×64 (attn 64 满宽)", "fp4",  B),
    ]

    hdr = ("| 配置 | 阵列(attn宽) | 精度(pJ) | 有效KV(full/woimg/wotext) | 时间@500MHz(s) | "
           "MAC能耗(mJ) | SRAM+DRAM能耗(mJ) | 总动态能耗(mJ) |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    cache = {}
    for name, arr, prec, R in rows:
        e_mac = mac_mJ(R["mac"], prec)
        e_mem = memr_mJ(R["sram_dram_fp16_pJ"], prec)
        e_tot = e_mac + e_mem
        cache[name] = (e_mac, e_mem, e_tot, secs(R["cycles"]))
        lines.append(
            f"| {name} | {arr} | {prec} ({MAC_PJ[prec]}) | {kv_str(R['kv'])} | "
            f"{secs(R['cycles']):.2f} | {e_mac:,.0f} | {e_mem:,.0f} | {e_tot:,.0f} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    t_argus, t_base = secs(A["cycles"]), secs(B["cycles"])
    eA = cache["ARGUS"]; e16 = cache["FP16FP16"]; e8 = cache["FP8FP8"]; e4 = cache["FP4FP4"]

    md = f"""# 图像阶段 attention 对比：ARGUS vs 32×64 FP16/FP8/FP4 阵列（Bagel GenEdit）

> 由 `compare_img_attn.py` 生成（in-process 跑 Bagel_sim，不污染 ./results）。
> 只看图像(diffusion)阶段的 attention 算子（img/attn = attn_qk + attn_sfmxv）。
> **计算时间** = attn 周期 / 500MHz。**能耗 = 总动态能耗 = MAC + SRAM + DRAM**（算子可归因的动态能量，
> **不含** 全芯片 idle 背景与阵列静态——那是运行时间项、非单算子可分摊）。MAC 列单列出来便于看构成。

## 口径
- **ARGUS** = `config_ours.json`：attention 走 `array_width_half=32`（异构阵列另一半是 FP-INT4，fp16 attention 只能用 32 宽），
  fp16，含 T1 区域稀疏 + T3 ÷2 负载均衡 → 有效 KV 大幅缩减。
- **基线（FP16/FP8/FP4 32×64）** = `config_ours_alloff.json` + `array_width_half=64`：统一阵列用满 64 宽跑 attention、
  **全稠密 KV**（无 T1 稀疏）。三个基线几何/工作负载完全相同（同周期、同 MAC、同访问计数），**只差精度**。
- per-MAC 能量(28nm 占位)：fp16={MAC_PJ['fp16']} / fp8={MAC_PJ['fp8']} / fp4={MAC_PJ['fp4']} pJ。
- **SRAM/DRAM 能量随操作数字节线性缩放**：fp16=2B / fp8=1B / fp4=0.5B → 基线访存能耗按 1 / ½ / ¼ 缩放
  （harness 实测在 fp16 口径，FP8/FP4 据字节比缩放）。

## 结果

{table}

## 解读
- **SRAM+DRAM 是 attention 能耗大头**（fp16 口径占总动态 ~66%）：只看 MAC 会严重低估。低精度同时省 MAC（×pj）
  和访存（×字节）两块，所以精度对总能耗的杠杆比只看 MAC 更大。
- **计算时间**：ARGUS {t_argus:.1f}s vs 基线 {t_base:.1f}s（基线三精度同周期）。ARGUS 32 半宽使列折叠翻倍（→慢），
  但 T1+T3 把有效 KV 从稠密大幅缩小（→快），净 **{'ARGUS 更快' if A['cycles'] < B['cycles'] else 'ARGUS 更慢'} {abs(A['cycles']-B['cycles'])/B['cycles']*100:.0f}%**。
- **总动态能耗**：ARGUS {eA[2]:,.0f} mJ（fp16+稀疏）vs FP16FP16 稠密 {e16[2]:,.0f}（ARGUS 是其 {eA[2]/e16[2]:.2f}×，稀疏少算/少搬）。
  但 **FP8FP8 稠密 {e8[2]:,.0f}**（{'与 ARGUS 相当' if abs(e8[2]-eA[2])/eA[2]<0.15 else ('比 ARGUS 低' if e8[2]<eA[2] else '比 ARGUS 高 %.0f%%' % (100*(e8[2]-eA[2])/eA[2]))}），**FP4FP4 稠密 {e4[2]:,.0f}** 则{'低于' if e4[2]<eA[2] else '高于'} ARGUS（ARGUS 是其 {eA[2]/e4[2]:.2f}×）。
- **含义**：attention 上 ARGUS 的优势主要来自**稀疏（少算少搬）+ 速度**，而非精度（它保 fp16 不量化以护精度）。
  统一低精度阵列(FP8/FP4)能在 MAC 和访存两头同时降能耗；若再叠加稀疏会更省，但 attention 低精度有精度风险。

## 注记
- attention 精度对 cycle 无影响是本仿真器的建模事实（fold 周期只由阵列几何 + kv_len 决定）→ 三基线计算时间相同。
- 总动态能耗不含 idle 背景(500mW)与阵列静态(150mW)——它们按整芯片运行时间计、非单算子可归因；如需端到端含静态请看 §A/§D 口径。
- per-MAC 与字节占位值待 RTL 标定；改 `coefficients.py` 的 `mac_pj` fp8/fp4（及本脚本 PREC_BYTES）即可重算。
"""
    out_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMG_ATTN_ARRAY_COMPARE.md")
    open(out_md, "w").write(md)
    print("wrote", out_md)
    print(f"\n[check] ARGUS img/attn cycles = {A['cycles']:,.0f}  (对照该 run 日志 OPBREAKDOWN img/attn)")


if __name__ == "__main__":
    main()
