#!/usr/bin/env python
"""
Regenerate need6_newmodel_summary.md from the fresh sweep JSONs (new energy
model: DDR4 + idle 500mW + array static 150mW + per-hw w4a16). Rewrites the
whole file. Tables:
  §1 per-technique contribution vs context length (cycles ratios)
  §2 per-technique contribution vs batch size      (cycles ratios)
  §3 batch throughput / energy-efficiency scan (ours all_on)
Inputs (same dir):
  need6_technique_ctx_sweep_results.json
  need6_technique_batch_sweep_results.json
  need6_batch_sweep_results.json
Run:  python devlog_rebuttal/overnight_2026-06-15/need6/gen_need6_summary.py
"""
import json
import os

N = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(N, "need6_newmodel_summary.md")
CTX_POINTS = [2566, 3211, 7609, 12018]
BATCH_POINTS = [1, 4, 16, 64]
BATCH_TPUT = [1, 2, 4, 8, 16, 32, 64]


def load(name):
    p = os.path.join(N, name)
    return [r for r in json.load(open(p)) if "error" not in r]


def by_pv(rows):
    return {(r["point"], r["variant"]): r for r in rows}


def cyc(d, p, v):
    r = d.get((p, v))
    return r["total_cycles"] if r else None


def ratio(a, b):
    return a / b if (a and b) else None


def fmt(x):
    return f"{x:.2f}" if x is not None else "—"


def tech_tables(rows, points, axis_hdr):
    d = by_pv(rows)
    lines = []
    lines.append(f"### 隔离加速比 only_Tx / all_off（cycles）\n")
    lines.append(f"| {axis_hdr} | only_T1 | only_T2 | only_T3 | all_on |")
    lines.append("|---|---|---|---|---|")
    for p in points:
        off = cyc(d, p, "all_off")
        cells = [ratio(off, cyc(d, p, v)) for v in ("only_T1", "only_T2", "only_T3", "all_on")]
        lines.append(f"| {p} | " + " | ".join(fmt(c) for c in cells) + " |")
    lines.append("")
    lines.append("### 边际(留一)加速比 noTx / all_on（cycles）—— 越大说明该技术越关键\n")
    lines.append(f"| {axis_hdr} | noT1→T1 | noT2→T2 | noT3→T3 |")
    lines.append("|---|---|---|---|")
    for p in points:
        on = cyc(d, p, "all_on")
        cells = [ratio(cyc(d, p, v), on) for v in ("noT1", "noT2", "noT3")]
        lines.append(f"| {p} | " + " | ".join(fmt(c) for c in cells) + " |")
    return "\n".join(lines)


def mmvet_table(abl, axis, points):
    """MM-Vet (pure-text MM) all_on vs all_off, from argus_ablation rows."""
    on = {r["point"]: r for r in abl if r["combo"] == "MMVet/all_on" and r["axis"] == axis}
    off = {r["point"]: r for r in abl if r["combo"] == "MMVet/all_off" and r["axis"] == axis}
    hdr = "Context" if axis == "ctxlen" else "Batch"
    lines = [f"| {hdr} | all_on 周期(e9) | all_on 能量(mJ) | 加速比(off/on) | 能效比(E_off/E_on) |",
             "|---|---|---|---|---|"]
    for p in points:
        o, f = on.get(p), off.get(p)
        if not o:
            lines.append(f"| {p} | — | — | — | — |")
            continue
        sp = ratio(f["total_cycles"], o["total_cycles"]) if f else None
        ee = ratio(f["energy_mJ"], o["energy_mJ"]) if f else None
        lines.append(f"| {p} | {o['total_cycles']/1e9:.1f} | {o['energy_mJ']:,.0f} | {fmt(sp)} | {fmt(ee)} |")
    return "\n".join(lines)


def tput_table(rows, abl):
    o = {r["batch"]: r for r in rows if r["hw"] == "ours"}
    base = o.get(1)
    # GEdit all_off cycles by batch (argus_ablation; only batch 1/4/16/64 run).
    off = {r["point"]: r["total_cycles"] for r in abl
           if r["combo"] == "GEdit/all_off" and r["axis"] == "batch"}
    lines = ["| batch | 总周期(e9) | 加速比(vs all_off) | 吞吐(req/s) | 吞吐相对 b=1 | 能量/req(e3 mJ) | 能效相对 b=1 |",
             "|---|---|---|---|---|---|---|"]
    for b in BATCH_TPUT:
        r = o.get(b)
        if not r:
            lines.append(f"| {b} | — | — | — | — | — | — |")
            continue
        tput = r["throughput_req_per_s"]
        epr = r["energy_per_req_mJ"]
        trel = tput / base["throughput_req_per_s"] if base else None
        erel = base["energy_per_req_mJ"] / epr if (base and epr) else None
        sp = ratio(off.get(b), r["total_cycles"])
        lines.append(f"| {b} | {r['total_cycles']/1e9:.1f} | {fmt(sp)} | {tput:.2e} | "
                     f"{fmt(trel)} | {epr/1e3:.1f} | {fmt(erel)} |")
    return "\n".join(lines)


def main():
    ctx = load("need6_technique_ctx_sweep_results.json")
    bat = load("need6_technique_batch_sweep_results.json")
    bsw = load("need6_batch_sweep_results.json")
    abl = load("need6_argus_ablation_results.json")

    md = f"""# Need-6 扫描汇总（新文本 memory-bound 模型 + 最终能量系数，2026-06-18 重跑）

> 模型：BAGEL / GenEdit（T1/T2 是图像阶段技术，纯文本对它们 no-op，不扫）。
> 数据：`need6_technique_{{ctx,batch}}_sweep_results.csv`、`need6_batch_sweep_results.csv`、
> `need6_ctxlen_sweep_results.csv`（全部新模型 + DDR4 20pJ/bit + idle 500mW + 阵列静态 150mW + per-hw w4a16）。
> §1/§2 是 cycle 比（不随能量系数变）；§3 的能量/req 反映最终能量系数。本文件由
> `gen_need6_summary.py` 从 JSON 重生成。

## 1. 技术点贡献 vs 上下文长度（GenEdit，batch=1）

{tech_tables(ctx, CTX_POINTS, "Context")}

## 2. 技术点贡献 vs batch size（GenEdit）

{tech_tables(bat, BATCH_POINTS, "Batch")}

## 3. Batch 吞吐 / 能效扫描（GenEdit，ours all_on）

> 新模型显式把「权重流式跨 M 摊薄」建进去 → decode 批处理收益由此而来。

{tput_table(bsw, abl)}

> 加速比(vs all_off) = cycles(GEdit/all_off) / cycles(ours all_on)，源自 `need6_argus_ablation_results.json`
> 的 GEdit/all_off batch 轴（仅跑了 batch 1/4/16/64，其余标 —）。GenEdit 由 array-saturated 图像阶段主导，
> 故加速比随 batch 几乎恒定 ~2.65×（对比 §4 MM-Vet 纯文本随 batch 跌到 1.0×）。

## 4. MM-Vet（纯文本 MM 任务，Bagel）all_on vs all_off

> §1–§3 是 GenEdit（图生成全流程）；MM-Vet 是纯文本 decode（无图像阶段），T1/T2 对它 no-op，
> all_on 的收益只来自 **T3 精度 + quant_proj**（投影/FFN 走 W4A16，对齐 figna）。数据源
> `need6_argus_ablation_results.json`（MMVet/all_on、MMVet/all_off）。加速比 = cycles_off/cycles_on。

### 加速比 / 能效比 vs 上下文长度（batch=1）

{mmvet_table(abl, "ctxlen", CTX_POINTS)}

### 加速比 / 能效比 vs batch size（base context）

{mmvet_table(abl, "batch", BATCH_POINTS)}

## 注记
- 隔离=只开一个技术 vs 全关；留一=关一个 vs 全开。T1/T2 是图像阶段技术，短上下文工作点贡献小、长上下文放大。
- batch 扫描：投影/FFN 权重跨 batch 复用（stream 摊薄）→ 吞吐随 batch 升；attention 不可批（每请求独立 KV）→ 收益递减。
- 能量/req 含 DRAM 背景(idle 500mW)+阵列静态(150mW)运行时间项；能效相对 b=1 = (E/req @b=1)/(E/req @b)。
- **MM-Vet 随 batch 加速比从 3.10×→1.00×**（GenEdit 几乎不变）：纯文本 decode 小 batch 是 memory-bound
  （W4A16 权重带宽优势显著），大 batch 转 compute-bound → 精度不再省 cycle，all_on/all_off 收敛。
"""
    open(OUT, "w").write(md)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
