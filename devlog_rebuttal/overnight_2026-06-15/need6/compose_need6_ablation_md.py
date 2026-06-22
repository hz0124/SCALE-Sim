#!/usr/bin/env python
"""
Compose the §3 ARGUS all_on-vs-all_off speedup section of
need6_context_length_sweep.md. Two compact tables (MM-Vet | GEdit columns):
  - context-length speedup at 2k/4k/8k/12k buckets
  - batch-size  speedup at batch 1/4/16/64
Speedup = cycles_all_off / cycles_all_on  (>1 = all_on faster).

Idempotent: rewrites the region between <!-- ABLATION_START --> and
<!-- ABLATION_END --> (appends if markers absent). Re-run as the sweep fills in.

  all_on  GEdit -> need6_ctxlen_sweep_results.json (ctxlen)
                   need6_batch_sweep_results.json   (batch, hw=ours)
  others        -> need6_argus_ablation_results.json
"""
import json
import os

N = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(N, "need6_context_length_sweep.md")
START, END = "<!-- ABLATION_START -->", "<!-- ABLATION_END -->"

# (nominal label, actual context length used)
CTX_ROWS = [("2k", 2566), ("4k", 3211), ("8k", 7609), ("12k", 12018)]
BATCH_ROWS = [1, 4, 16, 64]


def load():
    abl = [r for r in json.load(open(os.path.join(N, "need6_argus_ablation_results.json")))
           if "error" not in r]

    def pick(combo, axis):
        return {r["point"]: r for r in abl if r["combo"] == combo and r["axis"] == axis}

    on_ctx = {r["context_length"]: r for r in
              json.load(open(os.path.join(N, "need6_ctxlen_sweep_results.json")))}
    on_bat = {r["batch"]: r for r in
              json.load(open(os.path.join(N, "need6_batch_sweep_results.json")))
              if r["hw"] == "ours"}
    return {
        "GEdit/all_on/ctxlen": on_ctx,
        "GEdit/all_on/batch": on_bat,
        "GEdit/all_off/ctxlen": pick("GEdit/all_off", "ctxlen"),
        "GEdit/all_off/batch": pick("GEdit/all_off", "batch"),
        "MMVet/all_on/ctxlen": pick("MMVet/all_on", "ctxlen"),
        "MMVet/all_on/batch": pick("MMVet/all_on", "batch"),
        "MMVet/all_off/ctxlen": pick("MMVet/all_off", "ctxlen"),
        "MMVet/all_off/batch": pick("MMVet/all_off", "batch"),
    }


def speedup(d, dataset, axis, point):
    on = d[f"{dataset}/all_on/{axis}"].get(point)
    off = d[f"{dataset}/all_off/{axis}"].get(point)
    if on and off and on.get("total_cycles"):
        return off["total_cycles"] / on["total_cycles"]
    return None


def cell(v):
    return f"{v:.2f}×" if v is not None else "⏳"


def build():
    d = load()
    done = total = 0
    out = [START, "",
           "## §3 ARGUS all_on vs all_off 加速比(MM-Vet / GEdit)", "",
           "ARGUS 全技术栈(T1 稀疏注意力 / T2 FFN 复用 / T3 阶段自适应异构阵列 + INT4 投影量化)"
           "相对全关基线的端到端加速比,随 **上下文长度** 与 **batch** 的 scaling。", "",
           "- **all_on** = ours 默认(5 开关 + quant_proj);**all_off** = 5 开关 false + "
           "`quant_proj=false`(need-4 ablation 基线:text fp16、单路 full-KV 注意力、无复用、image 宽 32)。",
           "- **加速比 = cycles_all_off / cycles_all_on**(>1 = all_on 更快)。",
           "- MM-Vet = MM 任务,只跑 text decode(差异来自 T3 精度 + quant_proj);GEdit = 图生成全流程。",
           "- ⏳ = 仍在采集。", ""]

    out += ["### 加速比 vs context length(batch=1)", "",
            "| Context | MM-Vet | GEdit |", "|---|---|---|"]
    for label, L in CTX_ROWS:
        m = speedup(d, "MMVet", "ctxlen", L)
        g = speedup(d, "GEdit", "ctxlen", L)
        done += (m is not None) + (g is not None)
        total += 2
        out.append(f"| {label} ({L:,}) | {cell(m)} | {cell(g)} |")

    out += ["", "### 加速比 vs batch size(base context)", "",
            "| Batch size | MM-Vet | GEdit |", "|---|---|---|"]
    for b in BATCH_ROWS:
        m = speedup(d, "MMVet", "batch", b)
        g = speedup(d, "GEdit", "batch", b)
        done += (m is not None) + (g is not None)
        total += 2
        out.append(f"| {b} | {cell(m)} | {cell(g)} |")

    out += ["", f"> 进度:{done}/{total} 个加速比已算出。", "", END]
    return "\n".join(out)


def main():
    block = build()
    src = open(MD).read() if os.path.exists(MD) else ""
    if START in src and END in src:
        new = src[:src.index(START)] + block + src[src.index(END) + len(END):]
    else:
        new = src.rstrip() + "\n\n---\n\n" + block + "\n"
    open(MD, "w").write(new)
    print("Updated", MD)


if __name__ == "__main__":
    main()
