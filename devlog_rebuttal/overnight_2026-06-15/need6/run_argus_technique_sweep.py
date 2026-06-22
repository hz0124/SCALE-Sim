#!/usr/bin/env python
"""
Need-6 ARGUS per-technique ablation ladder, BAGEL + GenEdit, two axes
(context-length + batch-size).

Motivation: the overnight ablation ladder (RESULTS.md sec B) was taken at a
single operating point (kv_cache_init=3254, batch=1) where T1 (+1.08x) and T2
(+1.23x) look weak next to T3 (+1.60x). T1 saves *attention* cycles, which grow
with KV length while the FFN bottleneck does not -> T1's contribution should
*rise* with context. T2 (FFN reuse) and T3 (dual-array width) are
context-independent. Batch scales the array-saturated image stage x batch
uniformly across every variant, so per-technique ratios there are expected to be
near-flat. This sweep produces the per-technique contribution curves to confirm
that and quantify how the ranking shifts.

Per grid cell we run 8 variants (isolated + leave-one-out):

  variant   t1_soft t1_hard t2_soft t2_hard t3_soft quant_proj
  all_off      F       F       F       F       F        F
  only_T1      T       T       F       F       F        F
  only_T2      F       F       T       T       F        F
  only_T3      F       F       F       F       T        F
  noT1         F       F       T       T       T        T
  noT2         T       T       F       F       T        T
  noT3         T       T       T       T       F        T
  all_on       T       T       T       T       T        T

quant_proj (INT4 projections, a 4th knob) is held constant *within* each
comparison family so it never confounds a T-delta: OFF for the isolated family
(all_off / only_Tx), ON for the leave-one-out / all_on family. t2_hard is an
unmodelled placeholder (bagel_sim.py) and just mirrors t2_soft.

Derived metrics (computed at --summarize):
  isolated contribution of Tx = cycles(all_off) / cycles(only_Tx)
  marginal (leave-one-out)    = cycles(noTx)    / cycles(all_on)
  (same ratios for energy_mJ.)

Methodology / cross-checks:
  - ctxlen axis (batch=1): override kv_cache_init AND the matching profiled
    sparsity/reuse row (need6_context_length_sweep.md sec 1). image_len/text_len
    stay at the config_ours.json originals (3193/242). all_on then reproduces
    need6_ctxlen_sweep_results.json; all_off reproduces the GEdit/all_off ctxlen
    rows in need6_argus_ablation_results.json.
  - batch axis: hold config_ours.json base params (kv=3254, base sparsity/reuse),
    vary batch_size only. all_on reproduces the `ours` row of
    need6_batch_sweep_results.json; all_off reproduces GEdit/all_off batch rows
    in need6_argus_ablation_results.json.
  - NOTE the two axes use different reuse params at their nominal base point
    (ctxlen uses the profiled image_only_sim ~0.18-0.19; batch uses the
    config_ours.json default 0.71). They are intentionally kept separate (per
    need6_context_length_sweep.md) and never merged into one table.

Variants are materialised as temp JSON configs (NOT post-hoc attribute writes):
t3_soft drives derived state inside read_from_json (config_ffn_text selection and
array_width_ffn), so the switches must be present before read_from_json runs.
batch_size has no derived state and is set post-hoc (as the batch sweep does).

Outputs (incremental, resumable):
  need6_technique_ctx_sweep_results.{json,csv}
  need6_technique_batch_sweep_results.{json,csv}

Usage:
  python devlog_rebuttal/overnight_2026-06-15/need6/run_argus_technique_sweep.py             # both axes
  python devlog_rebuttal/overnight_2026-06-15/need6/run_argus_technique_sweep.py ctxlen      # one axis
  python devlog_rebuttal/overnight_2026-06-15/need6/run_argus_technique_sweep.py batch
  python devlog_rebuttal/overnight_2026-06-15/need6/run_argus_technique_sweep.py --summarize # print md tables
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from simulation_core.bagel_sim import Bagel_sim

OUTDIR = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6")
TMPDIR = "/tmp/argus_tech_sweep"
BASE_CFG = os.path.join(ROOT, "topologies", "bagel", "config_ours.json")

OUT = {
    "ctxlen": (os.path.join(OUTDIR, "need6_technique_ctx_sweep_results.json"),
               os.path.join(OUTDIR, "need6_technique_ctx_sweep_results.csv")),
    "batch":  (os.path.join(OUTDIR, "need6_technique_batch_sweep_results.json"),
               os.path.join(OUTDIR, "need6_technique_batch_sweep_results.csv")),
}

T, F = True, False
# variant -> the 6 switch overrides
VARIANTS = {
    "all_off": dict(t1_soft=F, t1_hard=F, t2_soft=F, t2_hard=F, t3_soft=F, quant_proj=F),
    "only_T1": dict(t1_soft=T, t1_hard=T, t2_soft=F, t2_hard=F, t3_soft=F, quant_proj=F),
    "only_T2": dict(t1_soft=F, t1_hard=F, t2_soft=T, t2_hard=T, t3_soft=F, quant_proj=F),
    "only_T3": dict(t1_soft=F, t1_hard=F, t2_soft=F, t2_hard=F, t3_soft=T, quant_proj=F),
    "noT1":    dict(t1_soft=F, t1_hard=F, t2_soft=T, t2_hard=T, t3_soft=T, quant_proj=T),
    "noT2":    dict(t1_soft=T, t1_hard=T, t2_soft=F, t2_hard=F, t3_soft=T, quant_proj=T),
    "noT3":    dict(t1_soft=T, t1_hard=T, t2_soft=T, t2_hard=T, t3_soft=F, quant_proj=T),
    "all_on":  dict(t1_soft=T, t1_hard=T, t2_soft=T, t2_hard=T, t3_soft=T, quant_proj=T),
}
VARIANT_ORDER = ["all_off", "only_T1", "only_T2", "only_T3", "noT1", "noT2", "noT3", "all_on"]

CTX_LENGTHS = [2566, 3211, 7609, 12018]
# Profiled sparsity/reuse per context length (need6_context_length_sweep.md sec 1,
# verbatim values used by the existing ctxlen sweep so all_on cross-checks).
CTX_PARAMS = {
    2566:  dict(sparsity_cross_attn=0.4090, low_precise_self_attn=0.3867, text_only_sim=0.2723, image_only_sim=0.1829),
    3211:  dict(sparsity_cross_attn=0.4217, low_precise_self_attn=0.4079, text_only_sim=0.2844, image_only_sim=0.1842),
    7609:  dict(sparsity_cross_attn=0.4978, low_precise_self_attn=0.5457, text_only_sim=0.3021, image_only_sim=0.1927),
    12018: dict(sparsity_cross_attn=0.5338, low_precise_self_attn=0.6237, text_only_sim=0.2636, image_only_sim=0.1873),
}
BATCHES = [1, 4, 16, 64]

OPS = ["text/qkv", "text/attn", "text/omap", "text/ffn_up", "text/ffn_down",
       "img/qkv", "img/attn", "img/omap", "img/ffn_up", "img/ffn_down"]


def canon_op(label):
    stage = "text" if label.startswith("text/") else "img"
    rest = label.split("/", 1)[1]
    for op in ("attn", "qkv", "omap", "ffn_up", "ffn_down"):
        if rest.startswith(op):
            return f"{stage}/{op}"
    return f"{stage}/{rest}"


def make_temp_cfg(base, variant, axis, point):
    """Build a variant config dict and write it to a temp JSON; return its path."""
    cfg = dict(base)
    cfg.update(VARIANTS[variant])
    if axis == "ctxlen":
        cfg["kv_cache_init"] = point
        cfg.update(CTX_PARAMS[point])
    # keep run_model from clobbering the real ./results logs
    cfg["result_path"] = os.path.join(TMPDIR, f"log_{axis}_{point}_{variant}.log")
    path = os.path.join(TMPDIR, f"cfg_{axis}_{point}_{variant}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_one(base, variant, axis, point):
    cfg_path = make_temp_cfg(base, variant, axis, point)
    b = Bagel_sim(hardware_type="ours", task="GenEdit")
    b.read_from_json(cfg_path=cfg_path)
    if axis == "batch":
        b.batch_size = point
    b.energy_enabled = True
    b.setup_energy()
    total_cycles = b.run_model()

    eb = b.energy.breakdown()
    energy_by_op = {}
    for r in b.energy.records:
        k = canon_op(r["label"])
        energy_by_op[k] = energy_by_op.get(k, 0.0) + r["subtotal_pJ"]

    batch = b.batch_size
    return {
        "variant": variant, "axis": axis, "point": point,
        "kv_cache_init": b.kv_cache_init, "batch": batch,
        "total_cycles": total_cycles,
        "seconds": total_cycles / 500e6,
        "throughput_req_per_s": batch / (total_cycles / 500e6),
        "energy_mJ": eb["total_mJ"],
        "compute_mJ": eb["compute_pJ"] * 1e-9,
        "sram_mJ": eb["sram_pJ"] * 1e-9,
        "dram_mJ": eb["dram_pJ"] * 1e-9,
        "avg_power_W": eb["avg_power_W"],
        "edp_mJs": eb["edp_pJs"] * 1e-9,
        "energy_per_req_mJ": eb["total_mJ"] / batch,
        "cycle_breakdown": dict(b.cycle_breakdown),
        "energy_by_op_mJ": {k: v * 1e-9 for k, v in energy_by_op.items()},
    }


def write_outputs(axis, rows):
    out_json, out_csv = OUT[axis]
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["variant", "axis", "point", "kv_cache_init", "batch",
            "total_cycles", "seconds", "throughput_req_per_s",
            "energy_mJ", "energy_per_req_mJ", "compute_mJ", "sram_mJ", "dram_mJ",
            "avg_power_W", "edp_mJs"]
    cols += [f"cyc:{op}" for op in OPS] + [f"e_mJ:{op}" for op in OPS]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            if "error" in r:
                w.writerow([r["variant"], r["axis"], r["point"], "ERROR: " + r["error"]])
                continue
            row = [r.get(c, "") for c in cols[:15]]
            row += [r["cycle_breakdown"].get(op, 0) for op in OPS]
            row += [r["energy_by_op_mJ"].get(op, 0) for op in OPS]
            w.writerow(row)


def load_rows(axis):
    out_json, _ = OUT[axis]
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                return [r for r in json.load(f) if "error" not in r]
        except Exception:
            return []
    return []


def run_axis(axis, base):
    points = CTX_LENGTHS if axis == "ctxlen" else BATCHES
    rows = load_rows(axis)
    done = {(r["variant"], r["point"]) for r in rows}
    # Interleave variants within each point so a finished point gives a full
    # ladder column as early as possible.
    for point in points:
        for variant in VARIANT_ORDER:
            if (variant, point) in done:
                print(f"===== {axis}={point} {variant} (cached, skip) =====", flush=True)
                continue
            print(f"\n===== {axis}={point} {variant} =====", flush=True)
            try:
                rows.append(run_one(base, variant, axis, point))
                r = rows[-1]
                print(f"  cycles={r['total_cycles']:.0f}  "
                      f"energy={r['energy_mJ']:.3f} mJ  power={r['avg_power_W']:.2f} W",
                      flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}", flush=True)
                rows.append({"variant": variant, "axis": axis, "point": point, "error": str(e)})
            write_outputs(axis, rows)
    print(f"\nWrote {OUT[axis][0]}", flush=True)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _ratio(num, den):
    return num / den if den else float("nan")


def summarize():
    for axis, label in (("ctxlen", "context length"), ("batch", "batch size")):
        rows = load_rows(axis)
        if not rows:
            print(f"\n## {axis}: (no results yet)\n")
            continue
        by_pv = {(r["point"], r["variant"]): r for r in rows}
        points = sorted({r["point"] for r in rows})
        axis_hdr = "Context" if axis == "ctxlen" else "Batch"

        def cyc(p, v):
            return by_pv.get((p, v), {}).get("total_cycles")

        def ene(p, v):
            return by_pv.get((p, v), {}).get("energy_mJ")

        print(f"\n## ARGUS per-technique contribution vs {label} (BAGEL/GEdit)\n")

        # isolated speedup (only_Tx vs all_off)
        print(f"### Isolated speedup  only_Tx / all_off  (cycles)\n")
        print(f"| {axis_hdr} | only_T1 | only_T2 | only_T3 | all_on |")
        print("|---|---|---|---|---|")
        for p in points:
            off = cyc(p, "all_off")
            cells = [_ratio(off, cyc(p, v)) for v in ("only_T1", "only_T2", "only_T3", "all_on")]
            print(f"| {p} | " + " | ".join(f"{c:.3f}x" for c in cells) + " |")

        # marginal speedup (noTx vs all_on)
        print(f"\n### Marginal (leave-one-out) speedup  noTx / all_on  (cycles)\n")
        print(f"| {axis_hdr} | T1 (noT1/all_on) | T2 (noT2/all_on) | T3 (noT3/all_on) |")
        print("|---|---|---|---|")
        for p in points:
            on = cyc(p, "all_on")
            cells = [_ratio(cyc(p, v), on) for v in ("noT1", "noT2", "noT3")]
            print(f"| {p} | " + " | ".join(f"{c:.3f}x" for c in cells) + " |")

        # isolated energy-efficiency
        print(f"\n### Isolated energy ratio  E(all_off) / E(only_Tx)\n")
        print(f"| {axis_hdr} | only_T1 | only_T2 | only_T3 | all_on |")
        print("|---|---|---|---|---|")
        for p in points:
            off = ene(p, "all_off")
            cells = [_ratio(off, ene(p, v)) for v in ("only_T1", "only_T2", "only_T3", "all_on")]
            print(f"| {p} | " + " | ".join(f"{c:.3f}x" for c in cells) + " |")

        # img/attn and img/ffn_up share, to show the mechanism
        print(f"\n### Mechanism check: all_off op shares (img/attn, img/ffn_up as % of total cycles)\n")
        print(f"| {axis_hdr} | img/attn % | img/ffn_up % |")
        print("|---|---|---|")
        for p in points:
            r = by_pv.get((p, "all_off"))
            if not r:
                continue
            tot = r["total_cycles"]
            cb = r["cycle_breakdown"]
            print(f"| {p} | {100*cb.get('img/attn',0)/tot:.1f} | {100*cb.get('img/ffn_up',0)/tot:.1f} |")


def main():
    os.makedirs(TMPDIR, exist_ok=True)
    args = sys.argv[1:]
    if "--summarize" in args:
        summarize()
        return
    axes = [a for a in args if a in ("ctxlen", "batch")] or ["ctxlen", "batch"]
    with open(BASE_CFG) as f:
        base = json.load(f)
    for axis in axes:
        run_axis(axis, base)
    summarize()


if __name__ == "__main__":
    main()
