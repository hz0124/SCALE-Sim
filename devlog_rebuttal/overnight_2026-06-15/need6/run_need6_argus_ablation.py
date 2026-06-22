#!/usr/bin/env python
"""
Need-6 ARGUS all-on vs all-off ablation, BAGEL, two datasets (GEdit + MMVet),
two axes (context-length + batch-size).

Goal: speedup of argus_all_on relative to argus_all_off, per dataset, per case.

  all_on  = ours default (5 technique switches true, quant_proj true).
  all_off = 5 switches false (t1_soft/t1_hard/t2_soft/t2_hard/t3_soft) + quant_proj
            false -> the need-4 ablation baseline (text fp16 cfg, single full-KV
            attention path, no FFN reuse, image width 32, fp16 projections).

What this script runs (the 3 combos NOT already collected):
  - GEdit / all_off   (task=GenEdit, config_ours_alloff.json)
  - MMVet / all_on    (task=MM,      config_ours_MM.json)
  - MMVet / all_off   (task=MM,      config_ours_MM_alloff.json)
The 4th combo (GEdit / all_on) is already in need6_ctxlen_sweep_results.json
(ctxlen) and need6_batch_sweep_results.json (batch, the `ours` row) and is
merged at md-composition time, not re-run here.

Per (combo, axis, point) it records total cycles / seconds / throughput, energy
(total + compute/SRAM/DRAM split, avg power, EDP) and per-operator cycle/energy
breakdowns. Runs are sequential (the harness writes a shared layer.csv).

Notes on the sweeps:
  - ctxlen axis: batch=1, override kv_cache_init over the 9 GEdit-derived lengths
    and recompute kv_cache_without_img/_text. For all_off the sparsity/reuse
    profiling values are inert (switches off); for MMVet/all_on the text-decode
    stage uses neither T1 sparsity nor T2 reuse, so only token length matters
    (user decision) -> both can sweep length with the base config untouched.
  - batch axis: kv_cache_init at the config's base value, vary batch_size.
  - MMVet = MM task = text-decode only (run_model skips the image stage), so its
    rows carry text/* ops only.

Outputs need6_argus_ablation_results.{json,csv} (incremental, resumable).

Usage:
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_argus_ablation.py
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_argus_ablation.py ctxlen   # one axis
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_argus_ablation.py batch
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from simulation_core.bagel_sim import Bagel_sim

OUTDIR = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6")
OUT_JSON = os.path.join(OUTDIR, "need6_argus_ablation_results.json")
OUT_CSV = os.path.join(OUTDIR, "need6_argus_ablation_results.csv")

# (combo_id, dataset, hw_label, task, cfg_path, image_len, text_len)
COMBOS = [
    ("GEdit/all_off", "GEdit", "all_off", "GenEdit",
     "./topologies/bagel/config_ours_alloff.json", 3193, 242),
    ("MMVet/all_on", "MMVet", "all_on", "MM",
     "./topologies/bagel/config_ours_MM.json", 3193, 48),
    ("MMVet/all_off", "MMVet", "all_off", "MM",
     "./topologies/bagel/config_ours_MM_alloff.json", 3193, 48),
]
# Trimmed to the 4 representative buckets the user wants (2k/4k/8k/12k and
# batch 1/4/16/64). The full 9-length / 7-batch lists are kept in git history.
CTX_LENGTHS = [2566, 3211, 7609, 12018]
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


def run_one(combo, axis, point):
    combo_id, dataset, hw_label, task, cfg_path, image_len, text_len = combo
    b = Bagel_sim(hardware_type="ours", task=task)
    b.read_from_json(cfg_path=os.path.join(ROOT, cfg_path))

    if axis == "ctxlen":
        b.kv_cache_init = point
        b.kv_cache_without_img = point - image_len
        b.kv_cache_without_text = point - text_len
        b.batch_size = 1
    elif axis == "batch":
        b.batch_size = point
    else:
        raise ValueError(axis)

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
        "combo": combo_id, "dataset": dataset, "hw": hw_label,
        "axis": axis, "point": point,
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


def write_outputs(rows):
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["combo", "dataset", "hw", "axis", "point", "kv_cache_init", "batch",
            "total_cycles", "seconds", "throughput_req_per_s",
            "energy_mJ", "energy_per_req_mJ", "compute_mJ", "sram_mJ", "dram_mJ",
            "avg_power_W", "edp_mJs"]
    cols += [f"cyc:{op}" for op in OPS] + [f"e_mJ:{op}" for op in OPS]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            if "error" in r:
                w.writerow([r["combo"], r["dataset"], r["hw"], r["axis"],
                            r["point"], "ERROR: " + r["error"]])
                continue
            row = [r.get(c, "") for c in cols[:17]]
            row += [r["cycle_breakdown"].get(op, 0) for op in OPS]
            row += [r["energy_by_op_mJ"].get(op, 0) for op in OPS]
            w.writerow(row)


def main():
    axes = sys.argv[1:] or ["ctxlen", "batch"]

    rows = []
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as f:
                rows = [r for r in json.load(f) if "error" not in r]
        except Exception:
            rows = []
    done = {(r["combo"], r["axis"], r["point"]) for r in rows}

    # Build the (combo, axis, point) work list. Order interleaves all_on/all_off
    # per point so each completed PAIR immediately fills one speedup cell in the
    # md (the table value needs both halves). GEdit/all_on comes from reference
    # files (not run here), so GEdit/all_off alone completes a GEdit cell.
    by_id = {c[0]: c for c in COMBOS}
    work = []
    if "ctxlen" in axes:
        for L in CTX_LENGTHS:
            for cid in ("MMVet/all_on", "MMVet/all_off", "GEdit/all_off"):
                if cid in by_id:
                    work.append((by_id[cid], "ctxlen", L))
    if "batch" in axes:
        for bsz in BATCHES:
            for cid in ("GEdit/all_off", "MMVet/all_on", "MMVet/all_off"):
                if cid in by_id:
                    work.append((by_id[cid], "batch", bsz))

    for combo, axis, point in work:
        combo_id = combo[0]
        if (combo_id, axis, point) in done:
            print(f"===== {combo_id} {axis}={point} (cached, skip) =====", flush=True)
            continue
        print(f"\n===== {combo_id} {axis}={point} =====", flush=True)
        try:
            rows.append(run_one(combo, axis, point))
            r = rows[-1]
            print(f"  cycles={r['total_cycles']:.0f}  "
                  f"energy={r['energy_mJ']:.3f} mJ  power={r['avg_power_W']:.2f} W",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}", flush=True)
            rows.append({"combo": combo_id, "dataset": combo[1], "hw": combo[2],
                         "axis": axis, "point": point, "error": str(e)})
        write_outputs(rows)

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
