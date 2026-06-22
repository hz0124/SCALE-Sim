#!/usr/bin/env python
"""
Need-6 batch-size sweep, BAGEL + GenEdit, all hardware.

For every (hw, batch) it runs the harness and records:
  - total cycles / seconds
  - energy (total + compute/SRAM/DRAM split, avg power, EDP)
  - per-operator CYCLE breakdown   (text|img x qkv/attn/omap/ffn_up/ffn_down)
  - per-operator ENERGY breakdown  (same canonical ops, from the audit trail)

Outputs devlog_rebuttal/overnight_2026-06-15/need6/need6_batch_sweep_results.{json,csv}.
Runs are sequential (the harness writes a shared topologies/bagel/layer.csv).

Usage:
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_batch_sweep.py             # all hw
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_batch_sweep.py ours figna  # subset
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from simulation_core.bagel_sim import Bagel_sim

CFG = {
    "ours":      "./topologies/bagel/config_ours.json",
    "figna":     "./topologies/bagel/config_figna.json",
    "flightvgm": "./topologies/bagel/config_flightvgm.json",
    "sdma":      "./topologies/bagel/config_sdma.json",
    "base":      "./topologies/bagel/config.json",
}
BATCHES = [1, 2, 4, 8, 16, 32, 64]
OPS = ["text/qkv", "text/attn", "text/omap", "text/ffn_up", "text/ffn_down",
       "img/qkv", "img/attn", "img/omap", "img/ffn_up", "img/ffn_down"]


def canon_op(label):
    """Map an energy-record label to a canonical stage/op key."""
    stage = "text" if label.startswith("text/") else "img"
    rest = label.split("/", 1)[1]
    for op in ("attn", "qkv", "omap", "ffn_up", "ffn_down"):
        if rest.startswith(op):
            return f"{stage}/{op}"
    return f"{stage}/{rest}"


def run_one(hw, batch):
    b = Bagel_sim(hardware_type=hw, task="GenEdit")
    b.read_from_json(cfg_path=os.path.join(ROOT, CFG[hw]))
    b.batch_size = batch
    b.energy_enabled = True
    b.setup_energy()
    total_cycles = b.run_model()

    eb = b.energy.breakdown()
    # per-op energy (pJ) from the subrun audit trail
    energy_by_op = {}
    for r in b.energy.records:
        energy_by_op[canon_op(r["label"])] = \
            energy_by_op.get(canon_op(r["label"]), 0.0) + r["subtotal_pJ"]

    return {
        "hw": hw, "batch": batch,
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
    """Persist rows to JSON + flat CSV. Called after EVERY run so a crash /
    session drop never loses completed points."""
    out_json = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6", "need6_batch_sweep_results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    out_csv = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6", "need6_batch_sweep_results.csv")
    cols = ["hw", "batch", "total_cycles", "seconds", "throughput_req_per_s",
            "energy_mJ", "energy_per_req_mJ", "compute_mJ", "sram_mJ", "dram_mJ",
            "avg_power_W", "edp_mJs"]
    cols += [f"cyc:{op}" for op in OPS] + [f"e_mJ:{op}" for op in OPS]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            if "error" in r:
                w.writerow([r["hw"], r["batch"], "ERROR: " + r["error"]])
                continue
            row = [r.get(c, "") for c in cols[:12]]
            row += [r["cycle_breakdown"].get(op, 0) for op in OPS]
            row += [r["energy_by_op_mJ"].get(op, 0) for op in OPS]
            w.writerow(row)


def main():
    hws = sys.argv[1:] or list(CFG.keys())
    # Resume support: keep already-computed (hw,batch) points from a prior run.
    out_json = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6", "need6_batch_sweep_results.json")
    rows = []
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                rows = [r for r in json.load(f) if "error" not in r]
        except Exception:
            rows = []
    done = {(r["hw"], r["batch"]) for r in rows}

    for hw in hws:
        for batch in BATCHES:
            if (hw, batch) in done:
                print(f"===== {hw} batch={batch} (cached, skip) =====", flush=True)
                continue
            print(f"\n===== {hw} batch={batch} =====", flush=True)
            try:
                rows.append(run_one(hw, batch))
                r = rows[-1]
                print(f"  cycles={r['total_cycles']:.0f}  "
                      f"energy={r['energy_mJ']:.3f} mJ  "
                      f"power={r['avg_power_W']:.2f} W", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                rows.append({"hw": hw, "batch": batch, "error": str(e)})
            write_outputs(rows)   # incremental persist after every point

    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
