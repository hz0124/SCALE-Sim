#!/usr/bin/env python
"""
Need-6 context-length sweep, BAGEL + GenEdit, ARGUS all_on (= config_ours.json
defaults, all 5 technique switches + quant_proj on).

For each context length we override kv_cache_init plus the *profiled* sparsity /
reuse values for that length (need6_context_length_sweep.md sec 1 -- longer
context => more cross-attn trimmed (sparsity_cross_attn) and more self-attn in
low precision (low_precise_self_attn); text/image reuse ratios also shift). The
image/text token lengths stay at the config_ours.json originals (image_len=3193,
text_len=242), so kv_cache_without_img/_text are recomputed per point.

This is the all_on/ctxlen line that run_argus_technique_sweep.py cross-checks
(its all_on rows reproduce these). Recreated 2026-06-18 (the original ran in a
throwaway git worktree); the 9 points + profiled params below are taken verbatim
from the prior need6_ctxlen_sweep_results.csv / sec-1 table so the sweep is
identical except for the refreshed energy model (DDR4 + idle 500mW + array
static 150mW + per-hw w4a16) and the run_model log-clear fix.

Outputs need6_ctxlen_sweep_results.{json,csv} (incremental, resumable).

Usage:
  python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_ctxlen_sweep.py
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from simulation_core.bagel_sim import Bagel_sim

OUTDIR = os.path.join(ROOT, "devlog_rebuttal", "overnight_2026-06-15", "need6")
OUT_JSON = os.path.join(OUTDIR, "need6_ctxlen_sweep_results.json")
OUT_CSV = os.path.join(OUTDIR, "need6_ctxlen_sweep_results.csv")
TMPDIR = "/tmp/argus_ctxlen_sweep"

BASE_CFG = os.path.join(ROOT, "topologies", "bagel", "config_ours.json")
IMAGE_LEN = 3193      # config_ours.json original
TEXT_LEN = 242        # config_ours.json original

# context_length -> profiled (sparsity_cross_attn, low_precise_self_attn,
# text_only_sim, image_only_sim). Verbatim from need6_context_length_sweep.md
# sec 1 (== prior need6_ctxlen_sweep_results.csv input columns).
CTX_PARAMS = {
    1497:  dict(sparsity_cross_attn=0.3702, low_precise_self_attn=0.3575, text_only_sim=0.2494, image_only_sim=0.2070),
    2566:  dict(sparsity_cross_attn=0.4090, low_precise_self_attn=0.3867, text_only_sim=0.2723, image_only_sim=0.1829),
    3211:  dict(sparsity_cross_attn=0.4217, low_precise_self_attn=0.4079, text_only_sim=0.2844, image_only_sim=0.1842),
    5648:  dict(sparsity_cross_attn=0.4877, low_precise_self_attn=0.5256, text_only_sim=0.2900, image_only_sim=0.1884),
    7609:  dict(sparsity_cross_attn=0.4978, low_precise_self_attn=0.5457, text_only_sim=0.3021, image_only_sim=0.1927),
    8754:  dict(sparsity_cross_attn=0.5125, low_precise_self_attn=0.5741, text_only_sim=0.2962, image_only_sim=0.1938),
    9909:  dict(sparsity_cross_attn=0.5297, low_precise_self_attn=0.6154, text_only_sim=0.2863, image_only_sim=0.1940),
    11969: dict(sparsity_cross_attn=0.5375, low_precise_self_attn=0.6256, text_only_sim=0.2536, image_only_sim=0.1932),
    12018: dict(sparsity_cross_attn=0.5338, low_precise_self_attn=0.6237, text_only_sim=0.2636, image_only_sim=0.1873),
}
CTX_LENGTHS = sorted(CTX_PARAMS)

OPS = ["text/qkv", "text/attn", "text/omap", "text/ffn_up", "text/ffn_down",
       "img/qkv", "img/attn", "img/omap", "img/ffn_up", "img/ffn_down"]


def canon_op(label):
    stage = "text" if label.startswith("text/") else "img"
    rest = label.split("/", 1)[1]
    for op in ("attn", "qkv", "omap", "ffn_up", "ffn_down"):
        if rest.startswith(op):
            return f"{stage}/{op}"
    return f"{stage}/{rest}"


def run_one(ctx, base):
    params = CTX_PARAMS[ctx]
    cfg = dict(base)
    cfg.update(params)
    cfg["kv_cache_init"] = ctx
    cfg["result_path"] = os.path.join(TMPDIR, f"log_ctx_{ctx}.log")
    cfg_path = os.path.join(TMPDIR, f"cfg_ctx_{ctx}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    b = Bagel_sim(hardware_type="ours", task="GenEdit")
    b.read_from_json(cfg_path=cfg_path)
    b.kv_cache_init = ctx
    b.kv_cache_without_img = ctx - IMAGE_LEN
    b.kv_cache_without_text = ctx - TEXT_LEN
    b.batch_size = 1
    b.energy_enabled = True
    b.setup_energy()
    total_cycles = b.run_model()

    eb = b.energy.breakdown()
    energy_by_op = {}
    for r in b.energy.records:
        k = canon_op(r["label"])
        energy_by_op[k] = energy_by_op.get(k, 0.0) + r["subtotal_pJ"]

    return {
        "context_length": ctx,
        "total_cycles": total_cycles,
        "seconds": total_cycles / 500e6,
        "energy_mJ": eb["total_mJ"],
        "compute_mJ": eb["compute_pJ"] * 1e-9,
        "sram_mJ": eb["sram_pJ"] * 1e-9,
        "dram_mJ": eb["dram_pJ"] * 1e-9,
        "avg_power_W": eb["avg_power_W"],
        "edp_mJs": eb["edp_pJs"] * 1e-9,
        "sparsity_cross_attn": params["sparsity_cross_attn"],
        "low_precise_self_attn": params["low_precise_self_attn"],
        "text_only_sim": params["text_only_sim"],
        "image_only_sim": params["image_only_sim"],
        "kv_cache_without_img": ctx - IMAGE_LEN,
        "cycle_breakdown": dict(b.cycle_breakdown),
        "energy_by_op_mJ": {k: v * 1e-9 for k, v in energy_by_op.items()},
    }


def write_outputs(rows):
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["context_length", "total_cycles", "seconds", "energy_mJ",
            "compute_mJ", "sram_mJ", "dram_mJ", "avg_power_W", "edp_mJs",
            "sparsity_cross_attn", "low_precise_self_attn", "text_only_sim",
            "image_only_sim", "kv_cache_without_img"]
    cols += [f"cyc:{op}" for op in OPS] + [f"e_mJ:{op}" for op in OPS]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            if "error" in r:
                w.writerow([r["context_length"], "ERROR: " + r["error"]])
                continue
            row = [r.get(c, "") for c in cols[:14]]
            row += [r["cycle_breakdown"].get(op, 0) for op in OPS]
            row += [r["energy_by_op_mJ"].get(op, 0) for op in OPS]
            w.writerow(row)


def main():
    os.makedirs(TMPDIR, exist_ok=True)
    with open(BASE_CFG) as f:
        base = json.load(f)
    rows = []
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as f:
                rows = [r for r in json.load(f) if "error" not in r]
        except Exception:
            rows = []
    done = {r["context_length"] for r in rows}

    for ctx in CTX_LENGTHS:
        if ctx in done:
            print(f"===== ctx={ctx} (cached, skip) =====", flush=True)
            continue
        print(f"\n===== ctx={ctx} =====", flush=True)
        try:
            rows.append(run_one(ctx, base))
            r = rows[-1]
            print(f"  cycles={r['total_cycles']:.0f}  "
                  f"energy={r['energy_mJ']:.3f} mJ  power={r['avg_power_W']:.2f} W",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}", flush=True)
            rows.append({"context_length": ctx, "error": str(e)})
        write_outputs(rows)

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
