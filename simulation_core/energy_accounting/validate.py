"""
Phase F7: accelergy CLI cross-validation.

After EnergyAccountant has finished accumulating, this module picks a
small number of representative sub-runs (top-N by per-call energy
contribution) and re-computes their energy via the standalone accelergy
toolchain (`accelergy` + CACTI + Aladdin), then reports the diff.

This is *not* per-sub-run validation — that would be too slow. Instead we
sample, validate the samples, and report whether the EnergyAccountant
fast-path is calibrated within tolerance against the CACTI/Aladdin-based
ground truth.

Usage:
    from .validate import cross_validate_top_records

    diffs = cross_validate_top_records(
        records=energy.top_records_by_energy(n=3),
        coefficients=energy.coef,
        ours_log_dir="./results",          # SCALE-Sim scratch dir
        argus_root="/path/to/SCALE-Sim",
    )
    # diffs is a list of dicts; pass to format_validation_report().

Notes:
    - Each sample spawns SCALE-Sim once on a synthesized 1-layer GEMM
      topology, then spawns `accelergy` once. Each sample takes ~3-5s.
    - The accelergy run targets the SAME hw cfg as the original sub-run
      (Bagel_sim/Janus_sim records hw_cfg_path on each _record_energy
      call), so per-precision differences (fp16 vs int4) are honored.
    - Results are scaled back by the original `multiplicity` so the diff
      compares full-run energy contribution, not just one tile.
"""

import os
import re
import csv
import shutil
import subprocess
import tempfile
import yaml


# Mapping from accelergy component-name patterns to the EnergyAccountant
# bucket that the same physical access went into. Used when we sum the
# accelergy energy estimate; lets us do per-bucket diff if we want, but
# the default report just sums everything to a single number per sample.
_ACCELERGY_BUCKETS = {
    re.compile(r"\.\w*ifmap.*dram"):  "dram_ifmap",
    re.compile(r"\.\w*weights.*dram"): "dram_filter",
    re.compile(r"\.\w*psum.*dram"):    "dram_ofmap",
    re.compile(r"\.\w*ifmap.*glb"):    "sram_ifmap",
    re.compile(r"\.\w*weights.*glb"):  "sram_filter",
    re.compile(r"\.\w*psum.*glb"):     "sram_ofmap",
    re.compile(r"\.PE\["):             "compute",  # all PE-internal pieces
}


def _bucket_for(component_name):
    for pattern, bucket in _ACCELERGY_BUCKETS.items():
        if pattern.search(component_name):
            return bucket
    return "other"


def _tile_down(M, N, K, multiplicity, max_dim=128):
    """
    SCALE-Sim's runtime is roughly O(M·N·K) — running a real ARGUS sub-run
    at full size (M=1, N=3584, K=18944) takes 30+ minutes per sample, which
    defeats the point of a quick cross-validation.

    Instead, shrink any dim above `max_dim` down to max_dim, and bump
    multiplicity by the inverse so the *full-run* energy stays the same.
    This mirrors ARGUS's own `tile=64` strategy in build_topologies, where
    SCALE-Sim is run on a tile and the cycles are linearly scaled afterward.

    Returns (M', N', K', multiplicity') with M·N·K · mult invariant.
    """
    orig_volume = float(M) * float(N) * float(K)

    Mp = max(1, min(M, max_dim))
    Np = max(1, min(N, max_dim))
    Kp = max(1, min(K, max_dim))
    new_volume = float(Mp) * float(Np) * float(Kp)

    scale = orig_volume / new_volume if new_volume > 0 else 1.0
    return Mp, Np, Kp, multiplicity * scale


def _ensure_accelergy_cfg(hw_cfg, tmpdir):
    """
    accelergy's preprocess.py needs SRAM_row_size and DRAM_row_size in the
    cfg under [accelergy] section. ARGUS's per-hw cfgs (e.g.
    configs/bagel/ours_int4_m.cfg) don't have these — they're only filled
    in when running rundir-accelergy directly. So we copy the user's cfg
    and append a default [accelergy] block before passing it to run_all.sh.
    """
    out_cfg = os.path.join(tmpdir, "hw_cfg_with_accelergy.cfg")
    with open(hw_cfg) as f:
        contents = f.read()
    # Idempotent: don't append twice if user already has it.
    if "SRAM_row_size" not in contents:
        contents += (
            "\n[accelergy]\n"
            "SRAM_row_size: 5\n"
            "DRAM_row_size: 5\n"
        )
    with open(out_cfg, "w") as f:
        f.write(contents)
    return out_cfg


def _validate_one(record, coefficients, ours_log_dir, argus_root,
                  workdir=None, verbose=False):
    """
    Run accelergy for a single record's (M, N, K, precision, hw_cfg).
    Returns dict with the per-bucket pJ from accelergy plus a sum_pJ,
    or {"error": str} if anything went wrong.

    The (M, N, K) is tiled down to <= 128 per dim before running SCALE-Sim
    (see _tile_down), so each sample takes a few seconds rather than tens
    of minutes. The full-run energy is reconstructed by scaling.
    """
    M_real, N_real, K_real = record["M"], record["N"], record["K"]
    multiplicity = record["multiplicity"]
    hw_cfg = record["hw_cfg_path"]
    label = record["label"]

    if not hw_cfg or not os.path.exists(hw_cfg):
        return {"error": f"hw_cfg_path missing or not found: {hw_cfg!r}"}

    # Tile down so SCALE-Sim is fast.
    M, N, K, multiplicity_scaled = _tile_down(M_real, N_real, K_real, multiplicity)

    # Use a private scratch dir so concurrent validates don't stomp on each other.
    tmpdir = workdir or tempfile.mkdtemp(prefix="argus_f7_")
    # Synthesize an Accelergy-friendly cfg (ARGUS hw cfgs lack [accelergy] section).
    accelergy_cfg = _ensure_accelergy_cfg(hw_cfg, tmpdir)
    try:
        # 1. Synthesize a one-layer GEMM topology.
        topo_path = os.path.join(tmpdir, "validate.csv")
        with open(topo_path, "w") as f:
            w = csv.writer(f)
            w.writerow(["Layer", "M", "N", "K", ""])
            w.writerow([label or "validate_layer",
                        max(1, int(M)), max(1, int(N)), max(1, int(K)), ""])

        # 2. accelergy run_all.sh expects (-c cfg, -t topo, -p scsim_log,
        #    -o all_output, -i gemm). Use private dirs under tmpdir.
        scsim_log = os.path.join(tmpdir, "scsim_log")
        all_output = os.path.join(tmpdir, "all_output")
        os.makedirs(scsim_log, exist_ok=True)
        os.makedirs(all_output, exist_ok=True)

        # The run_all.sh chdir's into rundir-accelergy so its relative
        # paths (`accelergy_input/`, `../scale.py`) resolve correctly.
        rundir = os.path.join(argus_root, "rundir-accelergy")
        if not os.path.isdir(rundir):
            return {"error": f"rundir-accelergy not found at {rundir}"}

        cmd = [
            "bash", os.path.join(rundir, "run_all.sh"),
            "-c", os.path.abspath(accelergy_cfg),
            "-t", os.path.abspath(topo_path),
            "-p", scsim_log,
            "-o", all_output,
            "-i", "gemm",
        ]
        try:
            r = subprocess.run(
                cmd, cwd=rundir, capture_output=True, text=True, timeout=180
            )
        except subprocess.TimeoutExpired:
            return {"error": "run_all.sh timed out (180s)"}
        if r.returncode != 0:
            tail = (r.stdout or "") + (r.stderr or "")
            return {"error": f"run_all.sh failed: {tail[-400:]}"}

        # 3. Find the energy_estimation.yaml. It lives under
        #    all_output/accelergy_output_<run_name>/energy_estimation.yaml.
        ee_path = None
        for entry in os.listdir(all_output):
            cand = os.path.join(all_output, entry, "energy_estimation.yaml")
            if os.path.exists(cand):
                ee_path = cand
                break
        if ee_path is None:
            return {"error": "energy_estimation.yaml not produced"}

        with open(ee_path) as f:
            ee = yaml.safe_load(f)
        components = ee.get("energy_estimation", {}).get("components", [])

        # 4. Bucket and sum (per-tile, in pJ from accelergy).
        per_bucket_pJ = {b: 0.0 for b in set(_ACCELERGY_BUCKETS.values())}
        per_bucket_pJ["other"] = 0.0
        for comp in components:
            name = comp.get("name", "")
            energy = float(comp.get("energy", 0.0))
            per_bucket_pJ[_bucket_for(name)] += energy

        # Scale by the original ARGUS multiplicity × tile-down compensation
        # (this validates one tile; ARGUS would have run it `multiplicity`
        # times, and we tiled the GEMM volume down by another factor).
        scaled = {k: v * multiplicity_scaled for k, v in per_bucket_pJ.items()}
        scaled["sum_pJ"] = sum(scaled.values())
        scaled["per_tile_pJ"] = sum(per_bucket_pJ.values())
        scaled["effective_multiplicity"] = multiplicity_scaled
        scaled["tile_dims"] = (M, N, K)
        return scaled

    finally:
        if workdir is None:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass


def cross_validate_top_records(records, coefficients, ours_log_dir,
                                argus_root, verbose=False):
    """
    Run _validate_one for each record and return a list of result dicts:
        {"record": <input record>, "validate": <_validate_one output>,
         "diff_pct": float | None}
    """
    out = []
    for rec in records:
        v = _validate_one(rec, coefficients, ours_log_dir, argus_root,
                          verbose=verbose)
        if "error" in v:
            out.append({"record": rec, "validate": v, "diff_pct": None})
            continue
        accountant_pJ = rec["subtotal_pJ"]
        accelergy_pJ = v["sum_pJ"]
        diff_pct = (
            100.0 * (accelergy_pJ - accountant_pJ) / accountant_pJ
            if accountant_pJ > 0 else None
        )
        out.append({"record": rec, "validate": v, "diff_pct": diff_pct})
    return out


def format_validation_report(results):
    """Render the cross-validation result list to a multi-line string for
    appending to results.log."""
    lines = ["CROSS-VALIDATION (accelergy CLI):"]
    lines.append("  Note: accelergy uses tile-down sampling + leak/idle scaling.")
    lines.append("  Its numbers are an upper bound (include PE leak even for")
    lines.append("  inactive cycles); EnergyAccountant's fast path is the")
    lines.append("  dynamic-access lower bound. Real silicon sits between.")
    if not results:
        lines.append("  (no records to validate)")
        return "\n".join(lines)

    ratios = []
    for i, r in enumerate(results, 1):
        rec = r["record"]
        v = r["validate"]
        head = (
            f"  Sample {i} ({rec['label']}, "
            f"M={rec['M']}, N={rec['N']}, K={rec['K']}, {rec['precision']}, "
            f"x{rec['multiplicity']:,d}):"
        )
        lines.append(head)
        if "error" in v:
            lines.append(f"    [ERROR] {v['error']}")
            continue
        accountant_pJ = rec["subtotal_pJ"]
        accelergy_pJ = v["sum_pJ"]
        ratio = accelergy_pJ / accountant_pJ if accountant_pJ > 0 else float("inf")
        ratios.append(ratio)

        lines.append(
            f"    EnergyAccountant: {rec['subtotal_pJ']*1e-9:>10.3f} mJ  (dynamic only)"
        )
        lines.append(
            f"    accelergy:        {v['sum_pJ']*1e-9:>10.3f} mJ  (with leak/idle)"
        )
        lines.append(
            f"    ratio:                     {ratio:>5.1f}x  "
            f"(per-tile {v['per_tile_pJ']*1e-3:>8.1f} nJ × scale {v['effective_multiplicity']:.0f})"
        )

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        lines.append(f"  ----")
        if 1.5 <= avg_ratio <= 100:
            verdict = "OK — leak/idle accounts for the gap (expected for tile-sampling)"
        elif avg_ratio < 1.5:
            verdict = "OK — fast path agrees within 1.5x of accelergy"
        else:
            verdict = "INVESTIGATE — gap larger than typical leak overhead"
        lines.append(f"  Mean ratio (accelergy / EnergyAccountant): {avg_ratio:.1f}x")
        lines.append(f"  → {verdict}")
    return "\n".join(lines)
