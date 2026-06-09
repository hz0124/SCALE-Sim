"""
Access-count extractors for the energy accountant.

Two modes, matching the two ARGUS sub-run paths:

    from_scalesim_reports(detail_csv, scale=1.0, layer_id=0)
        Parse SCALE-Sim's DETAILED_ACCESS_REPORT.csv. ARGUS calls
        run_sim_once with a "tile" GEMM (one row mapped to a small N=tile),
        then linearly scales the result by `dim/tile` outside. We accept a
        `scale` multiplier so the same scale applies to the access counts.
        Returns counts in element-count (words) so the caller can multiply by
        the per-precision word_bytes.

    from_geometric_estimate(M, N, K, arr_h, arr_w, *, dataflow='os')
        Closed-form estimate. No SCALE-Sim invocation. Used by ARGUS's
        run_sim_once_comp path, which itself is purely analytical.

Both return the same flat dict (SRAM/DRAM access counts only — MAC ops are
*not* derivable cleanly from the SCALE-Sim CSV alone, since the *_Read Counts
columns track PE-level reads which are mac_ops × col_fold in OS dataflow.
The caller knows (M, N, K) from build_topologies, so it should call
EnergyAccountant.add_mac_ops(M * N * K, precision) directly):

    {
        "sram_ifmap_reads":      int,
        "sram_filter_reads":     int,
        "sram_ofmap_writes":     int,
        "dram_ifmap_reads":      int,
        "dram_filter_reads":     int,
        "dram_ofmap_writes":     int,
    }

Calibration (sanity check vs SCALE-Sim, 64×64×64 GEMM on 32×32 OS array):
    SCALE-Sim:   ifmap=4096   filter=4096   ofmap=4096   sram_ifmap=8192
    Geometric:   ifmap=4096   filter=4096   ofmap=4096   sram_ifmap=8192
    → matches exactly for this tile-aligned case.
"""

import math
import pandas as pd


# Column names in DETAILED_ACCESS_REPORT.csv. We read by header (not index)
# so the order can shift without breaking us. Whitespace-insensitive.
_DETAIL_COLS = {
    "sram_ifmap_reads":  "SRAM IFMAP Reads",
    "sram_filter_reads": "SRAM Filter Reads",
    "sram_ofmap_writes": "SRAM OFMAP Writes",
    "dram_ifmap_reads":  "DRAM IFMAP Reads",
    "dram_filter_reads": "DRAM Filter Reads",
    "dram_ofmap_writes": "DRAM OFMAP Writes",
}


def from_scalesim_reports(detail_csv_path, scale=1.0, layer_id=0):
    """
    Parse a SCALE-Sim DETAILED_ACCESS_REPORT.csv and return access counts
    (in elements/words, multiplied by `scale` for the ARGUS dim/tile blowup).
    """
    df = pd.read_csv(
        detail_csv_path,
        sep=r"\s*,\s*",
        engine="python",
        skipinitialspace=True,
    )
    # pandas may keep trailing whitespace in column names; strip them.
    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[layer_id]

    return {
        key: int(int(row[col]) * scale)
        for key, col in _DETAIL_COLS.items()
    }


def from_geometric_estimate(M, N, K, arr_h, arr_w, *, dataflow="os"):
    """
    Closed-form access-count estimate for one GEMM (M × K) · (K × N) → (M × N)
    on an arr_h × arr_w systolic array. Assumes:

      - DRAM is read once per tensor (ifmap M×K, filter K×N, ofmap M×N),
        i.e. SRAM is sized to hold each operand. Real SCALE-Sim allows
        re-fetching when the SRAM is smaller; we ignore that here for
        simplicity (worst case ±2x for tight budgets, ±0% for generous).
        For M=1 token-by-token decode there is no weight reuse to spill, so
        this closed form is essentially exact for DRAM — unlike the tiled
        SCALE-Sim report, whose DRAM counts are dominated by fixed-size
        prefetch-buffer fills on the sub-array tile (see DEVLOG #4).
      - SRAM reads scale with MAC ops divided by the array dimension along
        which an operand is reused (rows for filter, cols for ifmap in OS;
        same total either way for this simple model).
      - SRAM ofmap writes scale with output element count M×N.

    The `dataflow` arg is accepted for symmetry with future variations but
    currently doesn't change the formulas (the totals are dataflow-agnostic
    at this granularity; only per-PE register-file traffic differs, which
    we don't model).

    Returns SRAM/DRAM access counts only. MAC ops = M × N × K — caller must
    pass that to EnergyAccountant.add_mac_ops directly with the right
    precision tag.
    """
    if M <= 0 or N <= 0 or K <= 0:
        return {
            "sram_ifmap_reads": 0, "sram_filter_reads": 0, "sram_ofmap_writes": 0,
            "dram_ifmap_reads": 0, "dram_filter_reads": 0, "dram_ofmap_writes": 0,
        }

    M, N, K = int(M), int(N), int(K)
    arr_h = max(1, int(arr_h))
    arr_w = max(1, int(arr_w))

    # DRAM: one fetch per tensor (no spill). Operand roles for a GEMM
    # (M×K)·(K×N)→(M×N): ifmap = M×K (input), filter = K×N (weights),
    # ofmap = M×N (output). Verified against SCALE-Sim's DETAILED_ACCESS_REPORT
    # (M=1,N=64,K=3584: SRAM IFMAP Reads=3584=M·K, DRAM OFMAP Writes=64=M·N).
    dram_ifmap  = M * K
    dram_filter = N * K
    dram_ofmap  = M * N

    # SRAM: each ifmap element is read once per output column it contributes
    # to, i.e. K times conceptually but we model it at fold granularity:
    # there are ceil(M/arr_h) × ceil(K/arr_w) folds, each issues arr_h × N
    # ifmap reads. Sum: row_fold × col_fold × arr_h × N.
    row_fold = math.ceil(M / arr_h)
    col_fold = math.ceil(K / arr_w)
    sram_ifmap  = row_fold * col_fold * arr_h * N
    sram_filter = row_fold * col_fold * arr_w * N

    # OFMAP writes: one write per output element = M × N. SCALE-Sim adds a
    # small skew tail of arr_h+arr_w-2 per fold; we ignore it.
    sram_ofmap = M * N

    return {
        "sram_ifmap_reads":  sram_ifmap,
        "sram_filter_reads": sram_filter,
        "sram_ofmap_writes": sram_ofmap,
        "dram_ifmap_reads":  dram_ifmap,
        "dram_filter_reads": dram_filter,
        "dram_ofmap_writes": dram_ofmap,
    }
