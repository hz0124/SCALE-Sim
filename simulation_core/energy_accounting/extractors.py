"""
Access-count extractor for the energy accountant.

    from_geometric_estimate(M, N, K, arr_h, arr_w, *, dataflow='os')
        Closed-form estimate of SRAM/DRAM access counts for one GEMM. No
        SCALE-Sim invocation. This is the single source of access counts for
        EnergyAccountant.

A second extractor (from_scalesim_reports) once parsed SCALE-Sim's
DETAILED_ACCESS_REPORT.csv directly, but it was removed: ARGUS runs each
sub-run as a small N=tile GEMM, and on tiles smaller than the SRAM the
report's DRAM counts are dominated by fixed-size prefetch-buffer fills
(DRAM IFMAP Reads stayed ~32768 regardless of M/N/K) rather than real
traffic — see DEVLOG #4. For M=1 token-by-token decode the closed form has
no weight reuse to spill, so it is essentially exact for DRAM.

Returns a flat dict (SRAM/DRAM access counts only — MAC ops are tracked
separately via EnergyAccountant.add_mac_ops(M * N * K, precision)):

    {
        "sram_ifmap_reads":  int, "sram_filter_reads":  int,
        "sram_ofmap_writes": int, "dram_ifmap_reads":   int,
        "dram_filter_reads": int, "dram_ofmap_writes":  int,
    }
"""

import math


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
