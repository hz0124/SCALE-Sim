"""
pJ-per-{op,byte,bit} coefficients for the ARGUS energy accountant.

All defaults are 28nm CMOS literature values for a generic systolic accelerator
with HBM2 backing memory. Numbers are *order-of-magnitude* references, not
calibrated for any specific tape-out. The user is expected to override them per
hardware via `topologies/{bagel,janus}/config_<hw>.json` (see README_MLLM.md).

Sources (cite in any paper using these numbers verbatim):
    [Horowitz14] M. Horowitz, "1.1 Computing's energy problem (and what we
                 can do about it)", ISSCC 2014.
                 → MAC fp16 ~1.5 pJ, MAC int8 ~0.2 pJ at 28nm.
    [Sze17]      V. Sze et al., "Efficient processing of deep neural networks:
                 A tutorial and survey", Proc. IEEE 2017.
                 → SRAM read ~5 pJ for a 32-bit (4 B) word in 32 KB at 45nm
                   ≈ 1.25 pJ/byte; node-scaling 45→28nm (≈×0.45 dynamic
                   energy) ≈ 0.5 pJ/byte.  (The earlier 0.05 value carried a
                   spurious extra ÷10 — corrected 2026-06-15, see §16 below.)
    [JEDEC-HBM2] HBM2 PHY+DRAM stack reports ~3.5–4.0 pJ/bit; we use 3.9.
    [JEDEC-HBM3] HBM3 reports ~2.5–3.5 pJ/bit; we use 3.0.
    [Micron-LP4] LPDDR4 ~5–10 pJ/bit; we use 7.0.
    [DDR4-spec]  DDR4 ~15–25 pJ/bit; we use 20.0.

Only one MAC precision is modelled per call (fp16 / int8 / int4). int4 is
extrapolated from int8 by halving (no widely-cited literature number).

Coefficient accuracy review (framework §16, 2026-06-15, dingli):
  Measured energy split on the 06-13 matrix (bagel ours/figna GenEdit):
  Compute(MAC) 65–80%, DRAM 20–35% (of which idle ~10%), SRAM 0.2–0.5%.
  So the conclusion-sensitive coefficients are the MAC table and DRAM:
    - mac_pj.w4a16 = 0.5 is a PLACEHOLDER and the single most sensitive
      number: figna's energy advantage flips below/above a crossover at
      ≈0.57 pJ (figna is ~90% W4A16 MACs). Defensible literature range for
      an int4-weight × 16-bit-activation MAC is ~0.45–0.6 pJ (between int4
      0.1 and fp16 1.5; the 16-bit activation path dominates). MUST be
      calibrated against RTL post-sim before any energy claim is final.
    - mac_pj.fp16 1.5 / int8 0.2 are Horowitz'14 45nm values used as a 28nm
      proxy (conservative — 28nm would be lower); fine for relative numbers.
    - dram_idle_mw 100 contributes ~10% of total energy via cycles×power —
      a rough placeholder; worth calibrating (affects all hw uniformly).
    - sram_*_pj_per_byte corrected 0.05→0.5 (the 0.05 was 10× low). Because
      SRAM is <0.5% of total this does NOT move any speedup/efficiency
      ratio; the 06-13 results need no re-run on this account.
"""

from copy import deepcopy


# pJ/bit for the DRAM access path. Selected per-hw via the `dram_type` field
# in the workload JSON (defaults to HBM2).
DRAM_PJ_PER_BIT = {
    "HBM2": 3.9,
    "HBM3": 3.0,
    "LPDDR4": 7.0,
    "DDR4": 20.0,
}


# Single 28nm baseline shared by all five hardware types (ours / sdma /
# flightvgm / figna / base). Per the user's design decision: don't pretend to
# know per-hw process / circuit differences. The baseline lets ARGUS produce
# energy numbers that differ by *access count* (which is what ARGUS's own
# sparsity / KV-cache / precision logic already affects) — not by guesswork at
# pJ-per-MAC. Real cross-hw comparisons must override these per-hw in JSON.
_BASELINE_28NM = {
    "node": "28nm",
    "mac_pj": {  # per-MAC energy at the given precision
        "fp16": 1.5,   # Horowitz '14
        "fp8": 0.4,    # FP8xFP8 (~2x int8: exponent/align overhead). Placeholder, 2026-06-18.
        "int8": 0.2,   # Horowitz '14
        "int4": 0.1,   # Extrapolated, half of int8 (pure int4 x int4)
        "fp4": 0.2,    # FP4xFP4 (~2x int4: exponent/align overhead). Placeholder, 2026-06-18.
        # W4A16: int4 weight x fp16/bf16 activation (FIGNA-style FP-INT
        # multiplier, and ARGUS's diffusion-stage INT4 array). NOT pure int4:
        # the 16-bit activation path dominates, so energy sits between int4
        # (0.1) and fp16 (1.5). Default 0.5 is a placeholder — calibrate
        # against RTL post-sim (framework §5 version A) and/or override via
        # the workload JSON "energy_coefficients": {"mac_pj": {"w4a16": ...}}.
        "w4a16": 0.5,
    },
    "sram_read_pj_per_byte":  0.5,    # Sze '17, 1.25 pJ/B @45nm → ~0.5 @28nm
    "sram_write_pj_per_byte": 0.5,    # Symmetric for SRAM, conservative
    "dram_type": "DDR4",              # → 20 pJ/bit (DDR4 off-chip), see DRAM_PJ_PER_BIT (2026-06-17)
    "dram_idle_mw": 500.0,            # DRAM background/standby power × wall-time
                                      # (DDR4-class subsystem ~32GB/s; standby+
                                      # refresh+periphery. Runtime term → penalizes
                                      # slow designs. Placeholder, calibrate to
                                      # target system. 2026-06-17: 100→500 for DDR4.)
    # Compute-array static/leakage + clock-tree power for the FULL physical
    # lane budget (2048 MAC, aligned across all hw). Charged as
    # array_static_mw × wall-time for EVERY hardware (accountant.
    # compute_array_static_pJ) — so a slower design pays MORE static energy
    # (it occupies the array longer). This removes the dynamic-only bias that
    # under-charged slow accelerators (e.g. FIGNA ran 2.6× longer but paid no
    # array leakage; see REBUTTAL_FRAMEWORK.md §6.5 / §8). 2026-06-17.
    # PLACEHOLDER magnitude (leakage+clock not separable from Table-3 totals
    # without RTL); calibrate against RTL (framework §5 version A). Override per
    # workload via JSON "energy_coefficients": {"array_static_mw": ...}; set 0
    # for a pure dynamic-energy (lower-bound) study.
    "array_static_mw": 150.0,
    "freq_hz": 500_000_000,           # ARGUS harness assumes 500 MHz
}


# All five ARGUS-supported hardware types share the baseline. Differentiation
# comes from JSON overrides in `topologies/{bagel,janus}/config_<hw>.json`.
DEFAULT_COEFFICIENTS = {
    "ours":         deepcopy(_BASELINE_28NM),
    "sdma":         deepcopy(_BASELINE_28NM),
    "flightvgm":    deepcopy(_BASELINE_28NM),
    "figna":        deepcopy(_BASELINE_28NM),
    "base":         deepcopy(_BASELINE_28NM),
    # need-9 disaggregated baseline: text-accel (FIGNA-style W4A16) + vision-
    # accel (S-DMA-style fp16), serial. Shares the baseline; its disagg-only
    # idle-leakage term comes from the JSON array_static_mw override.
    "disagg":       deepcopy(_BASELINE_28NM),
    # AxCore: same W4A16 scheme as FIGNA (int4 weight × fp16 activation), but its
    # FP-INT multiplier keeps the baseline w4a16 = 0.5 pJ (FIGNA is 0.57). Behaves
    # identically to figna in the harness dispatch; only this coefficient differs.
    "axcore":       deepcopy(_BASELINE_28NM),
}

# Per-hardware w4a16 MAC energy (int4 weight × fp16 activation), 2026-06-17.
# ARGUS's INT4+BF16 dual-array W4A16 unit is more efficient than FIGNA's FP-INT
# multiplier, so they get distinct (RTL-informed) coefficients rather than the
# shared 0.5 placeholder. disagg's text-accel is FIGNA-style → uses FIGNA's.
# sdma/flightvgm/base never emit w4a16 MACs, so their value is inert.
DEFAULT_COEFFICIENTS["ours"]["mac_pj"]["w4a16"] = 0.47
DEFAULT_COEFFICIENTS["figna"]["mac_pj"]["w4a16"] = 0.57
DEFAULT_COEFFICIENTS["disagg"]["mac_pj"]["w4a16"] = 0.57


def get_coefficients(hw_type, json_overrides=None, dram_type=None):
    """
    Build the per-run coefficient dict.

    Args:
        hw_type: one of {ours, sdma, flightvgm, figna, base, disagg, axcore}.
                 (per-hw w4a16 MAC pJ differs: ours 0.47 / figna·disagg 0.57 / axcore 0.50.)
        json_overrides: dict from cfg["energy_coefficients"], may be None.
                        Keys override the baseline at the leaf level. For nested
                        keys (e.g. mac_pj.fp16), pass {"mac_pj": {"fp16": 1.0}}
                        which merges (not replaces) the inner dict.
        dram_type: optional override of cfg["dram_type"]. If given, sets the
                   coefficient `dram_pj_per_bit` from DRAM_PJ_PER_BIT.

    Returns:
        Coefficient dict with at minimum these keys:
            node, mac_pj, sram_read_pj_per_byte, sram_write_pj_per_byte,
            dram_type, dram_pj_per_bit, dram_idle_mw, freq_hz.
    """
    if hw_type not in DEFAULT_COEFFICIENTS:
        raise KeyError(
            f"Unknown hw_type {hw_type!r}; expected one of "
            f"{sorted(DEFAULT_COEFFICIENTS)}"
        )

    coef = deepcopy(DEFAULT_COEFFICIENTS[hw_type])

    # JSON overrides (merge nested mac_pj dict instead of replacing).
    if json_overrides:
        for key, val in json_overrides.items():
            if key == "mac_pj" and isinstance(val, dict):
                coef["mac_pj"].update(val)
            else:
                coef[key] = val

    # DRAM type → pJ/bit lookup (after override-merge so explicit
    # `dram_pj_per_bit` in json wins over the type-based default).
    effective_type = dram_type or coef["dram_type"]
    if effective_type not in DRAM_PJ_PER_BIT:
        raise KeyError(
            f"Unknown dram_type {effective_type!r}; expected one of "
            f"{sorted(DRAM_PJ_PER_BIT)}"
        )
    coef["dram_type"] = effective_type
    coef.setdefault("dram_pj_per_bit", DRAM_PJ_PER_BIT[effective_type])

    return coef


def precision_from_config_path(cfg_path):
    """
    Heuristic: ARGUS's per-precision SCALE-Sim cfgs are named e.g.
    `ours_fp16.cfg`, `ours_int8.cfg`, `ours_int4.cfg`. Pull the precision
    suffix out so EnergyAccountant.add_mac_ops gets the right `precision`.

    Falls back to fp16 if no marker found (worst case for energy estimate).
    """
    path = (cfg_path or "").lower()
    for suffix in ("int4", "int8", "fp16"):
        if suffix in path:
            return suffix
    return "fp16"
