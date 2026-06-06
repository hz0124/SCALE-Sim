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
                 → SRAM read ~5 pJ for a 32-bit word in 32 KB at 45nm
                   ≈ 0.05 pJ/byte after node scaling.
    [JEDEC-HBM2] HBM2 PHY+DRAM stack reports ~3.5–4.0 pJ/bit; we use 3.9.
    [JEDEC-HBM3] HBM3 reports ~2.5–3.5 pJ/bit; we use 3.0.
    [Micron-LP4] LPDDR4 ~5–10 pJ/bit; we use 7.0.
    [DDR4-spec]  DDR4 ~15–25 pJ/bit; we use 20.0.

Only one MAC precision is modelled per call (fp16 / int8 / int4). int4 is
extrapolated from int8 by halving (no widely-cited literature number).
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
        "int8": 0.2,   # Horowitz '14
        "int4": 0.1,   # Extrapolated, half of int8
    },
    "sram_read_pj_per_byte":  0.05,   # Sze '17 scaled to 28nm
    "sram_write_pj_per_byte": 0.05,   # Symmetric for SRAM, conservative
    "dram_type": "HBM2",              # → 3.9 pJ/bit, see DRAM_PJ_PER_BIT
    "dram_idle_mw": 100.0,            # Background HBM stack power (rough)
    "freq_hz": 500_000_000,           # ARGUS harness assumes 500 MHz
}


# All five ARGUS-supported hardware types share the baseline. Differentiation
# comes from JSON overrides in `topologies/{bagel,janus}/config_<hw>.json`.
DEFAULT_COEFFICIENTS = {
    "ours":         deepcopy(_BASELINE_28NM),
    "ours_balence": deepcopy(_BASELINE_28NM),
    "sdma":         deepcopy(_BASELINE_28NM),
    "flightvgm":    deepcopy(_BASELINE_28NM),
    "figna":        deepcopy(_BASELINE_28NM),
    "base":         deepcopy(_BASELINE_28NM),
}


def get_coefficients(hw_type, json_overrides=None, dram_type=None):
    """
    Build the per-run coefficient dict.

    Args:
        hw_type: one of {ours, ours_balence, sdma, flightvgm, figna, base}.
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
