# ARGUS Simulation Framework

This repository contains the hardware modeling and simulation framework for the paper **"ARGUS: Enabling Efficient MLLM Inference through Hierarchical Compression and Algorithm-Hardware-Datatype Co-design"** (Submitted to **DAC 2026**).

Built upon [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim), this framework is designed to evaluate the decoding stage throughput performance of Unified Multimodal Large Language Models (MLLMs). It supports modeling our proposed **ARGUS** architecture as well as several state-of-the-art baseline accelerators (S-DMA, FlightVGM, FIGNA, AxCore).

## Supported Models & Tasks

The framework supports simulation for the following MLLMs and tasks:

*   **Bagel**:
    *   **GenEdit**: Image Editing
    *   **GenEval**: Image Generation
    *   **MM**: Image Understanding (Multimodal)
*   **Janus**:
    *   **GenEval**: Image Generation
    *   **MM**: Image Understanding

## Project Structure

The core logic is encapsulated within `simulation_core`, while external scripts handle argument parsing and execution.

```text
SCALE-Sim/
├── run_bagel.py              # Entry point for Bagel model simulation
├── run_janus.py              # Entry point for Janus model simulation
├── simulation_core/          # Core simulation logic
│   ├── bagel_sim.py          # Bagel simulation class
│   └── janus_sim.py          # Janus simulation class
├── topologies/               # Simulation flow configurations (JSON)
│   ├── bagel/                # Configs for Bagel tasks
│   └── janus/                # Configs for Janus tasks
├── configs/                  # Hardware architecture specifications (CFG)
│   └── bagel/                # Hardware specs (Array size, SRAM, Bandwidth)
└── results/                  # Simulation logs and outputs
```

## Usage

To run a simulation, use the provided entry scripts (`run_bagel.py` or `run_janus.py`). You need to specify the hardware architecture, the task type, and the path to the simulation configuration file.

### Arguments

*   `--hw`: The hardware architecture to simulate. Options: `ours`, `ours_balence`, `sdma`, `flightvgm`, `figna`, `base`.
*   `--task`: The task to evaluate. Options: `GenEdit`, `GenEval`, `MM`.
*   `--config`: Path to the specific JSON configuration file for the simulation run.

### Example Command

To simulate the **ARGUS (Ours)** hardware running the **GenEdit** task on the **Bagel** model:

```bash
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json
```

To simulate the Janus model on FlightVGM hardware for the **MM** task:

```bash
python run_janus.py --hw flightvgm --task MM --config ./topologies/janus/config_flightvgm.json
```
## Configuration System
The framework uses a two-level configuration system to provide flexibility in modeling both the workload flow and the hardware specifications.

1. Simulation Configuration (`.json`)
Located in `bagel/` or `janus/` directories.
These JSON files control the simulation flow, dataset parameters, and link to specific hardware configs. When creating a custom config, ensure the following keys are defined:

- **KV Cache Settings**: `kv_cache_size` (initial size), `num_heads`, `head_dim`
- **Hardware Config Paths**: Keys like `config_fp16`, `config_int8`, `config_base` point to the `.cfg` files described below.
- **Hardware Specifics**: Parameters like `sparsity`, `activate_rate`, `tile_size`, etc.
- **Output**: `result_path` defines where logs and logs are saved.

Example snippet (`config_ours.json`):
```json
{
    "kv_cache_init": 3254,
    "gen_text_len": 229,
    "config_fp16": "./configs/bagel/ours_fp16.cfg",
    "config_int8": "./configs/bagel/ours_int8.cfg",
    "sparsity_cross_attn": 0.3819
}
``` 
2. Hardware Configuration (`.cfg`)
Located in `bagel/` directory.
These are standard SCALE-Sim configuration files that define the physical hardware constraints.

Architecture: `ArrayHeight, ArrayWidth` (Systolic Array dimensions).
Memory: `IfmapSramSzkB, FilterSramSzkB, OfmapSramSzkB` (On-chip buffer sizes).
System: `Bandwidth` (DRAM bandwidth), `Dataflow`.
Example snippet (`ours_fp16.cfg`):
```
[architecture_presets]
ArrayHeight:    32
ArrayWidth:     32
IfmapSramSzkB:  32
Bandwidth :     8
```

## Energy Accounting (Phase F)

Both `run_bagel.py` and `run_janus.py` accept an opt-in `--energy` flag.
When set, the harness keeps a running tally of MAC ops + SRAM accesses +
DRAM accesses across every sub-run, multiplies by per-component pJ
coefficients, and appends an ENERGY BREAKDOWN section to `results.log`.

```bash
# Cycles only (default — no overhead)
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json

# Cycles + per-run energy report (~1.5x slower for the bookkeeping)
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json --energy
```

The report lines look like:

```
ENERGY BREAKDOWN:
  Compute (MAC):   75213.659 mJ  ( 86.1%)
  SRAM total:        243.810 mJ  (  0.3%)
    ifmap:            80.785 mJ
    filter:          161.571 mJ
    ofmap:             1.454 mJ
  DRAM total:      11923.115 mJ  ( 13.6%)  [type=HBM2]
    ifmap:           969.933 mJ
    filter:         3989.373 mJ
    ofmap:           907.115 mJ
    idle:           6056.694 mJ
  ----
  Total Energy:  87380.584 mJ
  Avg Power:         1.443 W   (cycles 30,283,470,864)
  EDP:          5292374.751 mJ·s
```

### Default coefficients

A single 28nm CMOS baseline is shared by all five hardware types
(`ours / ours_balence / sdma / flightvgm / figna / base`). Sources are
documented in `simulation_core/energy_accounting/coefficients.py`:

| Quantity | Default | Source |
|---|---|---|
| MAC fp16 | 1.5 pJ | Horowitz, ISSCC 2014 |
| MAC int8 | 0.2 pJ | Horowitz, ISSCC 2014 |
| MAC int4 | 0.1 pJ | extrapolated, half of int8 |
| SRAM read/write | 0.05 pJ/byte | Sze et al., Proc IEEE 2017 |
| HBM2 | 3.9 pJ/bit | JEDEC HBM2 spec |
| HBM3 | 3.0 pJ/bit | JEDEC HBM3 spec |
| LPDDR4 | 7.0 pJ/bit | Micron datasheet |
| DDR4 | 20.0 pJ/bit | DDR4 spec |
| DRAM idle background | 100 mW | rough estimate |

Differentiation between the five hardware types comes from access-count
differences (sparsity, KV-cache shrinking, precision selection) that ARGUS
already models elsewhere — *not* from these baseline pJ numbers. Real
cross-hardware energy comparisons need per-hw coefficient tuning (see
override path below).

### Per-hardware override (in workload JSON)

Add `dram_type` and/or `energy_coefficients` to your workload JSON:

```jsonc
{
    "kv_cache_init": 3254,
    "gen_text_len": 229,
    // ... existing fields ...

    "dram_type": "HBM3",           // optional: HBM2 | HBM3 | LPDDR4 | DDR4
    "energy_coefficients": {       // optional: any subset of the keys below
        "mac_pj": {"fp16": 0.8, "int8": 0.15},   // nested merge with defaults
        "sram_read_pj_per_byte":  0.03,
        "sram_write_pj_per_byte": 0.04,
        "dram_pj_per_bit":        2.5,            // overrides dram_type lookup
        "dram_idle_mw":           50,
        "freq_hz":                1000000000      // change clock from 500 MHz
    }
}
```

Precision per sub-run is sniffed from the SCALE-Sim `.cfg` filename suffix
(`*_fp16.cfg` / `*_int8.cfg` / `*_int4.cfg`); ARGUS picks the right pJ key
automatically so you don't have to.

### Cross-validation against accelergy CLI (`--validate`)

For reviewer-ready sanity checks, add `--validate` on top of `--energy`:

```bash
python run_bagel.py --hw ours --task MM --config <cfg> --energy --validate
```

This picks the **top-3 highest-energy sub-runs** (typically `ffn_up`,
`ffn_down`, and a representative `attn`) and re-computes each one through
the standalone Accelergy + CACTI + Aladdin toolchain, then appends a
`CROSS-VALIDATION` section to `results.log`. Output looks like:

```
CROSS-VALIDATION (accelergy CLI):
  Note: accelergy uses tile-down sampling + leak/idle scaling.
  Its numbers are an upper bound (include PE leak even for
  inactive cycles); EnergyAccountant's fast path is the
  dynamic-access lower bound. Real silicon sits between.

  Sample 1 (text/ffn_down, M=1, N=3584, K=18944, int4, x560):
    EnergyAccountant:    598.558 mJ  (dynamic only)
    accelergy:         40712.988 mJ  (with leak/idle)
    ratio:                      68.0x  (per-tile  17543.9 nJ × scale 2320640)
  ...
  Mean ratio (accelergy / EnergyAccountant): 68.0x
  → OK — leak/idle accounts for the gap (expected for tile-sampling)
```

**How to read the ratio.** Don't expect 1.0x — the two tools answer
different questions:

- **EnergyAccountant** (the `--energy` fast-path) reports *dynamic
  access* energy: pJ per MAC, per SRAM-byte, and per DRAM-bit, summed
  across actual access counts. No leakage, no idle background. This is
  the **lower bound**.
- **accelergy** reports the full Accelergy / CACTI / Aladdin estimate,
  which adds per-PE leak + idle for *every* clock the simulated
  accelerator is alive. Because we tile-down the sample GEMM to keep
  the run fast (1×128×128 instead of e.g. 1×3584×18944, see
  `_tile_down` in `validate.py`), the leak component gets multiplied
  by the same volume-restoration factor — pushing ratios into the
  10×–100× range. This is a **leak-dominated upper bound**.
- **Real silicon** sits in between, depending on activity factor and
  clock-gating effectiveness.

**What ratios mean in practice:**

| Mean ratio | Verdict |
|---|---|
| 1× – 1.5× | Fast path is well calibrated, both tools agree |
| 10× – 100× | Normal for ARGUS — leak overhead from 1024 PEs dominates |
| Wildly inconsistent across the 3 samples | Investigate: likely a bug in `_record_energy` multiplicity or label routing |
| Ratio drifts run-to-run | Check that `hw_cfg_path` (sniffed in `_record_energy`) is stable |

**Cost.** Each sample spawns a 1×128×128 SCALE-Sim run + a fresh
accelergy invocation, ~5 s each. Three samples + setup ≈ 5–8 minutes
on top of the base simulation.

**Caveats.**
- The 28nm baseline pJ table is *literature-derived*. If you want
  CACTI-derived SRAM/DRAM numbers as your fast-path coefficients,
  read CACTI yourself (e.g. for a 64 KB SRAM at 28nm) and override
  via `energy_coefficients` in the workload JSON. `--validate`
  doesn't update the table for you.
- The `--validate` path requires Accelergy + accelergy-cacti-plug-in
  + accelergy-aladdin-plug-in installed in the active conda env.
  Source clones are typically under `~/.local/src/accelergy-stack/`.
- `rundir-accelergy/` must contain its full upstream content
  (`preprocess.py`, the `accelergy_input/components/*.yaml` library,
  etc.) — these were imported in the Phase A commit `b504b5c`.

### Acknowledgement
This project is based on [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) (Systolic CNN Accelerator Simulator).