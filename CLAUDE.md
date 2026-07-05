# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

This repo is a fork of SCALE-Sim v3 customized for the **ARGUS** project (DAC 2026 submission / MICRO 2026 rebuttal, see `README_ARGUS.md`). On top of the upstream cycle-accurate systolic-array simulator it adds an MLLM-decoding-throughput harness for **Bagel** and **Janus**, plus several baseline accelerators (`ours`, `sdma`, `flightvgm`, `figna`, `base`). Most active development happens through the MLLM harness, not direct `scalesim.scale` invocations.

There are two parallel ways to run the simulator; expect to use both:

1. **Upstream single-workload path** — `python -m scalesim.scale -c <cfg> -t <topology.csv> -p <outdir>` (or `python scalesim/scale.py …`). Drives one config + one topology end-to-end and writes `COMPUTE_REPORT.csv`, `BANDWIDTH_REPORT.csv`, `DETAILED_ACCESS_REPORT.csv`, plus per-layer SRAM/DRAM traces. Pass `-i gemm` for MNK topologies.
2. **MLLM harness** — `run_bagel.py` / `run_janus.py`. These orchestrate many `scalesim` sub-runs to model end-to-end decode of an MLLM and aggregate cycle counts in `results/.../results.log`.

## Common commands

```bash
# Activate the project env (sets up conda + HF mirrors)
source env.sh                  # activates conda env zhb-scalesim-v3

# Install in editable mode if you'll modify scalesim/*
pip install -e .

# MLLM harness (primary entry points for this fork)
python run_bagel.py --hw ours      --task GenEdit --config ./topologies/bagel/config_ours.json
python run_bagel.py --hw flightvgm --task MM      --config ./topologies/bagel/config_flightvgm_MM.json
python run_janus.py --hw ours      --task GenEval --config ./topologies/janus/config_ours_janus_GenEval.json

# Upstream single-run path (CSV topology = conv layers; -i gemm for MNK)
python scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/Resnet18.csv -p ./results
python scalesim/scale.py -c configs/scale.cfg -t topologies/GEMM_mnk/vit_s.csv -p ./results -i gemm

# Regression diff (golden traces). Compares fresh run to test/general/golden_trace_calc.
bash test/general/scripts/diff_calc.sh         # also: diff_user_is.sh / _os.sh / _ws.sh
bash test/sparsity/scripts/function_test.sh    # sparsity feature regression

# Ramulator-coupled DRAM stall study (requires the ramulator submodule built)
./run_ramulator.sh <topology_basename> <tag>   # e.g. ./run_ramulator.sh Resnet18 run1
```

Notes:
- `env.sh` sources the user's conda (`zhb-scalesim-v3`) and HF mirror; it is the expected environment, not a generic venv. The `Makefile` venv flow exists but isn't what the harness expects.
- The diff scripts mutate `configs/scale.cfg` and `scalesim/scale.py` in place via `sed` before running; revert those edits before committing if you ran them locally.

## High-level architecture

### Upstream cycle-accurate engine (`scalesim/`)

`scalesim` is `scalesim → simulator → single_layer_sim → {compute, memory, layout}`:

- `scalesim/scale_sim.py::scalesim` is the public façade. It owns parsed `scale_config`, `topologies`, and `layouts` objects and delegates to `simulator`.
- `scalesim/simulator.py` instantiates one `single_layer_sim` per topology row and aggregates cycles, bandwidths, and traces into the three CSV reports.
- `scalesim/single_layer_sim.py` wires together a dataflow-specific compute model and the memory subsystem for one layer.
- `scalesim/compute/` holds the three dataflow implementations: `systolic_compute_ws.py`, `systolic_compute_os.py`, `systolic_compute_is.py` (plus `operand_matrix.py` and sparsity `compression.py`). Dataflow is selected by `Dataflow : ws|os|is` in the cfg.
- `scalesim/memory/` models double-buffered scratchpads (`double_buffered_scratchpad_mem.py`) backed by `read_buffer*.py` / `write_buffer*.py` and `read_port` / `write_port`. The bandwidth mode is governed by `InterfaceBandwidth: USER|CALC` — USER mode applies a fixed BW from cfg, CALC mode estimates the stall-free BW.
- `scalesim/layout_utils.py` + the `[layout]` cfg section model multi-bank SRAM with custom data layouts (`intraline_factor` / `intraline_order` / `interline_order`), used for bank-conflict studies.

A `[general]` cfg's `run_name` becomes the output sub-directory under `-p`.

### MLLM simulation harness (`simulation_core/` + `run_*.py`)

This is the layer that makes ARGUS-style end-to-end decode evaluation possible. It is *not* a fork of upstream `scalesim` — it calls into it.

- `run_bagel.py` / `run_janus.py` are thin argparse wrappers. They instantiate `Bagel_sim` / `Janus_sim` with `--hw` and `--task`, load the JSON, and call `run_model()`.
- `simulation_core/bagel_sim.py::Bagel_sim` and `simulation_core/janus_sim.py::Janus_sim` orchestrate a full decode trajectory:
  - `read_from_json(cfg_path)` — loads workload + hardware paths. The JSON keys it consumes depend on `hardware_type`:
    - `ours` / `figna`: separate cfgs per precision (`config_fp16`, `config_int4`, plus `config_ffn_text` for ARGUS's text-stage FFN slot) → fields `config_comp0/1`, `config_comm0/1`, `config_ffn_text`. `ours` also reads the five ARGUS technique switches `t1_soft`/`t1_hard`/`t2_soft`/`t2_hard`/`t3_soft` (default all true) and `array_width_fp`.
    - `base` / `flightvgm` / `sdma`: a single `config` key reused for all slots.
  - `build_topologies(kv_len, is_gen_text, part)` writes a fresh single-layer GEMM topology to `topologies/{bagel,janus}/layer.csv` for each sub-run; `part` selects which projections (`qkv`, `attn_qk`, `attn_sfmxv`, `omap`, `ffn_up`, `ffn_down`).
  - `run_sim_once(...)` invokes upstream `scalesim` (GEMM mode, layout `layouts/GEMM_mnk/vit_l_KM_KN.csv`) on that one-layer topology and returns its cycle count.
  - `run_sim_once_comp(...)` is an **analytical** alternative: it computes systolic-array fold cycles directly from `array_height/width`, `tile`, and the operation sizes — no SCALE-Sim invocation. Image generation mixes both: text-side projections go through `run_sim_once`, image-side go through `run_sim_once_comp` (much faster for inner loops).
  - `run_gen_text()` / `run_gen_image()` compose per-step cycles, scale by `num_layer`, head count, sample stride, and KV-cache length, then sum into `total_cycles_all`. Sparsity is modelled by shrinking the effective KV length per hardware type (`ours` uses `sparsity_cross_attn` + `low_precise_self_attn`; `sdma` uses `sparsity_kv`; baselines use the full KV).
  - `run_model()` runs text generation, then (for `GenEdit` / `GenImage`) image generation, and writes to `result_path`.

### Two-level configuration

- **`topologies/{bagel,janus}/*.json`** — the simulation flow: KV cache size, `gen_text_len`, `gen_image_step`, sparsity ratios, array dims, buffer sizes & bandwidths, sample rate, and pointers to the `.cfg` files. Naming convention: `config_<hw>[_<task>].json`.
- **`configs/bagel/*.cfg`** — SCALE-Sim hardware specs (`ArrayHeight/Width`, `*SramSzkB`, `Bandwidth`, `Dataflow`, optional `[layout]`, `[sparsity]`). Per-precision variants exist for `ours` and `figna` (`*_fp16.cfg`, `*_int8.cfg`, `*_int4.cfg`).
- **`topologies/<family>/*.csv`** — upstream-style workloads. CONV format by default; the `GEMM_mnk/` tree uses MNK and requires `-i gemm`.

When adding a new hardware type to the harness, the dispatch is hard-coded in `read_from_json` of both sim classes — extend that `if/elif` chain and add a matching `configs/bagel/<hw>*.cfg` and `topologies/{bagel,janus}/config_<hw>*.json`.

### Outputs

- Upstream runs: `<-p>/<run_name>/{COMPUTE,BANDWIDTH,DETAILED_ACCESS}_REPORT.csv` plus per-layer trace CSVs (and `SPARSE_REPORT.csv` when sparsity is enabled).
- Harness runs: aggregated cycles + per-step log lines in the path pointed to by `result_path` in the JSON (typically `results/{bagel,janus}/...`). The constant `500_000_000` Hz is used to convert cycles to seconds in the log.

## Conventions worth knowing

- The harness assumes a clock of 500 MHz when reporting seconds; change it in `_append_to_log` if needed.
- `run_sim_once` always runs upstream with `save_disk_space=True` and the GEMM layout `layouts/GEMM_mnk/vit_l_KM_KN.csv`. If you add a new layout, plumb it through there, not at call sites.
- The diff-based regression scripts in `test/general/scripts/` rewrite cfg/source files via `sed` before running — keep that in mind if a CI or pre-commit step picks up unrelated diffs.
- Submodules: `submodules/ramulator` (and a top-level `ramulator/` entry in `.gitmodules`) are only needed for the Ramulator DRAM-coupled flow described in `README_ramulator.md`.
- Branches: active work is on `dev_zhb_order`; PR target is `main`.
