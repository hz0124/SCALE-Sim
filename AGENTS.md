# Repository Guidelines

## Project Structure & Module Organization

SCALE-Sim is a Python simulator package. Core simulator code lives in `scalesim/`, with compute models in `scalesim/compute/`, memory models in `scalesim/memory/`, and CLI entry points such as `scalesim/scale.py`. Higher-level Bagel and Janus orchestration code is in `simulation_core/`, launched by `run_bagel.py` and `run_janus.py`.

Configuration and workload assets are data files: architecture configs in `configs/`, topology CSV/JSON files in `topologies/`, layout CSV files in `layouts/`, documentation images and notes in `documentation/`, and Ramulator helpers in `scripts/` and `submodules/`. Regression fixtures and golden traces are under `test/`.

## Build, Test, and Development Commands

- `source env.sh`: load the project conda environment and exported variables before running commands from the repository root.
- `make all`: removes generated Python/build artifacts, creates `venv/`, and installs `requirements.txt`.
- `source venv/bin/activate`: activates the local environment created by `make all`.
- `pip install -e .`: installs the package in editable mode for simulator development.
- `python3 -m scalesim.scale -c configs/scale.cfg -t topologies/conv_nets/test.csv -p results/`: runs a standard SCALE-Sim workload.
- `PYTHONPATH=$PWD python3 scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/test.csv`: runs directly from source.
- `python3 run_bagel.py --hw ours --task GenEval`: runs a Bagel simulation; use `--config` for a specific JSON topology config.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Prefer clear `snake_case` for modules, functions, variables, and CLI arguments; keep existing class and API names stable when editing established simulator components. Follow the local argparse-based CLI style for new scripts. CI runs `python3 -m pylint --fail-under=7.5 scalesim/`, so keep imports clean and avoid unused code.

## Testing Guidelines

This repository uses shell-based regression checks against golden CSV traces, not a pytest suite. Run the relevant scripts before submitting simulator changes:

- `./test/general/scripts/diff_calc.sh`
- `./test/general/scripts/diff_user_ws.sh`
- `./test/general/scripts/diff_user_is.sh`
- `./test/general/scripts/diff_user_os.sh`
- `./test/sparsity/scripts/function_test.sh`

Add or update golden traces in `test/` when behavior intentionally changes, and document why outputs changed.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive commits such as `fix Issue#152 IndexError in dram_latency.py` and feature summaries for Bagel/Janus work. Keep messages specific, reference issues when applicable, and avoid committing generated files.

Before opening a PR, discuss major changes with maintainers, update `README.md`, `CHANGELOG.md`, and `documentation/` for interface changes, add tests or tutorials for new features, and request the documented two-developer sign-off.

## Configuration & Generated Files

Keep reusable inputs in `configs/`, `topologies/`, and `layouts/`. Write run outputs to `results/` or another explicit log directory, and do not commit local `venv/`, `build/`, `dist/`, cache directories, or generated traces unless they are intentional test fixtures.
