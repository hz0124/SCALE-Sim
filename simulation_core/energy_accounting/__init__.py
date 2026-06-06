"""
Energy accounting for the ARGUS MLLM harness.

Adds an opt-in EnergyAccountant alongside the cycle-counting harness in
Bagel_sim / Janus_sim. Enabled with `--energy` on the CLI.

Module overview:
    coefficients.py — pJ-per-{op,byte,bit} tables, per-hw and per-DRAM-type.
    accountant.py   — EnergyAccountant: accumulates ops/bytes/bits, reports.
    extractors.py   — pulls access counts from SCALE-Sim CSVs (run_sim_once)
                       and from geometric estimates (run_sim_once_comp).
    validate.py     — optional per-sub-run accelergy CLI cross-check.
"""
