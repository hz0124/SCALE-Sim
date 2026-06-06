"""
EnergyAccountant: a lightweight running tally of MAC ops + SRAM bytes +
DRAM bits, multiplied by pJ coefficients at report time.

Intended use:

    from .coefficients import get_coefficients
    from .accountant import EnergyAccountant

    coef = get_coefficients("ours", json_overrides=cfg.get("energy_coefficients"))
    energy = EnergyAccountant(coef)

    # ... in each sub-run ...
    energy.add_mac_ops(M*N*K, precision="fp16")
    energy.add_sram_access("ifmap", words=8192, word_bytes=2)
    energy.add_dram_access("filter", words=4096, word_bytes=2)
    energy.add_cycles(cycles_this_sub_run)

    # at end of run
    print(energy.format_report())

Conventions:
    - kind ∈ {"ifmap", "filter", "ofmap"} for both add_sram_access and
      add_dram_access. We keep the buckets separate so the breakdown shows
      where the energy goes.
    - word_bytes is the precision-dependent byte width (fp16=2, int8=1, int4=0.5).
    - All energy accumulated in pJ internally. Convert to mJ / J only at report.
    - Idle DRAM background power is added once at format_report() time, derived
      from `cycles_total / freq_hz` (see coefficients["dram_idle_mw"] and
      coefficients["freq_hz"]).
"""

from collections import defaultdict


# Recognized buckets for SRAM/DRAM access. Anything else triggers an assert.
_SRAM_KINDS = ("ifmap", "filter", "ofmap")
_DRAM_KINDS = ("ifmap", "filter", "ofmap")


class EnergyAccountant:
    """
    Per-run energy + access-count accumulator.

    Multiple sub-runs (e.g. ARGUS's qkv / attn / ffn projections) call
    add_* repeatedly; only at format_report() do we compute the final
    pJ totals so coefficient changes can be made on-the-fly.
    """

    def __init__(self, coefficients):
        self.coef = coefficients

        # Raw counts. We delay the multiply-by-pJ until report time so that
        # tweaking coefficients post-hoc doesn't require re-simulation.
        self.mac_ops_per_precision = defaultdict(int)
        self.sram_words = defaultdict(int)         # key: ("ifmap","read") etc.
        self.sram_byte_words = defaultdict(int)    # bytes accumulated
        self.dram_words = defaultdict(int)
        self.dram_byte_words = defaultdict(int)
        self.cycles_total = 0

        # Per-sub-run audit trail (Phase F7). Each entry summarizes one
        # _record_energy call from Bagel_sim/Janus_sim:
        #   {"label": str, "M": int, "N": int, "K": int, "precision": str,
        #    "multiplicity": int, "scale": float, "hw_cfg_path": str,
        #    "subtotal_pJ": float}
        # Used by validate.py to pick the top-N highest-energy sub-runs
        # for accelergy CLI cross-validation.
        self.records = []

    # ------------------------------------------------------------------ ops

    def add_mac_ops(self, count, precision="fp16"):
        """Record `count` MAC operations at the given precision."""
        if count <= 0:
            return
        if precision not in self.coef["mac_pj"]:
            raise ValueError(
                f"Unknown precision {precision!r}; coefficient table has "
                f"{sorted(self.coef['mac_pj'])}"
            )
        self.mac_ops_per_precision[precision] += int(count)

    def add_sram_access(self, kind, words, word_bytes=1, op="read"):
        """
        Record SRAM accesses. `words` is the element count, `word_bytes` is
        bytes per element (fp16=2, int8=1, int4 should pass 0.5 — we accept
        floats and round at report time).
        """
        assert kind in _SRAM_KINDS, f"unknown sram kind {kind!r}"
        assert op in ("read", "write"), f"sram op must be read|write"
        if words <= 0:
            return
        self.sram_words[(kind, op)] += int(words)
        self.sram_byte_words[(kind, op)] += float(words) * float(word_bytes)

    def add_dram_access(self, kind, words, word_bytes=1, op="read"):
        """Record DRAM accesses. Conventions match add_sram_access."""
        assert kind in _DRAM_KINDS, f"unknown dram kind {kind!r}"
        assert op in ("read", "write"), f"dram op must be read|write"
        if words <= 0:
            return
        self.dram_words[(kind, op)] += int(words)
        self.dram_byte_words[(kind, op)] += float(words) * float(word_bytes)

    def add_cycles(self, n):
        """Add cycles to the running total. Used for idle-DRAM background."""
        if n > 0:
            self.cycles_total += int(n)

    def record_subrun(self, label, M, N, K, precision, multiplicity, scale,
                      hw_cfg_path, subtotal_pJ):
        """
        Append one entry to the per-sub-run audit trail. Called by
        Bagel_sim/Janus_sim._record_energy after the access counts have
        been added, with the energy that *this* call contributed.

        The audit trail is consumed by validate.py to pick the top-N
        highest-energy sub-runs for accelergy CLI cross-validation; the
        accumulated counts above are unaffected.
        """
        self.records.append({
            "label": label,
            "M": int(M), "N": int(N), "K": int(K),
            "precision": precision,
            "multiplicity": int(multiplicity),
            "scale": float(scale),
            "hw_cfg_path": hw_cfg_path,
            "subtotal_pJ": float(subtotal_pJ),
        })

    def top_records_by_energy(self, n=3):
        """Return the top-N records by subtotal_pJ, descending."""
        return sorted(self.records, key=lambda r: r["subtotal_pJ"], reverse=True)[:n]

    # -------------------------------------------------------- energy compute

    def compute_compute_pJ(self):
        """Sum MAC energy across precisions."""
        return sum(
            self.mac_ops_per_precision[p] * self.coef["mac_pj"][p]
            for p in self.mac_ops_per_precision
        )

    def compute_sram_pJ(self):
        """Per-{ifmap,filter,ofmap} SRAM energy in pJ."""
        out = {}
        rd_pj = self.coef["sram_read_pj_per_byte"]
        wr_pj = self.coef["sram_write_pj_per_byte"]
        for kind in _SRAM_KINDS:
            rd_bytes = self.sram_byte_words.get((kind, "read"), 0.0)
            wr_bytes = self.sram_byte_words.get((kind, "write"), 0.0)
            out[kind] = rd_bytes * rd_pj + wr_bytes * wr_pj
        return out

    def compute_dram_pJ(self):
        """Per-{ifmap,filter,ofmap} DRAM energy in pJ. Both rd and wr per-bit."""
        out = {}
        pj_per_bit = self.coef["dram_pj_per_bit"]
        for kind in _DRAM_KINDS:
            rd_bytes = self.dram_byte_words.get((kind, "read"), 0.0)
            wr_bytes = self.dram_byte_words.get((kind, "write"), 0.0)
            out[kind] = (rd_bytes + wr_bytes) * 8.0 * pj_per_bit
        return out

    def compute_dram_idle_pJ(self):
        """Background DRAM stack power × wall-time."""
        if self.cycles_total <= 0:
            return 0.0
        seconds = self.cycles_total / self.coef["freq_hz"]
        # mW × s = mJ. Convert to pJ: ×1e9.
        return self.coef["dram_idle_mw"] * seconds * 1e9

    # --------------------------------------------------------------- totals

    def total_compute_pJ(self):
        return self.compute_compute_pJ()

    def total_sram_pJ(self):
        return sum(self.compute_sram_pJ().values())

    def total_dram_pJ(self):
        return sum(self.compute_dram_pJ().values()) + self.compute_dram_idle_pJ()

    def total_pJ(self):
        return self.total_compute_pJ() + self.total_sram_pJ() + self.total_dram_pJ()

    def total_seconds(self):
        if self.cycles_total <= 0:
            return 0.0
        return self.cycles_total / self.coef["freq_hz"]

    def avg_power_W(self):
        secs = self.total_seconds()
        if secs <= 0:
            return 0.0
        # pJ / s = 1e-12 W
        return self.total_pJ() / secs * 1e-12

    def edp_pJs(self):
        """Energy-Delay Product in pJ·s. Common figure of merit."""
        return self.total_pJ() * self.total_seconds()

    # ------------------------------------------------------- text rendering

    def breakdown(self):
        """
        Returns a flat dict suitable for JSON/CSV serialization.
        Keys & units are stable across versions.
        """
        sram = self.compute_sram_pJ()
        dram = self.compute_dram_pJ()
        return {
            "compute_pJ": self.total_compute_pJ(),
            "sram_pJ": self.total_sram_pJ(),
            "sram_ifmap_pJ": sram["ifmap"],
            "sram_filter_pJ": sram["filter"],
            "sram_ofmap_pJ": sram["ofmap"],
            "dram_pJ": self.total_dram_pJ(),
            "dram_ifmap_pJ": dram["ifmap"],
            "dram_filter_pJ": dram["filter"],
            "dram_ofmap_pJ": dram["ofmap"],
            "dram_idle_pJ": self.compute_dram_idle_pJ(),
            "total_pJ": self.total_pJ(),
            "total_mJ": self.total_pJ() * 1e-9,
            "cycles_total": self.cycles_total,
            "seconds_total": self.total_seconds(),
            "avg_power_W": self.avg_power_W(),
            "edp_pJs": self.edp_pJs(),
            "mac_ops_per_precision": dict(self.mac_ops_per_precision),
            "dram_type": self.coef.get("dram_type"),
            "node": self.coef.get("node"),
        }

    def format_report(self):
        """Multi-line plaintext suitable for results.log."""
        b = self.breakdown()
        if b["total_pJ"] <= 0:
            return "ENERGY BREAKDOWN: no measurements recorded"

        # Pick a single readable unit for the whole report based on total_pJ.
        # The same scale applies to every line so percentages stay comparable.
        total_pj = b["total_pJ"]
        if total_pj >= 1e9:           # ≥ 1 mJ
            unit, scale, label = "mJ", 1e-9, "mJ"
        elif total_pj >= 1e6:         # ≥ 1 µJ
            unit, scale, label = "uJ", 1e-6, "µJ"
        elif total_pj >= 1e3:         # ≥ 1 nJ
            unit, scale, label = "nJ", 1e-3, "nJ"
        else:
            unit, scale, label = "pJ", 1.0, "pJ"

        def fmt(pj):
            return f"{pj * scale:>10.3f} {label}"

        def pct(x):
            return 100.0 * x / b["total_pJ"] if b["total_pJ"] else 0.0

        lines = []
        lines.append("ENERGY BREAKDOWN:")
        lines.append(
            f"  Compute (MAC):  {fmt(b['compute_pJ'])}  ({pct(b['compute_pJ']):5.1f}%)"
        )
        lines.append(
            f"  SRAM total:     {fmt(b['sram_pJ'])}  ({pct(b['sram_pJ']):5.1f}%)"
        )
        lines.append(f"    ifmap:        {fmt(b['sram_ifmap_pJ'])}")
        lines.append(f"    filter:       {fmt(b['sram_filter_pJ'])}")
        lines.append(f"    ofmap:        {fmt(b['sram_ofmap_pJ'])}")
        lines.append(
            f"  DRAM total:     {fmt(b['dram_pJ'])}  ({pct(b['dram_pJ']):5.1f}%)"
            f"  [type={b['dram_type']}]"
        )
        lines.append(f"    ifmap:        {fmt(b['dram_ifmap_pJ'])}")
        lines.append(f"    filter:       {fmt(b['dram_filter_pJ'])}")
        lines.append(f"    ofmap:        {fmt(b['dram_ofmap_pJ'])}")
        lines.append(f"    idle:         {fmt(b['dram_idle_pJ'])}")
        lines.append(f"  ----")
        lines.append(f"  Total Energy: {fmt(b['total_pJ'])}")
        lines.append(
            f"  Avg Power:    {b['avg_power_W']:>10.3f} W   (cycles {b['cycles_total']:,d})"
        )
        lines.append(
            f"  EDP:          {b['edp_pJs'] * 1e-9:>10.3f} mJ·s"
        )

        return "\n".join(lines)
