#!/usr/bin/env python3
"""Need-9 disagg split-sweep aggregator.

Parses the per-split result logs (Bagel GenEdit) for the 4-way per-stage
utilization (text/image x compute/mem, whole-chip 2048 basis), total cycles,
energy, idle leakage and EDP, then prints two markdown tables:
  Group A = full bandwidth (Bandwidth=8)
  Group B = half bandwidth  (Bandwidth=4, single off-chip pipe split in two)
plus a flat CSV. ARGUS T3-full (ours) is the reference row.

Each entry maps a config JSON -> its result_path log (read live so the script
stays in sync with the JSONs). Logs are append-mode, so every metric uses the
LAST occurrence.

Run from repo root:  python devlog_rebuttal/need9_logs/aggregate_splits.py
"""
import json, os, re, csv, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PE_TOTAL = 2048

# (label, config json relative to ROOT). vision array_width in the JSON gives
# text_lanes = PE_TOTAL - 32*array_width.
REF = ("ARGUS T3-full", "topologies/bagel/config_ours.json")
GROUP_A = [
    ("w2",  "topologies/bagel/config_disagg_w2.json"),
    ("w4",  "topologies/bagel/config_disagg.json"),
    ("w8",  "topologies/bagel/config_disagg_b.json"),
    ("w16", "topologies/bagel/config_disagg_w16.json"),
    ("w32", "topologies/bagel/config_disagg_w32.json"),
]
GROUP_B = [
    (f"{w}", f"topologies/bagel/config_disagg_{w}_bwhalf.json")
    for w in ("w2", "w4", "w8", "w16", "w32")
]


def grab(pat, text, cast=float, default=None):
    m = re.findall(pat, text)
    return cast(m[-1]) if m else default


def parse(label, cfg_rel):
    cfg = json.load(open(os.path.join(ROOT, cfg_rel)))
    log = cfg["result_path"]
    log = log if os.path.isabs(log) else os.path.join(ROOT, log.lstrip("./"))
    d = {"label": label, "log": log, "ok": os.path.exists(log)}
    # text_lanes from vision width (disagg); ARGUS = whole chip.
    vw = cfg.get("array_width")
    d["text_lanes"] = "—(2048)" if "ours" in cfg_rel else PE_TOTAL - 32 * int(vw)
    if not d["ok"]:
        return d
    t = open(log, encoding="utf-8", errors="ignore").read()
    d["total_cyc"] = grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)", t)
    d["text_cyc"]  = grab(r"Text Generation Completed, total cycles:\s*([\d.]+)", t)
    d["image_cyc"] = grab(r"Image Generation Completed, total cycles:\s*([\d.]+)", t, default=0.0)
    su = re.findall(
        r"STAGE UTIL .*?: text_compute=([\d.]+)% text_mem=([\d.]+)% "
        r"image_compute=([\d.]+)% image_mem=([\d.]+)%", t)
    if su:
        d["tc"], d["tm"], d["ic"], d["im"] = (float(x) for x in su[-1])
    d["energy_mJ"] = grab(r"Total Energy:\s*([\d.]+)\s*mJ", t)
    d["edp_mJs"]   = grab(r"EDP:\s*([\d.]+)\s*mJ", t)
    mac = grab(r"Total MAC ops:\s*([\d,]+)", t, cast=lambda s: int(s.replace(",", "")))
    if mac and d.get("total_cyc"):
        d["agg_util"] = 100.0 * mac / (PE_TOTAL * d["total_cyc"])
    return d


COLS = ["label", "text_lanes", "total_cyc", "text_cyc", "image_cyc",
        "tc", "tm", "ic", "im", "agg_util", "energy_mJ", "edp_mJs"]
HDR = ["配置", "text lanes", "总周期(e9)", "文本占比",
       "text_cmp%", "text_mem%", "img_cmp%", "img_mem%",
       "聚合util%", "能量(mJ,含静态)", "EDP"]


def fmt_row(d):
    g = lambda k, f="{:.2f}": (f.format(d[k]) if d.get(k) is not None else "—")
    tcyc, totc = d.get("text_cyc"), d.get("total_cyc")
    text_pct = f"{100.0*tcyc/totc:.1f}%" if (tcyc and totc) else "—"
    e9 = f"{totc/1e9:.1f}" if totc else "—"
    return [d["label"], str(d["text_lanes"]), e9, text_pct,
            g("tc"), g("tm"), g("ic"), g("im"), g("agg_util"),
            g("energy_mJ", "{:.0f}"), g("edp_mJs", "{:.3f}")]


def md_table(rows):
    out = ["| " + " | ".join(HDR) + " |", "|" + "---|" * len(HDR)]
    for d in rows:
        out.append("| " + " | ".join(fmt_row(d)) + " |")
    return "\n".join(out)


def main():
    ref = parse(*REF)
    a = [parse(l, c) for l, c in GROUP_A]
    b = [parse(l, c) for l, c in GROUP_B]
    print("## Group A — 满带宽 (Bandwidth=8)\n")
    print(md_table([ref] + a))
    print("\n## Group B — 半带宽 (Bandwidth=4)\n")
    print(md_table([ref] + b))
    missing = [d["label"] for d in [ref] + a + b if not d["ok"]]
    if missing:
        print(f"\n> ⚠ 未找到 log（未跑或路径不符）: {', '.join(missing)}")
    # flat CSV next to this script
    csv_path = os.path.join(os.path.dirname(__file__), "split_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group"] + COLS)
        for grp, rows in (("A_fullBW", [ref] + a), ("B_halfBW", b)):
            for d in rows:
                w.writerow([grp] + [d.get(c, "") for c in COLS])
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
