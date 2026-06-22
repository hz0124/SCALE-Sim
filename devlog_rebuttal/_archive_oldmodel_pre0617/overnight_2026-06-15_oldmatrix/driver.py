#!/usr/bin/env python3
"""Overnight ARGUS data-collection driver.

Runs every (model, hw, task[, ablation]) in manifest.json serially (they share
topologies/bagel/layer.csv so cannot parallelize), parses each result log for
cycles + energy + MAC counts, computes utilization, and writes an incremental
summary CSV/JSON so partial results survive a crash or interruption.
"""
import json, os, re, subprocess, time, csv, sys

ROOT = "/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
BASE = "/tmp/argus_rerun_0615"
PE_TOTAL = 2048   # physical lanes (all hw aligned to 2048), for utilization denominator

manifest = json.load(open(f"{BASE}/manifest.json"))
summary_csv = f"{BASE}/summary.csv"
summary_json = f"{BASE}/summary.json"

COLS = ["model","hw","task","ablation","status","total_cycles","text_cycles",
        "image_cycles","total_energy_mJ","compute_mJ","sram_mJ","dram_mJ",
        "avg_power_W","edp_mJs","total_mac","util_pct","wall_s"]

def grab(pat, text, cast=float, last=True, default=None):
    m = re.findall(pat, text)
    if not m: return default
    return cast(m[-1] if last else m[0])

def parse_log(path):
    if not os.path.exists(path): return {}
    t = open(path, encoding="utf-8", errors="ignore").read()
    d = {}
    d["total_cycles"] = grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)", t)
    txt = re.findall(r"Text Generation Completed, total cycles:\s*([\d.]+)", t)
    img = re.findall(r"Image Generation Completed, total cycles:\s*([\d.]+)", t)
    d["text_cycles"] = float(txt[-1]) if txt else None
    d["image_cycles"] = float(img[-1]) if img else 0.0
    d["compute_mJ"] = grab(r"Compute \(MAC\):\s*([\d.]+)\s*mJ", t)
    d["sram_mJ"] = grab(r"SRAM total:\s*([\d.]+)\s*mJ", t)
    d["dram_mJ"] = grab(r"DRAM total:\s*([\d.]+)\s*mJ", t)
    d["total_energy_mJ"] = grab(r"Total Energy:\s*([\d.]+)\s*mJ", t)
    d["avg_power_W"] = grab(r"Avg Power:\s*([\d.]+)\s*W", t)
    d["edp_mJs"] = grab(r"EDP:\s*([\d.]+)\s*mJ", t)
    d["total_mac"] = grab(r"Total MAC ops:\s*([\d,]+)", t, cast=lambda s:int(s.replace(",","")))
    if d.get("total_mac") and d.get("total_cycles"):
        d["util_pct"] = 100.0 * d["total_mac"] / (PE_TOTAL * d["total_cycles"])
    else:
        d["util_pct"] = None
    # Per-operator cycle breakdown: 'OPBREAKDOWN <label> <cycles> <pct>'
    d["op_breakdown"] = [(lbl, float(cyc), float(pct))
                         for lbl, cyc, pct in re.findall(
                             r"OPBREAKDOWN (\S+) ([\d.]+) ([\d.]+)", t)]
    return d

def write_summary(rows):
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in COLS})
    json.dump(rows, open(summary_json, "w"), indent=1)
    # Long-format per-operator breakdown CSV (req 3).
    with open(f"{BASE}/breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","hw","task","ablation","operator","cycles","pct_of_total"])
        for r in rows:
            for lbl, cyc, pct in r.get("op_breakdown", []) or []:
                w.writerow([r["model"],r["hw"],r["task"],r.get("ablation",""),lbl,f"{cyc:.0f}",f"{pct:.3f}"])

rows = []
t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] driver start: {len(manifest)} runs", flush=True)
for i, run in enumerate(manifest):
    model, hw, task, name = run[0], run[1], run[2], run[3]
    ablation = run[4] if len(run) > 4 else ""
    cfg = f"{BASE}/configs/{name}.json"
    log = f"{BASE}/logs/{name}.log"
    if os.path.exists(log): os.remove(log)   # fresh log (avoid append dup)
    script = "run_bagel.py" if model == "bagel" else "run_janus.py"
    cmd = ["python", script, "--hw", hw, "--task", task, "--config", cfg, "--energy"]
    rs = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ({i+1}/{len(manifest)}) {name} ...", flush=True)
    row = {"model":model,"hw":hw,"task":task,"ablation":ablation}
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=2400)
        parsed = parse_log(log)
        row.update(parsed)
        row["status"] = "ok" if parsed.get("total_cycles") else "no_result"
        if not parsed.get("total_cycles"):
            row["status"] = f"no_result(rc={p.returncode})"
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    except Exception as e:
        row["status"] = f"err:{type(e).__name__}"
    row["wall_s"] = round(time.time() - rs, 1)
    rows.append(row)
    write_summary(rows)   # incremental
    print(f"    -> {row['status']}  cyc={row.get('total_cycles')}  "
          f"E={row.get('total_energy_mJ')}mJ  util={row.get('util_pct')}  {row['wall_s']}s", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] ALL DONE in {(time.time()-t0)/60:.1f} min. "
      f"ok={sum(1 for r in rows if r['status']=='ok')}/{len(rows)}", flush=True)
