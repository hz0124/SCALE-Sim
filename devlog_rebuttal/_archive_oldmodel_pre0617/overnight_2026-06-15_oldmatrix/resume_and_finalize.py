#!/usr/bin/env python3
"""Resume the killed rerun: run only the incomplete runs (manifest_resume.json),
then rebuild summary.csv from ALL logs (main 50 + supp 5) and regenerate the
breakdown CSVs. Idempotent."""
import json, os, re, subprocess, time, csv

ROOT = "/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
BASE = "/tmp/argus_rerun_0615"
PE_TOTAL = 2048

resume = json.load(open(f"{BASE}/manifest_resume.json"))
print(f"[{time.strftime('%H:%M:%S')}] resume: {len(resume)} runs", flush=True)
for i, run in enumerate(resume):
    model, hw, task, name = run[0], run[1], run[2], run[3]
    cfg = f"{BASE}/configs/{name}.json"
    log = f"{BASE}/logs/{name}.log"
    if os.path.exists(log):
        os.remove(log)
    script = "run_bagel.py" if model == "bagel" else "run_janus.py"
    cmd = ["python", script, "--hw", hw, "--task", task, "--config", cfg, "--energy"]
    rs = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ({i+1}/{len(resume)}) {name} ...", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=2400)
    ok = os.path.exists(log) and "FINAL RESULT - Total Cycles" in open(log, errors="ignore").read()
    print(f"    -> {'ok' if ok else 'FAIL rc=%d'%p.returncode}  {time.time()-rs:.1f}s", flush=True)

# ---- rebuild summary.csv from ALL logs (main + supp) ----
main = json.load(open(f"{BASE}/manifest.json"))
supp = json.load(open(f"{BASE}/manifest_supp.json"))
allruns = main + supp

def grab(pat, t, cast=float):
    m = re.findall(pat, t)
    return cast(m[-1]) if m else None

COLS = ["model","hw","task","ablation","status","total_cycles","text_cycles",
        "image_cycles","total_energy_mJ","compute_mJ","sram_mJ","dram_mJ",
        "avg_power_W","edp_mJs","total_mac","util_pct"]
rows = []
for run in allruns:
    model, hw, task, name = run[0], run[1], run[2], run[3]
    abl = run[4] if len(run) > 4 else "all_on"
    log = f"{BASE}/logs/{name}.log"
    r = {"model":model,"hw":hw,"task":task,"ablation":abl,"status":"missing"}
    if os.path.exists(log):
        t = open(log, errors="ignore").read()
        r["total_cycles"] = grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)", t)
        txt = re.findall(r"Text Generation Completed, total cycles:\s*([\d.]+)", t)
        img = re.findall(r"Image Generation Completed, total cycles:\s*([\d.]+)", t)
        r["text_cycles"] = float(txt[-1]) if txt else None
        r["image_cycles"] = float(img[-1]) if img else 0.0
        r["compute_mJ"] = grab(r"Compute \(MAC\):\s*([\d.]+)\s*mJ", t)
        r["sram_mJ"] = grab(r"SRAM total:\s*([\d.]+)\s*mJ", t)
        r["dram_mJ"] = grab(r"DRAM total:\s*([\d.]+)\s*mJ", t)
        r["total_energy_mJ"] = grab(r"Total Energy:\s*([\d.]+)\s*mJ", t)
        r["avg_power_W"] = grab(r"Avg Power:\s*([\d.]+)\s*W", t)
        r["edp_mJs"] = grab(r"EDP:\s*([\d.]+)\s*mJ", t)
        r["total_mac"] = grab(r"Total MAC ops:\s*([\d,]+)", t, cast=lambda s:int(s.replace(",","")))
        if r.get("total_cycles"):
            r["status"] = "ok"
            if r.get("total_mac"):
                r["util_pct"] = 100.0 * r["total_mac"] / (PE_TOTAL * r["total_cycles"])
    rows.append(r)

with open(f"{BASE}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in rows: w.writerow({c: r.get(c) for c in COLS})
json.dump(rows, open(f"{BASE}/summary.json","w"), indent=1)
ok = sum(1 for r in rows if r["status"]=="ok")
print(f"[{time.strftime('%H:%M:%S')}] summary rebuilt: {ok}/{len(rows)} ok", flush=True)

# ---- breakdown ----
subprocess.run(["python", f"{BASE}/make_breakdown.py"], cwd=ROOT)
print(f"[{time.strftime('%H:%M:%S')}] FINALIZE DONE", flush=True)
