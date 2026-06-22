#!/usr/bin/env python3
"""A. 基础对比 re-run (new text model, 2026-06-17).

ARGUS (ours) vs the 4 DSA baselines (figna/flightvgm/sdma) across Bagel
{GenEdit,GenEval,MM} and Janus {GenEval,MM}. Uses the existing topologies/
configs directly (no /tmp staging). Serial (shared layer.csv). Writes an
incremental summary CSV/JSON so partial results survive interruption.

NOT included (flag): bagel `base` hw has no config in topologies/ (need-1 SA,
reconstruct separately); ablation ladder (§B) is a later run.
"""
import json, os, re, subprocess, time, csv

ROOT = "/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
OUT = os.path.dirname(os.path.abspath(__file__))
PE_TOTAL = 2048

# (model, hw, task, config relative to ROOT)
RUNS = [
    # ---- Bagel ----
    ("bagel", "ours",      "GenEdit", "topologies/bagel/config_ours.json"),
    ("bagel", "ours",      "GenEval", "topologies/bagel/config_ours_GenEval.json"),
    ("bagel", "ours",      "MM",      "topologies/bagel/config_ours_MM.json"),
    ("bagel", "figna",     "GenEdit", "topologies/bagel/config_figna.json"),
    ("bagel", "figna",     "GenEval", "topologies/bagel/config_figna_GenEval.json"),
    ("bagel", "figna",     "MM",      "topologies/bagel/config_figna_MM.json"),
    ("bagel", "flightvgm", "GenEdit", "topologies/bagel/config_flightvgm.json"),
    ("bagel", "flightvgm", "GenEval", "topologies/bagel/config_flightvgm_GenEval.json"),
    ("bagel", "flightvgm", "MM",      "topologies/bagel/config_flightvgm_MM.json"),
    ("bagel", "sdma",      "GenEdit", "topologies/bagel/config_sdma.json"),
    ("bagel", "sdma",      "GenEval", "topologies/bagel/config_sdma_GenEval.json"),
    ("bagel", "sdma",      "MM",      "topologies/bagel/config_sdma_MM.json"),
    # ---- Janus ----
    ("janus", "ours",      "GenEval", "topologies/janus/config_ours_janus_GenEval.json"),
    ("janus", "ours",      "MM",      "topologies/janus/config_ours_janus_MM.json"),
    ("janus", "figna",     "GenEval", "topologies/janus/config_figna_janus_GenEval.json"),
    ("janus", "figna",     "MM",      "topologies/janus/config_figna_janus_MM.json"),
    ("janus", "flightvgm", "GenEval", "topologies/janus/config_flightvgm_janus_GenEval.json"),
    ("janus", "flightvgm", "MM",      "topologies/janus/config_flightvgm_janus_MM.json"),
    ("janus", "sdma",      "GenEval", "topologies/janus/config_sdma_janus_GenEval.json"),
    ("janus", "sdma",      "MM",      "topologies/janus/config_sdma_janus_MM.json"),
    ("janus", "base",      "GenEval", "topologies/janus/config_base_janus_GenEval.json"),
]

COLS = ["model","hw","task","status","total_cycles","text_cycles","image_cycles",
        "total_energy_mJ","compute_mJ","sram_mJ","dram_mJ","avg_power_W","edp_mJs",
        "total_mac","util_pct","text_cmp","text_mem","img_cmp","img_mem","wall_s"]

def grab(pat, t, cast=float, default=None):
    m = re.findall(pat, t)
    return cast(m[-1]) if m else default

def parse_log(path):
    if not os.path.exists(path): return {}
    t = open(path, encoding="utf-8", errors="ignore").read()
    d = {}
    d["total_cycles"] = grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)", t)
    d["text_cycles"]  = grab(r"Text Generation Completed, total cycles:\s*([\d.]+)", t)
    d["image_cycles"] = grab(r"Image Generation Completed, total cycles:\s*([\d.]+)", t, default=0.0)
    d["compute_mJ"]   = grab(r"Compute \(MAC\):\s*([\d.]+)\s*mJ", t)
    d["sram_mJ"]      = grab(r"SRAM total:\s*([\d.]+)\s*mJ", t)
    d["dram_mJ"]      = grab(r"DRAM total:\s*([\d.]+)\s*mJ", t)
    d["total_energy_mJ"] = grab(r"Total Energy:\s*([\d.]+)\s*mJ", t)
    d["avg_power_W"]  = grab(r"Avg Power:\s*([\d.]+)\s*W", t)
    d["edp_mJs"]      = grab(r"EDP:\s*([\d.]+)\s*mJ", t)
    d["total_mac"]    = grab(r"Total MAC ops:\s*([\d,]+)", t, cast=lambda s:int(s.replace(",","")))
    su = re.findall(r"STAGE UTIL .*?: text_compute=([\d.]+)% text_mem=([\d.]+)% "
                    r"image_compute=([\d.]+)% image_mem=([\d.]+)%", t)
    if su:
        d["text_cmp"],d["text_mem"],d["img_cmp"],d["img_mem"] = (float(x) for x in su[-1])
    if d.get("total_mac") and d.get("total_cycles"):
        d["util_pct"] = round(100.0*d["total_mac"]/(PE_TOTAL*d["total_cycles"]),2)
    return d

def write(rows):
    with open(f"{OUT}/summary_base.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c:r.get(c) for c in COLS})
    json.dump(rows, open(f"{OUT}/summary_base.json","w"), indent=1)

rows=[]; t0=time.time()
print(f"[{time.strftime('%H:%M:%S')}] base-comparison: {len(RUNS)} runs", flush=True)
for i,(model,hw,task,cfgrel) in enumerate(RUNS):
    cfg=os.path.join(ROOT,cfgrel)
    log=json.load(open(cfg))["result_path"]
    log=log if os.path.isabs(log) else os.path.join(ROOT,log.lstrip("./"))
    if os.path.exists(log): os.remove(log)   # fresh (logs append otherwise)
    script="run_bagel.py" if model=="bagel" else "run_janus.py"
    cmd=["python",script,"--hw",hw,"--task",task,"--config",cfgrel,"--energy"]
    rs=time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ({i+1}/{len(RUNS)}) {model}_{hw}_{task} ...", flush=True)
    row={"model":model,"hw":hw,"task":task}
    try:
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=2400)
        row.update(parse_log(log))
        row["status"]="ok" if row.get("total_cycles") else f"no_result(rc={p.returncode})"
    except subprocess.TimeoutExpired:
        row["status"]="timeout"
    except Exception as e:
        row["status"]=f"err:{type(e).__name__}"
    row["wall_s"]=round(time.time()-rs,1)
    rows.append(row); write(rows)
    print(f"    -> {row['status']} cyc={row.get('total_cycles')} util={row.get('util_pct')} {row['wall_s']}s", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] DONE {(time.time()-t0)/60:.1f}min ok={sum(1 for r in rows if r['status']=='ok')}/{len(rows)}", flush=True)
