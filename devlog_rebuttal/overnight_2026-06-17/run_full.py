#!/usr/bin/env python3
"""B. ablation 阶梯 + bagel base hw re-run (new text model, 2026-06-17).

Generates the per-variant ablation configs (onlyT1/onlyT2/onlyT3/noT3 from the
all_on template + recipe; alloff uses the existing config) and the bagel `base`
hw configs (from the flightvgm template, swapping in base.cfg), then runs them
serially with --energy. all_on rows are already in summary_base.csv.
Recipe per §4.5 switch table + need6 quant_proj convention (isolation off /
leave-one-out on). Writes summary_full.csv.
"""
import json, os, re, subprocess, time, csv, copy

ROOT = "/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
OUT = os.path.dirname(os.path.abspath(__file__))
GEN = f"{OUT}/gen_configs"; os.makedirs(GEN, exist_ok=True)
PE_TOTAL = 2048
T, F = True, False
RECIPE = {
    "alloff": dict(t1_soft=F,t1_hard=F,t2_soft=F,t2_hard=F,t3_soft=F,quant_proj=F),
    "onlyT1": dict(t1_soft=T,t1_hard=T,t2_soft=F,t2_hard=F,t3_soft=F,quant_proj=F),
    "onlyT2": dict(t1_soft=F,t1_hard=F,t2_soft=T,t2_hard=T,t3_soft=F,quant_proj=F),
    "onlyT3": dict(t1_soft=F,t1_hard=F,t2_soft=F,t2_hard=F,t3_soft=T,quant_proj=F),
    "noT3":   dict(t1_soft=T,t1_hard=T,t2_soft=T,t2_hard=T,t3_soft=F,quant_proj=T),
}
# (model, task, all_on template config relative to ROOT)
OURS_TPL = [
    ("bagel","GenEdit","topologies/bagel/config_ours.json"),
    ("bagel","GenEval","topologies/bagel/config_ours_GenEval.json"),
    ("bagel","MM",     "topologies/bagel/config_ours_MM.json"),
    ("janus","GenEval","topologies/janus/config_ours_janus_GenEval.json"),
    ("janus","MM",     "topologies/janus/config_ours_janus_MM.json"),
]
# bagel base hw: from flightvgm template (single `config` key) -> base.cfg
BASE_TPL = [
    ("GenEdit","topologies/bagel/config_flightvgm.json"),
    ("GenEval","topologies/bagel/config_flightvgm_GenEval.json"),
    ("MM",     "topologies/bagel/config_flightvgm_MM.json"),
]

def gen_ablation(model, task, tplrel, variant):
    cfg = json.load(open(os.path.join(ROOT, tplrel)))
    cfg.update(RECIPE[variant])
    name = f"{model}_ours_{task}_{variant}"
    cfg["result_path"] = f"{GEN}/{name}.log"
    p = f"{GEN}/{name}.json"; json.dump(cfg, open(p,"w"), indent=2)
    return ("ours", task, model, p, variant)

def gen_base(task, tplrel):
    cfg = json.load(open(os.path.join(ROOT, tplrel)))
    for k in ("config","config_comp","config_comp0","config_comp1","config_comm0","config_comm1"):
        if k in cfg: cfg[k] = "./configs/bagel/base.cfg"
    name = f"bagel_base_{task}"
    cfg["result_path"] = f"{GEN}/{name}.log"
    p = f"{GEN}/{name}.json"; json.dump(cfg, open(p,"w"), indent=2)
    return ("base", task, "bagel", p, "")

# build run list
RUNS = []
for model,task,tpl in OURS_TPL:
    for v in ("alloff","onlyT1","onlyT2","onlyT3","noT3"):
        if v == "alloff" and model=="bagel" and task in ("GenEdit","MM"):
            # use the committed alloff configs (identical recipe) where they exist
            existing = {"GenEdit":"topologies/bagel/config_ours_alloff.json",
                        "MM":"topologies/bagel/config_ours_MM_alloff.json"}[task]
            RUNS.append(("ours",task,model,os.path.join(ROOT,existing),"alloff"))
        else:
            RUNS.append(gen_ablation(model,task,tpl,v))
for task,tpl in BASE_TPL:
    RUNS.append(gen_base(task,tpl))

COLS = ["model","hw","task","variant","status","total_cycles","text_cycles","image_cycles",
        "total_energy_mJ","total_mac","util_pct","text_cmp","text_mem","img_cmp","img_mem","wall_s"]

def grab(pat,t,cast=float,default=None):
    m=re.findall(pat,t); return cast(m[-1]) if m else default
def parse_log(path):
    if not os.path.exists(path): return {}
    t=open(path,encoding="utf-8",errors="ignore").read(); d={}
    d["total_cycles"]=grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)",t)
    d["text_cycles"]=grab(r"Text Generation Completed, total cycles:\s*([\d.]+)",t)
    d["image_cycles"]=grab(r"Image Generation Completed, total cycles:\s*([\d.]+)",t,default=0.0)
    d["total_energy_mJ"]=grab(r"Total Energy:\s*([\d.]+)\s*mJ",t)
    d["total_mac"]=grab(r"Total MAC ops:\s*([\d,]+)",t,cast=lambda s:int(s.replace(",","")))
    su=re.findall(r"STAGE UTIL .*?: text_compute=([\d.]+)% text_mem=([\d.]+)% image_compute=([\d.]+)% image_mem=([\d.]+)%",t)
    if su: d["text_cmp"],d["text_mem"],d["img_cmp"],d["img_mem"]=(float(x) for x in su[-1])
    if d.get("total_mac") and d.get("total_cycles"):
        d["util_pct"]=round(100.0*d["total_mac"]/(PE_TOTAL*d["total_cycles"]),2)
    return d
def write(rows):
    with open(f"{OUT}/summary_full.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({c:r.get(c) for c in COLS})
    json.dump(rows,open(f"{OUT}/summary_full.json","w"),indent=1)

rows=[]; t0=time.time()
print(f"[{time.strftime('%H:%M:%S')}] ablation+base: {len(RUNS)} runs", flush=True)
for i,(hw,task,model,cfg,variant) in enumerate(RUNS):
    log=json.load(open(cfg))["result_path"]
    log=log if os.path.isabs(log) else os.path.join(ROOT,log.lstrip("./"))
    if os.path.exists(log): os.remove(log)
    script="run_bagel.py" if model=="bagel" else "run_janus.py"
    cmd=["python",script,"--hw",hw,"--task",task,"--config",cfg,"--energy"]
    rs=time.time(); row={"model":model,"hw":hw,"task":task,"variant":variant}
    print(f"[{time.strftime('%H:%M:%S')}] ({i+1}/{len(RUNS)}) {model}_{hw}_{task}_{variant} ...", flush=True)
    try:
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=2400)
        row.update(parse_log(log))
        row["status"]="ok" if row.get("total_cycles") else f"no_result(rc={p.returncode})"
    except Exception as e:
        row["status"]=f"err:{type(e).__name__}"
    row["wall_s"]=round(time.time()-rs,1); rows.append(row); write(rows)
    print(f"    -> {row['status']} cyc={row.get('total_cycles')} util={row.get('util_pct')} {row['wall_s']}s", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] DONE {(time.time()-t0)/60:.1f}min ok={sum(1 for r in rows if r['status']=='ok')}/{len(rows)}", flush=True)
