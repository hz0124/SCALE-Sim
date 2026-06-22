#!/usr/bin/env python3
"""§A base comparison (HCA-normalized): HCA, FIGNA, AxCore, FlightVGM, S-DMA, ARGUS
across Bagel {GenEdit,GenEval,MM} + Janus {GenEval,MM}. 30 runs, serial.
HCA = ours all-off (heterogeneous array baseline). AxCore = figna scheme w/ w4a16=0.5.
Writes summary_baseA.csv. (base hw excluded per request.)"""
import json, os, re, subprocess, time, csv
ROOT="/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
OUT=os.path.dirname(os.path.abspath(__file__)); GEN=f"{OUT}/genA"; os.makedirs(GEN,exist_ok=True)
PE=2048; FREQ=500e6
F=False
ALLOFF=dict(t1_soft=F,t1_hard=F,t2_soft=F,t2_hard=F,t3_soft=F,quant_proj=F)
# (model, task, script, ours_allon_cfg, figna_cfg, flightvgm_cfg, sdma_cfg)
TASKS=[
 ("bagel","GenEdit","run_bagel.py","config_ours.json","config_figna.json","config_flightvgm.json","config_sdma.json"),
 ("bagel","GenEval","run_bagel.py","config_ours_GenEval.json","config_figna_GenEval.json","config_flightvgm_GenEval.json","config_sdma_GenEval.json"),
 ("bagel","MM","run_bagel.py","config_ours_MM.json","config_figna_MM.json","config_flightvgm_MM.json","config_sdma_MM.json"),
 ("janus","GenEval","run_janus.py","config_ours_janus_GenEval.json","config_figna_janus_GenEval.json","config_flightvgm_janus_GenEval.json","config_sdma_janus_GenEval.json"),
 ("janus","MM","run_janus.py","config_ours_janus_MM.json","config_figna_janus_MM.json","config_flightvgm_janus_MM.json","config_sdma_janus_MM.json"),
]
TOPO=lambda model,c: f"topologies/{model}/{c}"

def gen_cfg(srcrel, updates, tag):
    c=json.load(open(os.path.join(ROOT,srcrel))); c.update(updates)
    c["result_path"]=f"{GEN}/{tag}.log"
    p=f"{GEN}/{tag}.json"; json.dump(c,open(p,"w"),indent=2); return p

def grab(pat,t,cast=float,d=None):
    m=re.findall(pat,t); return cast(m[-1]) if m else d
def parse(log):
    if not os.path.exists(log): return {}
    t=open(log,errors="ignore").read(); d={}
    d["total_cycles"]=grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)",t)
    d["total_energy_mJ"]=grab(r"Total Energy:\s*([\d.]+)\s*mJ",t)
    d["total_mac"]=grab(r"Total MAC ops:\s*([\d,]+)",t,cast=lambda s:int(s.replace(",","")))
    if d.get("total_mac") and d.get("total_cycles"): d["util_pct"]=round(100*d["total_mac"]/(PE*d["total_cycles"]),2)
    return d

rows=[]; t0=time.time()
for model,task,script,ours_c,figna_c,fvgm_c,sdma_c in TASKS:
    # build the 6 hw run specs: (hw_label, --hw, config_path_relative_to_ROOT_or_abs)
    hca_cfg = gen_cfg(TOPO(model,ours_c), ALLOFF, f"HCA_{model}_{task}")
    ax_cfg  = gen_cfg(TOPO(model,figna_c), {}, f"AxCore_{model}_{task}")
    # SA (=base, plain dense fp16 systolic array): flightvgm template + base.cfg
    sa_cfg  = gen_cfg(TOPO(model,fvgm_c), {"config":"./configs/bagel/base.cfg","config_comp":"./configs/bagel/base.cfg"}, f"SA_{model}_{task}")
    specs=[
        ("HCA","ours",hca_cfg),
        ("SA","base",sa_cfg),
        ("FIGNA","figna",os.path.join(ROOT,TOPO(model,figna_c))),
        ("AxCore","axcore",ax_cfg),
        ("FlightVGM","flightvgm",os.path.join(ROOT,TOPO(model,fvgm_c))),
        ("S-DMA","sdma",os.path.join(ROOT,TOPO(model,sdma_c))),
        ("ARGUS","ours",os.path.join(ROOT,TOPO(model,ours_c))),
    ]
    for label,hw,cfg in specs:
        log=json.load(open(cfg))["result_path"]; log=log if os.path.isabs(log) else os.path.join(ROOT,log.lstrip("./"))
        if os.path.exists(log): os.remove(log)
        rs=time.time(); row={"model":model,"task":task,"hw":label}
        try:
            subprocess.run(["python",script,"--hw",hw,"--task",task,"--config",cfg,"--energy"],cwd=ROOT,capture_output=True,text=True,timeout=2400)
            row.update(parse(log)); row["status"]="ok" if row.get("total_cycles") else "no_result"
        except Exception as e: row["status"]=f"err:{type(e).__name__}"
        row["wall_s"]=round(time.time()-rs,1); rows.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {model}/{task} {label}: {row['status']} cyc={row.get('total_cycles')} util={row.get('util_pct')} {row['wall_s']}s",flush=True)
        # incremental write
        COLS=["model","task","hw","status","total_cycles","total_energy_mJ","total_mac","util_pct","wall_s"]
        with open(f"{OUT}/summary_baseA.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); [w.writerow({c:r.get(c) for c in COLS}) for r in rows]
print(f"[{time.strftime('%H:%M:%S')}] DONE {(time.time()-t0)/60:.1f}min ok={sum(1 for r in rows if r['status']=='ok')}/{len(rows)}",flush=True)
