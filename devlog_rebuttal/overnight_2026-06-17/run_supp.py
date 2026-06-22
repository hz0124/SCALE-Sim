import json, os, re, subprocess, time, csv
ROOT="/share/liujun/xinhaolee/workspace/simulators/ARGUS-SCALE-Sim/SCALE-Sim"
OUT=os.path.dirname(os.path.abspath(__file__)); GEN=f"{OUT}/gen_configs"; os.makedirs(GEN,exist_ok=True)
PE=2048
# allon_pure = all_on template + quant_proj=False (5 ARGUS switches on, no proj-quant)
ALLON=[("bagel","GenEdit","topologies/bagel/config_ours.json"),
       ("bagel","GenEval","topologies/bagel/config_ours_GenEval.json"),
       ("bagel","MM","topologies/bagel/config_ours_MM.json"),
       ("janus","GenEval","topologies/janus/config_ours_janus_GenEval.json"),
       ("janus","MM","topologies/janus/config_ours_janus_MM.json")]
RUNS=[]
for model,task,tpl in ALLON:
    c=json.load(open(os.path.join(ROOT,tpl))); c["quant_proj"]=False
    name=f"{model}_ours_{task}_allonpure"; c["result_path"]=f"{GEN}/{name}.log"
    p=f"{GEN}/{name}.json"; json.dump(c,open(p,"w"),indent=2)
    RUNS.append(("ours",task,model,p,"allon_pure"))
# janus base MM (manifest had it; only GenEval config existed)
c=json.load(open(os.path.join(ROOT,"topologies/janus/config_base_janus_GenEval.json")))
c["result_path"]=f"{GEN}/janus_base_MM.log"
p=f"{GEN}/janus_base_MM.json"; json.dump(c,open(p,"w"),indent=2)
RUNS.append(("base","MM","janus",p,""))

COLS=["model","hw","task","variant","status","total_cycles","text_cycles","image_cycles","total_energy_mJ","total_mac","util_pct","wall_s"]
def grab(pat,t,cast=float,d=None):
    m=re.findall(pat,t); return cast(m[-1]) if m else d
def parse(log):
    if not os.path.exists(log): return {}
    t=open(log,errors="ignore").read(); d={}
    d["total_cycles"]=grab(r"FINAL RESULT - Total Cycles:\s*([\d.]+)",t)
    d["text_cycles"]=grab(r"Text Generation Completed, total cycles:\s*([\d.]+)",t)
    d["image_cycles"]=grab(r"Image Generation Completed, total cycles:\s*([\d.]+)",t,0.0)
    d["total_energy_mJ"]=grab(r"Total Energy:\s*([\d.]+)\s*mJ",t)
    d["total_mac"]=grab(r"Total MAC ops:\s*([\d,]+)",t,cast=lambda s:int(s.replace(",","")))
    if d.get("total_mac") and d.get("total_cycles"): d["util_pct"]=round(100*d["total_mac"]/(PE*d["total_cycles"]),2)
    return d
rows=[]
for i,(hw,task,model,cfg,var) in enumerate(RUNS):
    log=json.load(open(cfg))["result_path"]
    if os.path.exists(log): os.remove(log)
    s="run_bagel.py" if model=="bagel" else "run_janus.py"
    rs=time.time(); row={"model":model,"hw":hw,"task":task,"variant":var}
    print(f"({i+1}/{len(RUNS)}) {model}_{hw}_{task}_{var}",flush=True)
    try:
        subprocess.run(["python",s,"--hw",hw,"--task",task,"--config",cfg,"--energy"],cwd=ROOT,capture_output=True,text=True,timeout=2400)
        row.update(parse(log)); row["status"]="ok" if row.get("total_cycles") else "no_result"
    except Exception as e: row["status"]=f"err:{type(e).__name__}"
    row["wall_s"]=round(time.time()-rs,1); rows.append(row)
    print(f"   -> {row['status']} cyc={row.get('total_cycles')} util={row.get('util_pct')}",flush=True)
with open(f"{OUT}/summary_supp.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); [w.writerow({c:r.get(c) for c in COLS}) for r in rows]
print("SUPP DONE",flush=True)
