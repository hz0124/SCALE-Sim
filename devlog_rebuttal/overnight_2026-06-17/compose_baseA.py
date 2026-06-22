#!/usr/bin/env python3
"""§A big table: hardware as rows (HCA first), tasks as column groups, all
normalized to HCA (加速比/功率比/能效比, no × sign). Reads summary_baseA.csv."""
import csv, os
OUT=os.path.dirname(os.path.abspath(__file__)); FREQ=500e6
rows=list(csv.DictReader(open(f"{OUT}/summary_baseA.csv")))
def fl(x):
    try: return float(x)
    except: return None
# index (task_key -> hw -> metrics)
TASKS=[("bagel","GenEdit","GEdit"),("bagel","GenEval","GEval"),("bagel","MM","MM"),
       ("janus","GenEval","J-GEval"),("janus","MM","J-MM")]
HW=["HCA","FIGNA","AxCore","FlightVGM","S-DMA","ARGUS"]
M={}
for r in rows:
    c=fl(r.get("total_cycles")); e=fl(r.get("total_energy_mJ"))
    if not c or not e: continue
    P=e*1e-3/(c/FREQ)
    M[(r["model"],r["task"],r["hw"])]=dict(cyc=c,E=e,P=P,util=r.get("util_pct",""))

# header
hdr=["硬件"]
for _,_,lab in TASKS:
    hdr += [f"{lab}加速比",f"{lab}功率比",f"{lab}能效比"]
L=["## A. 基础对比（HCA 归一；加速比=cyc_HCA/cyc_hw，功率比=P_HCA/P_hw，能效比=E_HCA/E_hw=加速比×功率比）\n",
   "| "+" | ".join(hdr)+" |",
   "|"+"---|"*len(hdr)]
for hw in HW:
    cells=[("**"+hw+"**") if hw=="ARGUS" else hw]
    for model,task,lab in TASKS:
        h=M.get((model,task,"HCA")); d=M.get((model,task,hw))
        if not h or not d:
            cells += ["—","—","—"]; continue
        sp=h["cyc"]/d["cyc"]; pr=h["P"]/d["P"]; ee=h["E"]/d["E"]
        cells += [f"{sp:.2f}",f"{pr:.2f}",f"{ee:.2f}"]
    L.append("| "+" | ".join(cells)+" |")
table="\n".join(L)
print(table)
# also dump a flat csv (no × ) for pasting
with open(f"{OUT}/baseA_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr)
    for hw in HW:
        row=[hw]
        for model,task,lab in TASKS:
            h=M.get((model,task,"HCA")); d=M.get((model,task,hw))
            if not h or not d: row+=["","",""]; continue
            row+=[f"{h['cyc']/d['cyc']:.3f}",f"{h['P']/d['P']:.3f}",f"{h['E']/d['E']:.3f}"]
        w.writerow(row)
# write/replace §A in RESULTS_baseA.md (standalone so we don't clobber the full RESULTS yet)
open(f"{OUT}/RESULTS_baseA.md","w").write(table+"\n\n> DDR4 dram_idle=500mW；ARGUS w4a16=0.47 / FIGNA·disagg=0.57 / AxCore=0.50。HCA=ours all-off 异构阵列基线。base 未列。\n")
print(f"\n-> baseA_table.csv + RESULTS_baseA.md")
