#!/usr/bin/env python3
"""Assemble RESULTS.md: §A base comparison (HCA-normalized, hw rows × task cols;
TWO tables — ratios[加速比/能效比] and power[功率/功率比]), §B ablation, §C time
breakdown. Reads summary_baseA.csv (§A), summary_full.csv+summary_supp.csv (§B),
/tmp/timetable.json (§C)."""
import csv, os, json, math
OUT=os.path.dirname(os.path.abspath(__file__)); FREQ=500e6
def geomean(vals):
    vals=[v for v in vals if v and v>0]
    return math.exp(sum(math.log(v) for v in vals)/len(vals)) if vals else None
def load(p): return list(csv.DictReader(open(f"{OUT}/{p}"))) if os.path.exists(f"{OUT}/{p}") else []
def fl(x):
    try: return float(x)
    except: return None

# ---------- §A ----------
A=load("summary_baseA.csv")
TASKS=[("bagel","GenEdit","GEdit"),("bagel","GenEval","GEval"),("bagel","MM","MM"),
       ("janus","GenEval","J-GEval"),("janus","MM","J-MM")]
HW=["HCA","SA","FIGNA","AxCore","FlightVGM","S-DMA","ARGUS"]
M={}
for r in A:
    c=fl(r.get("total_cycles")); e=fl(r.get("total_energy_mJ"))
    if c and e: M[(r["model"],r["task"],r["hw"])]=dict(cyc=c,E=e,P=e*1e-3/(c/FREQ),util=r.get("util_pct",""))
L=["# Overnight 2026-06-17 结果（DDR4 20pJ/bit, idle 500mW, per-hw w4a16, 阵列静态 150mW）\n"]
L.append("## A. 基础对比（HCA=ARGUS all-off 归一；恒等式 能效比=加速比×功率比；不带×号）\n")
# A0: SA / HCA / ARGUS 摘要（HCA 归一）
h0=["硬件"]+[f"{lab}{m}" for _,_,lab in TASKS for m in ("加速比","能效比")]+["Geomean加速比","Geomean能效比"]
L+=["### A0 摘要：SA / HCA / ARGUS（HCA 归一）\n","| "+" | ".join(h0)+" |","|"+"---|"*len(h0)]
for hw in ["SA","HCA","ARGUS"]:
    cells=[("**"+hw+"**") if hw=="ARGUS" else hw]; sps=[]; ees=[]
    for model,task,_ in TASKS:
        hb=M.get((model,task,"HCA")); d=M.get((model,task,hw))
        if hb and d:
            sp=hb['cyc']/d['cyc']; ee=hb['E']/d['E']; sps.append(sp); ees.append(ee)
            cells += [f"{sp:.2f}",f"{ee:.2f}"]
        else: cells += ["—","—"]
    gsp=geomean(sps); gee=geomean(ees)
    cells += [f"**{gsp:.2f}**" if gsp else "—", f"**{gee:.2f}**" if gee else "—"]
    L.append("| "+" | ".join(cells)+" |")
L.append("")
# A1: 加速比 + 能效比
h1=["硬件"]+[f"{lab}{m}" for _,_,lab in TASKS for m in ("加速比","能效比")]+["Geomean加速比","Geomean能效比"]
L+=["### A1 加速比 / 能效比（Geomean = 跨 5 任务几何平均）\n","| "+" | ".join(h1)+" |","|"+"---|"*len(h1)]
for hw in HW:
    cells=[("**"+hw+"**") if hw=="ARGUS" else hw]; sps=[]; ees=[]
    for model,task,_ in TASKS:
        hb=M.get((model,task,"HCA")); d=M.get((model,task,hw))
        if hb and d:
            sp=hb['cyc']/d['cyc']; ee=hb['E']/d['E']; sps.append(sp); ees.append(ee)
            cells += [f"{sp:.2f}",f"{ee:.2f}"]
        else: cells += ["—","—"]
    gsp=geomean(sps); gee=geomean(ees)
    cells += [f"**{gsp:.2f}**" if gsp else "—", f"**{gee:.2f}**" if gee else "—"]
    L.append("| "+" | ".join(cells)+" |")
# A2: 功率 + 功率比
L.append("")
h2=["硬件"]+[f"{lab}{m}" for _,_,lab in TASKS for m in ("功率W","功率比")]+["Geomean功率比"]
L+=["### A2 功率 / 功率比（功率比=P_HCA/P_hw；Geomean = 跨 5 任务几何平均）\n","| "+" | ".join(h2)+" |","|"+"---|"*len(h2)]
for hw in HW:
    cells=[("**"+hw+"**") if hw=="ARGUS" else hw]; prs=[]
    for model,task,_ in TASKS:
        hb=M.get((model,task,"HCA")); d=M.get((model,task,hw))
        if hb and d:
            pr=hb['P']/d['P']; prs.append(pr); cells += [f"{d['P']:.3f}",f"{pr:.2f}"]
        else: cells += ["—","—"]
    gpr=geomean(prs)
    cells += [f"**{gpr:.2f}**" if gpr else "—"]
    L.append("| "+" | ".join(cells)+" |")
L.append("")

# ---------- §B ablation ----------
full=load("summary_full.csv"); supp=load("summary_supp.csv")
Ab={(r["model"],r["task"],r["variant"]):r for r in full if r["hw"]=="ours"}
for r in supp:
    if r.get("variant")=="allon_pure": Ab[(r["model"],r["task"],"allon_pure")]=r
# all_on from summary_baseA (ARGUS row)
L.append("## B. 技术点 ablation（ours，相对 all_off=HCA）\n")
for model in ["bagel","janus"]:
    for task in (["GenEdit","GenEval","MM"] if model=="bagel" else ["GenEval","MM"]):
        allon=M.get((model,task,"ARGUS"))
        if not allon: continue
        off=Ab.get((model,task,"alloff")); offc=fl(off["total_cycles"]) if off else None
        offe=fl(off["total_energy_mJ"]) if off else None   # all_off=HCA energy → 能效比基准
        def cell(label, cyc, util, ene):
            sp=f"{offc/cyc:.2f}" if (offc and cyc) else "—"
            ee=f"{offe/ene:.2f}" if (offe and ene) else "—"
            c9=f"{cyc/1e9:.1f}" if cyc else "—"
            return f"| {label} | {c9} | {util} | {sp} | {ee} |"
        L+=[f"### {model} {task}\n","| 变体 | 总周期(e9) | util% | 加速比(vs all_off) | 能效比(vs all_off) |","|---|---|---|---|---|"]
        L.append(cell("**all_on**", allon['cyc'], allon['util'], allon['E']))
        for key in ["alloff","onlyT1","onlyT2","onlyT3","noT3","allon_pure"]:
            r=Ab.get((model,task,key))
            if not r or not fl(r.get("total_cycles")): L.append(f"| {key} | — | — | — | — |"); continue
            L.append(cell(key, fl(r["total_cycles"]), r.get("util_pct",""), fl(r.get("total_energy_mJ"))))
        L.append("")

# ---------- §C time breakdown ----------
if os.path.exists("/tmp/timetable.json"):
    d=json.load(open("/tmp/timetable.json"))
    L.append("## C. 阶段×算子 时间分解（秒，HCA 归一）\n")
    for task,title in [("GenEdit","GenEdit"),("MM","MM")]:
        if f"{task}|HCA" not in d: continue
        base=d[f"{task}|HCA"]["total"]
        L+=[f"### bagel/{title}\n","| | AR-Attention | AR-FFN | Diff-Attention | Diff-FFN | Other | 归一化时间(HCA) | 总时间(s) |","|---|---|---|---|---|---|---|---|"]
        for disp in ["HCA","FIGNA","S-DMA","ARGUS"]:
            if f"{task}|{disp}" not in d: continue
            r=d[f"{task}|{disp}"]
            L.append(f"| {disp} | {r['ar_attn']:.2f} | {r['ar_ffn']:.2f} | {r['diff_attn']:.2f} | {r['diff_ffn']:.2f} | {r['other']:.2f} | {r['total']/base:.2f} | {r['total']:.2f} |")
        L.append("")

L+=["## 注记",
 "- 系数：DRAM 读写 DDR4 20pJ/bit、DRAM 背景 idle 500mW、阵列静态 150mW（按运行时间计，罚慢设计）；w4a16 按 hw：ARGUS 0.47 / FIGNA·disagg 0.57 / AxCore 0.50。",
 "- 能效比=加速比×功率比（恒等式 E=P·T）。ARGUS 能效优势主要来自**运行时间短**（静态+背景+访问少），compute 与 FIGNA 接近。",
 "- AxCore=FIGNA 同款 W4A16，仅 w4a16 系数 0.50<0.57 → 周期/util 与 FIGNA 完全相同、能量略低。HCA=ARGUS all-off 异构阵列基线。base 不列。",
 "- §A 含静态/DDR4 最新系数；§B/§C 同口径重刷。"]
open(f"{OUT}/RESULTS.md","w").write("\n".join(L)+"\n")
print("wrote RESULTS.md", len(L), "lines")
