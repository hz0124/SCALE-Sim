#!/usr/bin/env python3
"""#18 breakdown generator. Parses OPBREAKDOWN lines from each rerun log and
emits two CSVs:

  breakdown.csv         per-operator cycles (model,hw,task,ablation,operator,cycles,pct)
  breakdown_4bucket.csv text/image stage × {qkvo, attn, ffn_up, ffn_down}
                        (qkvo = qkv + omap; misc = drain). 5 ops -> 4 buckets.

Covers every log present, including all_off ablations for all model×dataset.
Run after the matrix rerun finishes.
"""
import os, re, csv, glob, json

BASE = "/tmp/argus_rerun_0615"
manifest = json.load(open(f"{BASE}/manifest.json"))
manifest += json.load(open(f"{BASE}/manifest_supp.json"))   # include allon_pure
# name -> (model, hw, task, ablation)
meta = {}
for r in manifest:
    name = r[3]; abl = r[4] if len(r) > 4 else "all_on"
    meta[name] = (r[0], r[1], r[2], abl)

OP_RE = re.compile(r"OPBREAKDOWN\s+(\S+)\s+([\d.]+)\s+([\d.]+)")

def parse(path):
    ops = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = OP_RE.search(line)
        if m:
            ops[m.group(1)] = float(m.group(2))   # label -> cycles (last wins)
    return ops

per_op_rows = []
bucket_rows = []
for log in sorted(glob.glob(f"{BASE}/logs/*.log")):
    name = os.path.basename(log)[:-4]
    if name not in meta:
        continue
    model, hw, task, abl = meta[name]
    ops = parse(log)
    if not ops:
        continue
    total = sum(ops.values())
    for op, cyc in sorted(ops.items()):
        per_op_rows.append([model, hw, task, abl, op, int(cyc),
                            round(100.0*cyc/total, 3) if total else 0])
    # 4-bucket per stage
    def g(*keys):
        return sum(ops.get(k, 0.0) for k in keys)
    for stage, pfx in (("text", "text/"), ("image", "img/")):
        qkvo   = g(pfx+"qkv", pfx+"omap")
        attn   = g(pfx+"attn")
        ffn_up = g(pfx+"ffn_up")
        ffn_dn = g(pfx+"ffn_down")
        misc   = g(pfx+"drain", pfx+"misc")
        st     = qkvo+attn+ffn_up+ffn_dn+misc
        if st <= 0:
            continue
        bucket_rows.append([model, hw, task, abl, stage,
                            int(qkvo), int(attn), int(ffn_up), int(ffn_dn),
                            int(misc), int(st)])

with open(f"{BASE}/breakdown.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","hw","task","ablation","operator","cycles","pct_of_total"])
    w.writerows(per_op_rows)

with open(f"{BASE}/breakdown_4bucket.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","hw","task","ablation","stage",
                "qkvo","attn","ffn_up","ffn_down","misc","stage_total"])
    w.writerows(bucket_rows)

print(f"breakdown.csv: {len(per_op_rows)} rows")
print(f"breakdown_4bucket.csv: {len(bucket_rows)} rows")
n_logs = len({r[0] for r in [(os.path.basename(l)[:-4],) for l in glob.glob(f'{BASE}/logs/*.log')]})
print(f"logs parsed: {n_logs}")
