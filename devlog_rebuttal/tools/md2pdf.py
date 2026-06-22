#!/usr/bin/env python
"""Render REBUTTAL_FRAMEWORK.md to a markdown-preview-style PDF.

Usage:  python md2pdf.py [input.md] [output.pdf]
Needs:  pandoc + PyMuPDF (both present in the zhb-scalesim-v3 env).
"""
import fitz, re, subprocess, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "REBUTTAL_FRAMEWORK.md"
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"

body = subprocess.run(["pandoc", src, "-f", "markdown", "-t", "html5"],
                      capture_output=True, text=True, check=True).stdout

# Replace the ASCII three-layer stack diagram with styled stacked boxes
layers = [
    ("MLLM Decode Harness（本仓库新增, simulation_core/）",
     "按“模型结构 × 任务 × 硬件”组合出 decode 全过程的算子序列，对每个代表性算子调用下层仿真一次，再解析放大成端到端周期/能量"),
    ("SCALE-Sim v3（上游 cycle-accurate 脉动阵列仿真器）",
     "对单个 GEMM 做逐周期仿真：阵列折叠、双缓冲 SRAM、DRAM 带宽"),
    ("能量账本 EnergyAccountant（本仓库新增, Phase F）",
     "MAC/SRAM/DRAM 访问次数 × pJ 系数（RTL/Accelergy 系数可换）"),
]
diagram = "".join(
    f'<div style="border:1.5px solid #57606a; background-color:#f6f8fa; '
    f'padding:6px 10px; margin:0 30px 4px 30px;"><b>{t}</b><br/>{d}</div>'
    for t, d in layers)
body = re.sub(r'<pre[^>]*>(?:(?!</pre>).)*MLLM Decode Harness(?:(?!</pre>).)*</pre>',
              diagram, body, count=1, flags=re.S)

# Zero-width break opportunities in inline code so long identifiers wrap in cells
ZW = "​"
def breakable(m):
    inner = m.group(1)
    if "\n" in inner:
        return m.group(0)
    return "<code>" + re.sub(r"([_=/.])", r"\1" + ZW, inner) + "</code>"
body = re.sub(r"<code>((?:(?!</code>).)*)</code>", breakable, body, flags=re.S)

css = """
body { font-family: sans-serif; font-size: 10.5px; line-height: 1.55; color: #1f2328; }
h1 { font-size: 19px; }
h2 { font-size: 15px; color: #1a4f8b; margin-top: 16px; }
h3 { font-size: 12.5px; color: #1a4f8b; }
code { font-family: monospace; font-size: 9.5px; background-color: #eff1f3; }
pre { background-color: #f6f8fa; border: 1px solid #d8dde3; padding: 8px; font-size: 9px; }
pre code { background-color: #f6f8fa; }
table { border-collapse: collapse; font-size: 9px; margin: 6px 0; width: 100%; }
th, td { border: 1px solid #c8cdd3; padding: 3px 5px; vertical-align: top; }
th { background-color: #eef1f4; }
blockquote { border-left: 3px solid #c8cdd3; padding-left: 10px; margin-left: 4px; color: #57606a; }
li { margin: 2px 0; }
hr { border: 0.5px solid #d8dde3; }
"""

story = fitz.Story(html=f"<html><body>{body}</body></html>", user_css=css)
writer = fitz.DocumentWriter(dst + ".tmp")
page = fitz.paper_rect("a4")
where = page + (46, 42, -46, -52)
more = 1
while more:
    dev = writer.begin_page(page)
    more, _ = story.place(where)
    story.draw(dev)
    writer.end_page()
writer.close()

doc = fitz.open(dst + ".tmp")
for i, p in enumerate(doc):
    p.insert_text((46, page.height - 24),
                  "ARGUS Rebuttal Framework v3 · 2026-06-11",
                  fontsize=7.5, color=(0.45, 0.45, 0.45))
    p.insert_text((page.width - 86, page.height - 24),
                  f"{i+1} / {len(doc)}", fontsize=7.5, color=(0.45, 0.45, 0.45))
doc.save(dst, deflate=True)
os.remove(dst + ".tmp")
print(f"{dst}: {len(doc)} pages")
