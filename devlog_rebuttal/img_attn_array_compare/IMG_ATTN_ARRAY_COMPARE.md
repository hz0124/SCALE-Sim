# 图像阶段 attention 对比：ARGUS vs 32×64 FP16/FP8/FP4 阵列（Bagel GenEdit）

> 由 `compare_img_attn.py` 生成（in-process 跑 Bagel_sim，不污染 ./results）。
> 只看图像(diffusion)阶段的 attention 算子（img/attn = attn_qk + attn_sfmxv）。
> **计算时间** = attn 周期 / 500MHz。**能耗 = 总动态能耗 = MAC + SRAM + DRAM**（算子可归因的动态能量，
> **不含** 全芯片 idle 背景与阵列静态——那是运行时间项、非单算子可分摊）。MAC 列单列出来便于看构成。

## 口径
- **ARGUS** = `config_ours.json`：attention 走 `array_width_half=32`（异构阵列另一半是 FP-INT4，fp16 attention 只能用 32 宽），
  fp16，含 T1 区域稀疏 + T3 ÷2 负载均衡 → 有效 KV 大幅缩减。
- **基线（FP16/FP8/FP4 32×64）** = `config_ours_alloff.json` + `array_width_half=64`：统一阵列用满 64 宽跑 attention、
  **全稠密 KV**（无 T1 稀疏）。三个基线几何/工作负载完全相同（同周期、同 MAC、同访问计数），**只差精度**。
- per-MAC 能量(28nm 占位)：fp16=1.5 / fp8=0.4 / fp4=0.2 pJ。
- **SRAM/DRAM 能量随操作数字节线性缩放**：fp16=2B / fp8=1B / fp4=0.5B → 基线访存能耗按 1 / ½ / ¼ 缩放
  （harness 实测在 fp16 口径，FP8/FP4 据字节比缩放）。

## 结果

| 配置 | 阵列(attn宽) | 精度(pJ) | 有效KV(full/woimg/wotext) | 时间@500MHz(s) | MAC能耗(mJ) | SRAM+DRAM能耗(mJ) | 总动态能耗(mJ) |
|---|---|---|---|---|---|---|---|
| ARGUS | 32×64 (attn 32 半宽) | fp16 (1.5) | 1354/744/1307 | 120.34 | 70,629 | 144,423 | 215,052 |
| FP16FP16 | 32×64 (attn 64 满宽) | fp16 (1.5) | 4861/1668/4619 | 213.64 | 231,240 | 442,520 | 673,760 |
| FP8FP8 | 32×64 (attn 64 满宽) | fp8 (0.4) | 4861/1668/4619 | 213.64 | 61,664 | 221,260 | 282,924 |
| FP4FP4 | 32×64 (attn 64 满宽) | fp4 (0.2) | 4861/1668/4619 | 213.64 | 30,832 | 110,630 | 141,462 |

## 解读
- **SRAM+DRAM 是 attention 能耗大头**（fp16 口径占总动态 ~66%）：只看 MAC 会严重低估。低精度同时省 MAC（×pj）
  和访存（×字节）两块，所以精度对总能耗的杠杆比只看 MAC 更大。
- **计算时间**：ARGUS 120.3s vs 基线 213.6s（基线三精度同周期）。ARGUS 32 半宽使列折叠翻倍（→慢），
  但 T1+T3 把有效 KV 从稠密大幅缩小（→快），净 **ARGUS 更快 44%**。
- **总动态能耗**：ARGUS 215,052 mJ（fp16+稀疏）vs FP16FP16 稠密 673,760（ARGUS 是其 0.32×，稀疏少算/少搬）。
  但 **FP8FP8 稠密 282,924**（比 ARGUS 高 32%），**FP4FP4 稠密 141,462** 则低于 ARGUS（ARGUS 是其 1.52×）。
- **含义**：attention 上 ARGUS 的优势主要来自**稀疏（少算少搬）+ 速度**，而非精度（它保 fp16 不量化以护精度）。
  统一低精度阵列(FP8/FP4)能在 MAC 和访存两头同时降能耗；若再叠加稀疏会更省，但 attention 低精度有精度风险。

## 注记
- attention 精度对 cycle 无影响是本仿真器的建模事实（fold 周期只由阵列几何 + kv_len 决定）→ 三基线计算时间相同。
- 总动态能耗不含 idle 背景(500mW)与阵列静态(150mW)——它们按整芯片运行时间计、非单算子可归因；如需端到端含静态请看 §A/§D 口径。
- per-MAC 与字节占位值待 RTL 标定；改 `coefficients.py` 的 `mac_pj` fp8/fp4（及本脚本 PREC_BYTES）即可重算。
