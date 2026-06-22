> ℹ️ **2026-06-18 全部重跑（新文本 memory-bound 模型 + 最终能量系数 DDR4/idle500/static150/per-hw w4a16）**：
> ctxlen sweep 的生产脚本已重建（`run_need6_ctxlen_sweep.py`，9 个上下文点，ours all_on）并重跑，
> §3 的 all_on-vs-all_off 加速比块（`<!-- ABLATION -->` 之间）由 `compose_need6_ablation_md.py` 刷新。
> §1 的 profiling 等效参数表是输入口径、不随模型变。**技术点贡献的权威汇总见 `need6_newmodel_summary.md`**。

# 需求6:longer context + batch size sweep

ARGUS rebuttal framework 需求6,两个独立的 sweep。**两组数据不要混到一起**:context-length 改的是 KV/工作负载规模,batch-size 改的是并发请求数,坐标轴和结论都不同。

- §1 多 context-length sweep —— profiling 统计(每个上下文长度的稀疏度/复用率等效参数)。
- §2 多 batch-size sweep —— 固定当前 BAGEL+GEdit 设置,扫 batch=1,2,4,8,16,32,64,**所有硬件都要跑**(不只 ARGUS)。

---

## §1 多 context-length sweep

每个 context length 对应的逐点 profiling 统计,用作驱动 harness 的等效参数。

列说明:
- **context length** — 上下文长度
- **cross 部分占比** — cross-attention 占比
- **self 部分占比** — self-attention 占比
- **frac_text** — text 部分占比
- **cross 稀疏度** — cross-attention 稀疏度
- **self 稀疏度** — self-attention 稀疏度
- **sp_overall** — 整体稀疏度 = (cross 稀疏度 + self 稀疏度) / 2(简单算术平均,未按占比加权)
- **cfg_text 复用率** — text-cfg 复用率(对应 JSON `text_only_sim`)
- **cfg_img 复用率** — image-cfg 复用率

| context length | cross 部分占比 | self 部分占比 | frac_text | cross 稀疏度 | self 稀疏度 | sp_overall | cfg_text 复用率 | cfg_img 复用率 |
|---|---|---|---|---|---|---|---|---|
| 1497 | 0.6346 | 0.2779 | 0.0875 | 0.3702 | 0.3575 | 0.3638 | 0.2494 | 0.2070 |
| 2566 | 0.6633 | 0.2853 | 0.0514 | 0.4090 | 0.3867 | 0.3978 | 0.2723 | 0.1829 |
| 3211 | 0.6686 | 0.2909 | 0.0408 | 0.4217 | 0.4079 | 0.4148 | 0.2844 | 0.1842 |
| 5648 | 0.6811 | 0.2943 | 0.0246 | 0.4877 | 0.5256 | 0.5067 | 0.2900 | 0.1884 |
| 7609 | 0.6855 | 0.2964 | 0.0180 | 0.4978 | 0.5457 | 0.5217 | 0.3021 | 0.1927 |
| 8754 | 0.6875 | 0.2967 | 0.0159 | 0.5125 | 0.5741 | 0.5433 | 0.2962 | 0.1938 |
| 9909 | 0.6875 | 0.2984 | 0.0140 | 0.5297 | 0.6154 | 0.5726 | 0.2863 | 0.1940 |
| 11969 | 0.6884 | 0.2986 | 0.0129 | 0.5375 | 0.6256 | 0.5816 | 0.2536 | 0.1932 |
| 12018 | 0.6861 | 0.2996 | 0.0144 | 0.5338 | 0.6237 | 0.5787 | 0.2636 | 0.1873 |

趋势:随 context 增大,cross / self attention 稀疏度均上升,整体稀疏度 `sp_overall` 约 0.36 → 0.58;text 占比 `frac_text` 持续下降趋向 ~0.014;attention 占比趋稳(cross ~0.69、self ~0.30);复用率基本持平(text-cfg ~0.25–0.30,image-cfg ~0.18–0.21)。

### 仿真结果(ours,BAGEL + GEdit,batch=1)

每个 context length 用上表对应的等效参数驱动 harness(`kv_cache_init`=context length;`sparsity_cross_attn`=cross 稀疏;`low_precise_self_attn`=self 稀疏;`text_only_sim`=cfg_text 复用;`image_only_sim`=cfg_img 复用;`image_len`/`text_len` 保持 JSON 原值 3193/242)。完整含每算子 cycle/energy 拆分见 `need6_ctxlen_sweep_results.{json,csv}`,采集脚本 `run_need6_ctxlen_sweep.py`。

| context length | total cycles | seconds(@500MHz) | energy (mJ) | avg power (W) |
|---|---|---|---|---|
| 1497 | 665,296,248,986 | 1330.59 | 2,590,915 | 1.95 |
| 2566 | 675,109,997,707 | 1350.22 | 2,653,951 | 1.97 |
| 3211 | 677,898,316,212 | 1355.80 | 2,686,183 | 1.98 |
| 5648 | 712,563,105,245 | 1425.13 | 2,882,066 | 2.02 |
| 7609 | 741,722,118,394 | 1483.44 | 3,031,947 | 2.04 |
| 8754 | 760,951,778,599 | 1521.90 | 3,137,799 | 2.06 |
| 9909 | 781,827,574,694 | 1563.66 | 3,254,447 | 2.08 |
| 11969 | 827,217,795,234 | 1654.44 | 3,468,392 | 2.10 |
| 12018 | 821,033,388,775 | 1642.07 | 3,455,296 | 2.10 |

cycles/energy 随 context 单调上升(KV 变长 → attention 工作量增大),但增速被稀疏度上升抵消了一部分(context ×8 而 cycles 仅 ×1.23);12018 略低于 11969 是因为其稀疏度更高。avg power 随 context 略升(attention 占比上升,而 DRAM 背景+阵列静态这类运行时间项随更长运行累积)。

> ⚠️ 注意:context length=1497、2566 小于输入图像 token 数 `image_len`=3193,导致 `kv_cache_without_img` 为负(-1696 / -627)。ours 的 image-attn 公式里 `image_input_len` 占主导,实际传给模拟器的 KV 仍为正,不会崩;但这两点的 `without_img` 分支语义上是退化的(context 容不下整张输入图)。需要的话可对小 context 单独调整 `image_len`。

---

## §2 多 batch-size sweep

### 需求

证明 batch 对 MLLM decode 吞吐的影响。固定当前 BAGEL + GEdit 的工作负载/硬件设置(`config_<hw>.json` 原值),只扫 batch size。**适用于所有硬件类型**(`ours` / `figna` / `flightvgm` / `sdma` / `base`),不只 ARGUS —— 用于对比各加速器在批处理下的吞吐 scaling 行为。

- **task**:GenEdit
- **batch sizes**:1, 2, 4, 8, 16, 32, 64
- **硬件**:ours, figna, flightvgm, sdma, base(5 种全跑)
- **配置文件**:`config_ours.json` / `config_figna.json` / `config_flightvgm.json` / `config_sdma.json` / `config.json`(base)

### 建模口径(真实 decode batching)

batch 对三类算子的影响不同(在 `simulation_core/bagel_sim.py` 中实现):

1. **投影 / FFN(qkv、omap、ffn_up、ffn_down)** —— 权重在 batch 内共享,B 个 M=1 的 decode GEMM 合并成一个 **M=batch 的 batched GEMM**(`build_topologies(..., m=batch_size)`)。当 batch ≤ array_height(32)时只占 1 个 row-fold,cycles 几乎不随 batch 增长 → **吞吐近线性提升**;batch=64 时 2 个 row-fold,投影 cycles 约翻倍。
2. **attention(attn_qk、attn_sfmxv)** —— 每个 request 有独立 KV cache,无法合并成大 GEMM → 工作量 **× batch_size**,线性增长,无利用率收益。
3. **image 阶段** —— M=image_input_len=1378 ≫ array_height,投影本来就把阵列填满,batch 无利用率收益 → 整个 image 阶段简单 **× batch_size**。

预期结论:小 batch 时投影/FFN 占主导,吞吐随 batch 近线性上升;大 batch 时 attention(线性)与 image 阶段(纯 ×batch)主导,**per-request 吞吐饱和**。Bagel GEdit 以 image 生成为主,batch 的端到端收益受 image 阶段(无收益)拖累。

### 运行方式

新增 `--batch` CLI 覆盖 JSON 的 `batch_size`(默认 1,batch=1 与原无 batch 数完全一致)。

```bash
# 单点
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json --batch 8 --energy
# 全 sweep(5 硬件 × 7 batch,串行,自动收集 cycles+energy+breakdown)
python devlog_rebuttal/overnight_2026-06-15/need6/run_need6_batch_sweep.py     # 或附 hw 名跑子集
```

### 记录项

driver `devlog_rebuttal/overnight_2026-06-15/need6/run_need6_batch_sweep.py` 对每个 (hw, batch) 记录,产物为
`devlog_rebuttal/overnight_2026-06-15/need6/need6_batch_sweep_results.{json,csv}`:

- **总量**:total_cycles、seconds(@500 MHz)、throughput(req/s = batch/seconds)。
- **能量**:total energy(mJ)、energy_per_req、compute / SRAM / DRAM 拆分、avg power(W)、EDP。energy 已做成 batch-aware(text 投影 M=batch、attention 与 image 阶段 multiplicity ×batch)。
- **算子 breakdown**(text / img × qkv / attn / omap / ffn_up / ffn_down):每算子的 cycle 贡献(`cycle_breakdown`)与 energy 贡献(按 audit-trail label 聚合);随 batch 缩放。

### 结果

> total_cycles(@500 MHz → 秒);per-request throughput ∝ batch / total_cycles。完整含 energy 拆分与每算子 cycle/energy breakdown 见 `need6_batch_sweep_results.{json,csv}`。

**total_cycles**(5 硬件 × 7 batch 全完成):

| hw | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 |
|---|---|---|---|---|---|---|---|
| ours | 611,855,247,099 | 1,201,832,955,382 | 2,381,788,371,948 | 4,741,699,205,080 | 9,462,942,386,096 | 18,925,451,423,584 | 37,850,902,847,168 |
| figna | 1,579,980,055,547 | 3,148,285,962,230 | 6,284,897,775,596 | 12,558,121,402,328 | 25,115,976,886,192 | 50,231,953,772,384 | 100,463,907,544,768 |
| flightvgm | 814,192,457,339 | 1,581,688,319,222 | 3,116,680,042,988 | 6,186,663,490,520 | 12,326,630,385,584 | 24,606,564,175,712 | 49,166,431,755,968 |
| sdma | 846,126,972,539 | 1,645,557,349,622 | 3,244,418,103,788 | 6,442,139,612,120 | 12,837,582,628,784 | 25,628,468,662,112 | 51,210,240,728,768 |
| base | 856,869,026,939 | 1,667,041,458,422 | 3,287,386,321,388 | 6,528,076,047,320 | 13,009,455,499,184 | 25,972,214,402,912 | 51,897,732,210,368 |

**per-request 吞吐**(req/s = batch / (cycles/500MHz),归一化到各自 b=1):

| hw | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 |
|---|---|---|---|---|---|---|---|
| ours | 1.00× | 1.02× | 1.03× | 1.03× | 1.03× | 1.03× | 1.03× |
| figna | 1.00× | 1.00× | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× |
| flightvgm | 1.00× | 1.03× | 1.04× | 1.05× | 1.06× | 1.06× | 1.06× |
| sdma | 1.00× | 1.03× | 1.04× | 1.05× | 1.05× | 1.06× | 1.06× |
| base | 1.00× | 1.03× | 1.04× | 1.05× | 1.05× | 1.06× | 1.06× |

**绝对吞吐(b=1,req/s)与能量**:

| hw | b=1 吞吐 (req/s) | vs ours | energy b=1 (mJ) | energy b=64 (mJ) | avg power b=1→b=64 (W) |
|---|---|---|---|---|---|
| ours | 8.172e-4 | 1.00× | 2,443,049 | 140,451,585 | 2.00 → 1.86 |
| flightvgm | 6.141e-4 | 0.75× | 4,718,390 | 267,888,608 | 2.90 → 2.72 |
| sdma | 5.909e-4 | 0.72× | 4,832,757 | 275,208,136 | 2.86 → 2.69 |
| base | 5.835e-4 | 0.71× | 4,912,847 | 280,333,912 | 2.87 → 2.70 |
| figna | 3.165e-4 | 0.39× | 3,978,337 | 246,115,267 | 1.26 → 1.22 |

### 结论

- **batch 吞吐收益普遍很小**(最多 +6% 后饱和):GEdit 端到端被 image 生成阶段主导,而 image 阶段已 array-saturated(M=1378≫32),batch 对它纯 ×B、无利用率收益;收益只来自 text 阶段投影/FFN 的 batched-GEMM 摊销,而**新 memory-bound 文本模型下 text 阶段占比很小**,故批处理收益比旧模型(曾报 +16~23%)弱得多。
- **各硬件 batch scaling 不同**:flightvgm/sdma/base ~+6%(text 投影摊销空间最大),ours ~+3%,figna ~+1%。
- **ours 绝对吞吐最高**:b=1 是 base/sdma/flightvgm 的 ~1.3–1.4×、figna 的 ~2.6×。
- **能量随 batch 近线性**;avg power:flightvgm/sdma/base 最高(~2.9W)、ours 2.0W、figna 最低(1.26W,W4A16)但延迟最高。注:含 DRAM 背景(500mW)+阵列静态(150mW)运行时间项后,功率普遍比旧的纯动态口径高。

---

<!-- ABLATION_START -->

## §3 ARGUS all_on vs all_off 加速比(MM-Vet / GEdit)

ARGUS 全技术栈(T1 稀疏注意力 / T2 FFN 复用 / T3 阶段自适应异构阵列 + INT4 投影量化)相对全关基线的端到端加速比,随 **上下文长度** 与 **batch** 的 scaling。

- **all_on** = ours 默认(5 开关 + quant_proj);**all_off** = 5 开关 false + `quant_proj=false`(need-4 ablation 基线:text fp16、单路 full-KV 注意力、无复用、image 宽 32)。
- **加速比 = cycles_all_off / cycles_all_on**(>1 = all_on 更快)。
- MM-Vet = MM 任务,只跑 text decode(差异来自 T3 精度 + quant_proj);GEdit = 图生成全流程。
- ⏳ = 仍在采集。

### 加速比 vs context length(batch=1)

| Context | MM-Vet | GEdit |
|---|---|---|
| 2k (2,566) | 3.23× | 2.35× |
| 4k (3,211) | 3.10× | 2.39× |
| 8k (7,609) | 2.52× | 2.50× |
| 12k (12,018) | 2.19× | 2.54× |

### 加速比 vs batch size(base context)

| Batch size | MM-Vet | GEdit |
|---|---|---|
| 1 | 3.10× | 2.65× |
| 4 | 2.10× | 2.66× |
| 16 | 1.23× | 2.67× |
| 64 | 1.00× | 2.66× |

> 进度:16/16 个加速比已算出。

<!-- ABLATION_END -->
