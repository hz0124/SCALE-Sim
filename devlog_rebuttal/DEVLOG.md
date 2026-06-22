# 开发日志 (DEVLOG)

> 维护者：dingli。项目全貌见同目录 `PROJECT_OVERVIEW.md`。
> 记录格式：每条带日期 + 状态。状态用 ✅完成 / 🔄进行中 / ⏸搁置 / ❓待确认 / 🐞问题。
> **倒序排列，最新在最上面。**

---

## 进度记录

### 2026-06-22 — devlog_rebuttal 接手整理：补全 README 导航 + 修文档与实物不符 ✅
- **目标**：让后续接手人靠 `README.md` 一份就能不踩坑地上手；只动文档，不挪/删数据文件。
- **修 README 三处「文档指向不存在的脚本」陷阱**（接手最容易卡的地方）：
  - RESULTS.md §C 写「由 `run_timetable.py` 生成 `/tmp/timetable.json`」——该脚本**不在仓库**（曾在临时
    worktree、已丢，同 need6 ctxlen 的情况）。已改注明要重刷 §C 得重建生成 `/tmp/timetable.json` 的脚本。
  - RESULTS.md §D 写「由 `argus_energy.py` 追加」——**无此脚本**；§D 实为手工维护（数字取自 `--energy` 的
    ENERGY BREAKDOWN 段）。§S(SenseNova) 同为手工。已注明 `compose_RESULTS.py` 只重生成 §A/§B/§C，重跑会
    冲掉手工的 §D/§S，记得贴回。
  - 补上 `run_base.py`/`summary_base.*` 说明（HCA 归一前的早期 §A 跑，run_full ablation 在其 all_on 行上展开），
    此前 README 完全没提。
- **补 `img_attn_array_compare/` 进 README 专题目录**：06-18 02:25 新增的独立支线分析（图像阶段 attention
  ARGUS vs 统一 32×64 FP16/FP8/FP4），晚于上次 README 重写故一直没被导航收录；不在 RESULTS.md 里。
- **未动**：所有数据/脚本/归档目录结构保持原样（结构本身已干净）。

### 2026-06-18 — rebuttal 数据冻结 + 代码审计修 bug + devlog 整理 ✅
- **数据冻结**：rebuttal 暂无待跑数据。权威结果全部在 `overnight_2026-06-17/`：`RESULTS.md`
  §A0(SA/HCA/ARGUS 摘要) + §A1(加速比/能效比+Geomean) + §A2(功率/功率比) + §B(ablation 加速比+能效比) +
  §C(阶段×算子时间，HCA 归一) + §D(ARGUS 能量分解 Compute/SRAM/DRAM/Static)；`BREAKDOWN.md`、
  `SENSENOVA.md` 同口径。统一系数：DDR4 20pJ/bit + DRAM 背景 500mW + 阵列静态 150mW + per-hw w4a16。
- **修真 bug（日志污染根因）**：`run_model` 里"清空旧日志"写成 `os.path.join(result_path, "results.log")`，
  但 `result_path` 是文件路径 → 得到永不存在的路径 → `os.remove` 从不执行 → 每次 append 污染。
  本 session 反复手动删日志/取 [-1] 全因它。三处（bagel/janus/sensenova `run_model`）改成 `log_file = self.result_path`。
  冒烟验证：连跑 2 次只剩 1 个 ENERGY BREAKDOWN。**不改已提交数据**（driver 跑前都手动删过 log）。
- **顺手修注释**：`coefficients.get_coefficients` docstring 补 disagg/axcore + per-hw w4a16；
  sensenova 残留的 "Bagel/Janus" 复制注释改成 SenseNova。
- **审计记录的已知简化（未改，会动数字/非 bug）**：①GQA 文本注意力 KV 按 `num_head_q` 重复计（应 ×kv head，
  bagel/janus/sensenova 一致的历史简化，会高估文本 KV 访存）；②janus/sensenova 无分阶段利用率日志块（只 bagel 有）；
  ③janus 无 batch_size（need-6 batch 只跑 bagel）。见 `PROJECT_OVERVIEW.md` §10。
- **need-6 全部重跑**：在新模型 + 最终能量系数下重跑 5 套扫描——ctxlen(9 点) + batch(5hw×7) +
  argus_ablation(24) + technique(ctx32+batch32)，共 132 run，全成功。ctxlen 的生产脚本原在临时
  worktree（已丢），据现有 CSV 的 9 个上下文点 + profiled 稀疏参数**重建** `run_need6_ctxlen_sweep.py`。
  新增 `gen_need6_summary.py` 从 JSON 重生成权威总结 `need6_newmodel_summary.md`；`compose_need6_ablation_md.py`
  刷新 `need6_context_length_sweep.md` 加速比块；两个 MD 的过时 banner 更新。cycle 比与旧版基本一致
  （image 主导），能量随最终系数刷新。是日志清空 bug 修复后的干净跑（无 append 污染）。
- **devlog 整理**：旧模型 06-15 全矩阵（RESULTS/BREAKDOWN/summary/logs/driver）归档到
  `_archive_oldmodel_pre0617/overnight_2026-06-15_oldmatrix/`；`overnight_2026-06-15/` 现只剩新模型 need6。
  重写 `README.md`（新导航 + 权威/归档分界）、`PROJECT_OVERVIEW.md`（硬件表 +axcore/disagg/HCA/sensenova、
  能量系数更新）、`RERUN_PLAN.md`（三任务全标完成）。

### 2026-06-17 (晚) — DDR4 + AxCore + HCA 归一，§A 大表重做 ✅
- **DRAM 改 DDR4**：`dram_type` 默认 HBM2→**DDR4 (20 pJ/bit)**；`dram_idle_mw` 100→**500mW**（DDR4 子系统背景）。
  发现：改每比特读写能量对能效比近乎中性（DRAM 动态两家占比相近，~8%→32%同步涨）；真正杠杆是
  **背景/静态这类"运行时间项"**（罚慢的 FIGNA）——dram_idle 100→500 把 ARGUS-vs-FIGNA 能效从 1.40→1.64。
- **per-hw w4a16**：ARGUS 0.47 / FIGNA·disagg 0.57 / **AxCore 0.50**（int4×fp16 由 16-bit 激活主导，
  介于 int4 0.1 与 fp16 1.5；ARGUS 双阵列单元最省）。能效比对此最敏感（FIGNA 90% MAC 是 w4a16）。
- **新增 AxCore 硬件** = FIGNA 同款 W4A16，仅 w4a16 系数 0.50（图 figna 0.57）。代码：coefficients 加 axcore、
  两个 sim 的 figna 分支补 axcore、run_bagel/janus choices 加 axcore。验证 AxCore 周期/util 与 FIGNA 完全一致。
- **新增 HCA** = ARGUS all-off（异构阵列基线）作 §A **归一基准**。
- **§A 大表重做**：硬件作行（HCA/FIGNA/AxCore/FlightVGM/S-DMA/ARGUS，HCA 首行=1.00）、任务作列组
  （Bagel GenEdit/GenEval/MM + Janus GenEval/MM 合一张表）、加速比/功率比/能效比全归一到 HCA、去×号。
  脚本 `run_baseA.py`+`compose_baseA.py` → `baseA_table.csv`/`RESULTS_baseA.md`。base 不进表。
- ⏳ 待办：§B/need6/NEED9/BREAKDOWN/time-breakdown 受 DDR4+idle500+axcore 影响，需在新系数下重刷。

### 2026-06-17 (下午) — 能量加全芯片静态功耗底（修慢设计偏袒）+ 文本访存模型微调 ✅
- **问题**：FIGNA 功率算出来只 0.461W（ARGUS 1.045W），能效看着比 ARGUS 还接近。查因：能量模型是
  **动态下界**（只 MAC/SRAM/DRAM 动态 + DRAM 背景），**不含计算阵列漏电+时钟树**。FIGNA 跑 2.6× 长
  却不付阵列静态 → 慢设计被系统性偏袒、能效高估。
- **修复**：`array_static_mw`（默认 0→**150mW 占位**）按 `静态功率×运行时间` 给**所有 hw**计全芯片
  (2048-lane)静态能量（`accountant.compute_array_static_pJ`），慢的多付。取代旧的"仅 disagg idle 半芯片"
  特例（whole chip 一直漏电，与活跃无关，更正确）。效果：ARGUS-vs-FIGNA 能效 1.14×→1.32×
  （charge 满 tile 静态时 ~1.55×，接近论文 1.52× → 论文口径很可能含静态）。占位值待 RTL 标定。
- **文本访存模型微调**：`_text_gemm_cycles` 的流式项补上激活字节（ifmap/ofmap），使文本访存利用率
  ≤100%（之前循环论证恒 100%，且 attn 因 ofmap 略 >100%）。bagel+janus 同步。
- **能效口径澄清**（user）：能效比 = 加速比 × 功率比（恒等于能量比，已逐行验证）；FIGNA 低功率使其能效
  基线不弱——ARGUS"用功率换速度"，净能效增益 < 加速比。RESULTS §A 改 **FIGNA 归一** + 加 加速/功率/
  功率比/能效 分解列。
- **全量重刷**：静态功耗 + 文本微调影响所有 hw 能量 → base/full/supp/need6/timetable/need9-split 全部
  在新口径重跑，RESULTS/BREAKDOWN/need6/NEED9 文档重生成。framework §6.5 改为"动态+静态底"。

### 2026-06-17 — 文本 decode 改解析 memory-bound 模型 + need-9 split/带宽扫描 ✅
- **根因（need-9 带宽实验触发）**：想给 disagg 做"带宽劈半"对比，但实测砍带宽对文本阶段零影响。
  3 探针 + 解析 AI 定位：SCALE-Sim 的 OS 把 M=1 decode 的 M 映射到阵列 32 行 → 只用 1/32 阵列
  （util ~3%），被欠利用放大 ~37× 的计算时间盖死所有访存，stall 恒 0、任何带宽旋钮（片外/片上）
  都不影响周期（os/ws/is 全验证）。物理上 M=1 应是 memory-bound（roofline 拐点之下），是 M=1 欠利用
  把等效计算屋顶砸到访存屋顶之下。SCALE-Sim 为大 M CONV/GEMM 设计，表达不了 GEMV。
- **修复**：文本投影/FFN/注意力改解析 roofline `cycles = max(M·N·K/text_array_macs, N·K·operand_bytes/BW)`
  （helper `_text_gemm_cycles`/`_text_array_macs`/`_text_bw_bpc`，**bagel + janus 都改**）。
  文本周期缩 ~14×、变 memory-bound（text_mem≈100%）、半带宽精确 2×、权重精度成主导（int4 反超 fp16）。
  能量账本 + 图像阶段（路径 A/B）不变。详见 `docs/REBUTTAL_FRAMEWORK.md` §3.1 路径 C / §6.13。
- **need-9 产出**：5 档 split（w2/4/8/16/32）× 满/半带宽 × 分阶段 4 利用率（text/image × compute/mem，
  全芯片 2048 口径，能量账本快照差分）。今日全 11 配置在新模型上重跑：ARGUS 全胜（util 85%、611.8e9
  最快）；disagg 最优 split 从旧模型 w8 翻转到 w2。脚本 `need9_logs/aggregate_splits.py` →
  `split_sweep.csv`；文档 `docs/NEED9_DISAGG.md` §6-7（§4 标记过时）。
- **影响面 + 整理**：所有含文本旧数作废（MM 全部、GenEval/GenEdit 文本部分、ablation；旧不变量
  `ours MM=127,169,309,589` 故意作废）。重跑计划见 `docs/RERUN_PLAN.md`。整理 devlog_rebuttal：旧模型数据
  （`overnight_2026-06-13/`、旧 need9 日志）移入 `_archive_oldmodel_pre0617/`（可逆，可删）；need-6 数据
  （`overnight_2026-06-15/need6/`，14:37 跑、旧模型）待 dingli 自行重跑。

### 2026-06-09 — 通读所有 cfg：发现①精度不在 cfg ②FlightVGM 稀疏没配 🔎
- **①精度仅靠文件名**：所有 `configs/bagel/*.cfg` 无任何数据类型/位宽字段，只有 array/SRAM/BW/dataflow/layout/sparsity。精度纯靠 `precision_from_config_path` 嗅文件名（`ours_int4.cfg`→int4）。各精度 cfg 差异在架构（int4 filter SRAM 128kB / int8 BW16）。→ **#2 的 per-tensor 数据类型 repo 里没有，必须查论文/问 xinhao。**
- **②FlightVGM 稀疏未配（两条路都没开）**：
  - SCALE-Sim `[sparsity] SparsitySupport`：**所有 cfg 都是 false**（含 flightvgm）。
  - harness KV-缩短稀疏：flightvgm 落 `else` 分支(`bagel_sim.py:602-605`)→ **全量 KV**。仅 `ours`/`ours_balenced`/`sdma` 吃稀疏（`ours`=sparsity_cross_attn+low_precise_self_attn，`sdma`=sparsity_kv）。flightvgm JSON 无 sparsity 字段。
  - 另：harness 稀疏机制只有「缩 KV 注意力长度」一种，**FlightVGM 论文的稀疏（偏激活/视觉）类型不同，现机制也表达不了**。
- ⚠️ **baseline 公平性隐患**（比 #2 更值得注意）：现对比里只有 ours/sdma 吃稀疏红利，FlightVGM/FIGNA/base 全密。若 FlightVGM 卖点是稀疏 → 等于削弱后再比，reviewer 可戳。需确认是 xinhao 有意简化还是漏配。
- 待办：①向 xinhao/论文要 per-tensor 数据类型（#2）；②确认 FlightVGM「密建模」是否有意。

### 2026-06-09 — 能耗模型定案：几何闭式 + 修 ifmap/ofmap 角色 ✅
- 背景一句话：曾试过把文本路径能耗改读 tile CSV 的真实 DRAM 计数，但实测发现 SCALE-Sim 的 `DRAM IFMAP Reads` 恒≈32768、完全不随 free 维 N 变化——是「填满固定 prefetch buffer」的 artifact 而非真实流量。M=1 解码无 SRAM spill，**几何闭式反而更贴物理**，故回退 CSV 路径。
- 决策：回退 CSV 改动，改用几何闭式 + 修角色 bug。
- 操作：`git checkout` 抹掉两个 sim 文件的全部 CSV 改动（含注意力重排）→ 回到 prefer_csv 死代码态。**只改 `extractors.py`**：
  - `dram_ifmap`: `M*N` → `M*K`；`dram_ofmap`: `M*K` → `M*N`；`sram_ofmap`: `M*K` → `M*N`。角色按 SCALE-Sim 实测（M=1,N=64,K=3584：SRAM IFMAP=3584=M·K、DRAM OFMAP=64=M·N）校正，docstring 同步。
- 验证（Bagel MM `--energy`）：**总能耗 29635.328 mJ**（=原几何基线，角色互换对总账无影响，符合预期）；DRAM ifmap/ofmap 数值恰好互换（13.123/13.020 → 13.020/13.123）；cycles 52.2B 不变；exit 0。
- 结论：能耗模型现在是**干净、物理可辩护的几何闭式**（M=1 解码无 SRAM spill，闭式近似精确），无 prefetch 假象。`--energy` 数量级与之前一致，可放心用。
- 残留处理：**已删除** `prefer_csv`/`detail_csv`/`_last_run_detail_csv` 整套死代码（两个 sim 的参数+分支+run_sim_once stash+__init__ 属性）及 `extractors.py::from_scalesim_reports`（+`_DETAIL_COLS`+pandas import）。`extractors.py` docstring 留一段说明「为何移除：tiled DRAM 计数是 prefetch artifact」。`--validate`(accelergy CLI) 路径不依赖它，未受影响。

### 2026-06-09 — Accelergy / Phase-F 能耗集成：审查 + 端到端验证 ✅（能跑、数量级合理）
- 结构：能耗是两层——①上游 PR#7（`rundir-accelergy/` + `accelergy_plugin.py` + hook/PE 计数/DETAILED_ACCESS 加列，已并入本分支）；②Phase F 自研 pJ 核算（`simulation_core/energy_accounting/`，`--energy` 快速核算 + `--validate`/F7 调 accelergy 交叉校验，F7 自标 "(simplified)"）。
- 跑通证据：`run_bagel --hw ours --task MM --energy --validate` exit 0；Total 52.2B cyc / 29635 mJ，分项 Compute 0.4% / SRAM 0.2% / **DRAM 99.4%**（HBM2，memory-bound）；accelergy ratio 数量级 sanity（int4 68x / int8 33.8x，非 1.0x 是设计使然）。
- ⚠️ 仍存在的已知短板（抗 review 时要注意）：
  1. **单精度套三张量**：`word_bytes` 由单一 precision 推出同时套 ifmap/filter/ofmap → 低估激活/输出字节。`bagel_sim.py:158`。
  2. **「3 样本校验」实为 1 个形状**：`_tile_down` 砍到≤128 + M=1 → top-3 子仿真全塌缩成同一 1×128³ → ratio 恒定是数学必然非校准证据；护栏永不触发。**最易被 reviewer 戳。**
  3. **PR#7 插件 copy-paste bug**：`accelergy_plugin.py:38` `ofmap_sram_repeat` 误用 `ifmap_trace_matrix`；仅影响 ofmap-SRAM(~0.2%)，可忽略但是真 bug。
  4. **跨硬件区分度**：6 种硬件共用一张 28nm 系数表，差异只来自 access-count，非 per-hw 系数。若论文要严肃跨硬件能耗图需补。

### 2026-06-09 — 接手起点 ✅
- dingli 接手项目（git 署名已设为 `dl0321`，分支 `feat/argus-energy-accounting` 已 push 到 origin `hz0124/SCALE-Sim`）。Claude 通读代码产出 `PROJECT_OVERVIEW.md`。
- 跑通证据基线：`results/bagel/results_ours_GEdit_balenced_new.log`（2026-06-06）含完整 ENERGY BREAKDOWN + CROSS-VALIDATION 段。
