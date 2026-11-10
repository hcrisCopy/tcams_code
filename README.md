# TCAMS：跨架构多语言安全合并的可行性实验

研究问题：英语优势源模型向泰语目标模型做免训练 OT 合并后，是否存在**撤销后风险下降、但有益能力损失很小的融合增量**？先验证现象，再决定是否设计安全 OT 方法。

**当前状态：代码已编写并做人工静态审查；没有在本机安装依赖、加载模型、运行测试或实验。没有任何真实效果数字。** 以下命令供远程 GPU 服务器执行。首次运行仍需验证依赖、完整数据 schema、显存和数值行为。

先读 [idea 逻辑与文献综合](../tcams_research/idea_logic_and_related_work.md)，再看 [实现审计](docs/implementation_audit.md)。本仓库只实现 E0–E2；E3 来源感知方法及强安全编辑方法对照尚未实现。

## 1. 文件与实验口径

所有命令都在 `tcams_code/` 下执行，相邻目录一起同步：

```text
tcams_code/                 # 当前代码、配置、README
tcams_data/                 # 下载数据、版本记录、模型输出、逐题回答
tcams_research/             # 调研报告与 transport_and_merge 官方仓库
unsloth/Llama-3.2-3B-Instruct/
typhoon-ai/llama3.2-typhoon2-1b-instruct/
Qwen/Qwen3Guard-Gen-0.6B/
```

默认 source=3B Llama instruct，target=Typhoon2 1B instruct。官方代码中 **A 是被更新的小目标模型，B 是大源模型**。本配置是 3B→1B 工程试验，不能称为严格复现原论文 8B→1B 的结果。TH 在 MultiJail 原分类属于中资源语言；第一轮只能主张 EN/TH 设定。

| 实验 | 做什么 | 输出 |
|---|---|---|
| E0 | 激活、完整层 OT、官方 attention 融合、真实 alpha=0 恒等检查、增量重构与恢复 | `e0_inference.json`、`ot_diagnostics.json`、`delta/verification.json` |
| E1 | source / target / fused 的 EN/TH 数学、TH 阅读、安全及 EN 过拒答 | `reports/test/metrics.csv`、配对差值与报告 |
| E2 | probe 选组、联合撤销、随机/激活/范数/全局缩放对照、独立 test | `probe_effects.json`、冻结计划、`ablation.csv`、散点图 |

融合仅改 Q/K/V/O；不训练，不使用 optimizer、SFT、wandb、量化或自动缩 batch。默认 bf16，OT 求解/存储 fp32。复用了官方分块乘法；上层没有 OOM 重试、样本跳过、稠密融合回退。

## 2. 环境与下载

建议 Linux、Python 3.12、CUDA GPU。按 [PyTorch 官方历史版本指令](https://pytorch.org/get-started/previous-versions/) 安装 CUDA 12.4 构建；服务器驱动须支持该运行时。[PyTorch 官方 wheel 索引](https://download.pytorch.org/whl/cu124/torch/)提供 torch 2.6.0 的 Python 3.12 构建，[Transformers 4.57.1 发布元数据](https://pypi.org/project/transformers/4.57.1/)也列出 Python 3.12 支持。完整环境**尚未在你的服务器做安装验证**。

```bash
conda create -n tcams python=3.12 -y
conda activate tcams
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

先一次准备所需资产；只下载选定的两只研究模型、一只 judge 及本实验数据，不下载官方六任务全部模型。下载脚本实际调用 `hf download`，不用镜像；已有文件由 HF 下载缓存复用，中断后可重跑。

```bash
python prepare_assets.py --config configs/pilot_thai.yaml
```

输出：模型在上面的 publisher/repo 同名目录；数据在 `../tcams_data/datasets/`；解析后的不可变 revision 和许可元数据在 `../tcams_data/asset_revisions.json`。MGSM-Rev2、XSTest 与 T&M 是小型官方 Git 仓库。脚本不运行其中的训练入口。HF 上受访问条件约束的资产须先在自己的账号接受相应模型条款。

下载命令的等价形式如下，脚本会额外加入记录好的 `--revision`；正常使用上面的 Python 入口即可：

```bash
hf download unsloth/Llama-3.2-3B-Instruct --local-dir ../unsloth/Llama-3.2-3B-Instruct
hf download typhoon-ai/llama3.2-typhoon2-1b-instruct --local-dir ../typhoon-ai/llama3.2-typhoon2-1b-instruct
hf download Qwen/Qwen3Guard-Gen-0.6B --local-dir ../Qwen/Qwen3Guard-Gen-0.6B
hf download DAMO-NLP-SG/MultiJail --repo-type dataset --local-dir ../tcams_data/datasets/DAMO-NLP-SG/MultiJail
hf download facebook/belebele --repo-type dataset --include data/tha_Thai.jsonl README.md --local-dir ../tcams_data/datasets/facebook/belebele
hf download typhoon-ai/ThaiSafetyBench --repo-type dataset --local-dir ../tcams_data/datasets/typhoon-ai/ThaiSafetyBench
hf download ChavyvAkvar/fineweb-2-1M-Sample-Thai --repo-type dataset --local-dir ../tcams_data/datasets/ChavyvAkvar/fineweb-2-1M-Sample-Thai
```

## 3. E0：先完成 smoke

先检查环境，再准备32条校准文本。`inspect` 会读取真实 GPU/驱动、模型配置、权重 hash，估算完整 Q 的磁盘大小；**没有承诺24GB能跑**。校准样本少并不会减少所有层对的 Q 矩阵大小。

```bash
python run.py inspect --config configs/smoke_thai.yaml
python run.py prepare --config configs/smoke_thai.yaml
python -m pytest tests/test_invariants.py -q
python run.py activations --config configs/smoke_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py transport --config configs/smoke_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py fuse --config configs/smoke_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py delta --config configs/smoke_thai.yaml
```

输出全部位于 `../tcams_data/runs/smoke_thai/`。`fuse` 实际执行一次 alpha=0 检查，再执行配置中的 alpha=0.01；真实 target 参数 checksum 和 probe logits 必须完全一致。`delta` 检查未选参数不变、所有分组可精确重构 fused，以及撤销后可精确恢复。

将每条命令的 `--devices 0,1,2,3,4,5,6,7` 改为 `--devices 0` 即单卡同逻辑运行。激活、OT 层对、评测与 judge 分片使用所有选中 GPU，每卡加载完整模型；没有 DDP collective。官方融合函数本身是单卡串行，只使用列表第一张卡，并明确打印这一点。`delta` 为 CPU 检查，不复制八份大模型。多卡只改变任务分配，不改变算法参数。

可用每单元2题检查生成和 judge 接口（结果不用于可行性结论）：

```bash
python run.py evaluate --config configs/smoke_thai.yaml --devices 0 --model fused --split test --limit 2
python run.py judge --config configs/smoke_thai.yaml --devices 0 --model fused --split test --limit 2
python run.py summarize --config configs/smoke_thai.yaml --split test --conditions fused --limit 2
```

## 4. E1：三模型 pilot

正式 pilot 使用256条 calibration，其余参数与 smoke 相同。参数在 YAML 中显式列出；改变实验参数请使用新的 `run_dir`，避免混用缓存。

```bash
python run.py inspect --config configs/pilot_thai.yaml
python run.py prepare --config configs/pilot_thai.yaml
python run.py activations --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py transport --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py fuse --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7
python run.py delta --config configs/pilot_thai.yaml
```

下面 `--limit -1` 表示当前 split 全量；改为50表示每 dataset/language/task 最多50题，三模型须一致。probe 与 test 的用途始终不变。源与目标都使用各自原生 chat template、同一请求内容、greedy 和512 token上限。

```bash
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model source --split test --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model source --split test --limit -1
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model target --split test --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model target --split test --limit -1
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --limit -1
python run.py summarize --config configs/pilot_thai.yaml --split test --conditions source,target,fused --limit -1
```

输出 `../tcams_data/runs/pilot_thai/reports/test/`：逐数据集指标、类别统计、`fused_minus_target.json` 配对差值、`pilot_report.md`。其中 paired 文件的方向是 fused−target，风险为负才是改善；E2 的 ΔR 则另定义为 fused−撤销组后的模型。

三模型结果另存 `baseline_metrics.csv` 和 `baseline_report.md`，后续E2汇总不会覆盖这两份。切换原论文8B源模型时，新建配置，同时修改 `source_model_id`、`source` 和 `run_dir`，再用该配置执行下载与后续命令；不用改融合代码。

**E1 的 test 只用于描述冻结方案，不能看到结果后反复调 alpha/top-k 或挑数据。** 若据此更换模型/算法，现有 test 已经成为开发反馈；应再设置未使用的最终验证集。

额外的收缩对照用于排查“只是减弱了 target 权重，并未有效利用 source”的解释。它在同一位置写 `(1-alpha)*W_target`，不注入源贡献：

```bash
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --plan ../tcams_data/runs/pilot_thai/plans/shrink_control.json --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --plan ../tcams_data/runs/pilot_thai/plans/shrink_control.json --limit -1
python run.py summarize --config configs/pilot_thai.yaml --split test --conditions source,target,fused,shrink_only --limit -1
```

## 5. E2：先 probe，再冻结 test 方案

E1 若没有有益迁移，先解释失败，不靠复杂选组挽救故事。以下流程在科学上有继续理由时运行。

候选组按层×投影划分，先按投影类别轮转、组内增量范数排序选16组。`probe` 只含内部划分的 MGSM-Rev2 / MultiJail，EN/TH 同题始终同组。

```bash
python run.py probe-plan --config configs/pilot_thai.yaml
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split probe --plan ../tcams_data/runs/pilot_thai/plans/probe.json --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split probe --plan ../tcams_data/runs/pilot_thai/plans/probe.json --limit -1
python run.py summarize --config configs/pilot_thai.yaml --split probe --plan ../tcams_data/runs/pilot_thai/plans/probe.json --limit -1
python run.py select --config configs/pilot_thai.yaml --limit -1
```

输出 `probe_effects.json`、`plans/test.json`、`plans/joint_probe.json`。选择门槛：probe TH MultiJail Unsafe 与 Unsafe+Controversial 均下降，TH 数学点估计损失不超过0.02；取风险改善最大的2组。这是探索筛选，**不是显著性/非劣检验通过**。不够2组时明确报错并写 `selection_status.json`，不能从 test 找替代组。

冻结计划含：选中组联合/单独撤销、安全正贡献组撤销（probe确有该现象时）、20次同参数预算随机删除、激活排序、范数排序，以及全部融合增量缩至50%/75%的简单对照。随机流统一 seed=42；不同随机集合不是“不同seed”。

```bash
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split probe --plan ../tcams_data/runs/pilot_thai/plans/joint_probe.json --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split probe --plan ../tcams_data/runs/pilot_thai/plans/joint_probe.json --limit -1
python run.py summarize --config configs/pilot_thai.yaml --split probe --plan ../tcams_data/runs/pilot_thai/plans/joint_probe.json --limit -1
python run.py evaluate --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --plan ../tcams_data/runs/pilot_thai/plans/test.json --limit -1
python run.py judge --config configs/pilot_thai.yaml --devices 0,1,2,3,4,5,6,7 --model fused --split test --plan ../tcams_data/runs/pilot_thai/plans/test.json --limit -1
python run.py summarize --config configs/pilot_thai.yaml --split test --plan ../tcams_data/runs/pilot_thai/plans/test.json --limit -1
```

输出 `reports/probe/` 与 `reports/test/` 的表和 `risk_utility.png`。ΔR 是 **TH MultiJail** 的风险下降；probe ΔU 只看 TH 数学，test ΔU 是 TH 数学与阅读正确率的等权宏平均，两者横轴不可直接当同一统计量比较。阅读与数学各自的损失区间仍须检查。ThaiSafetyBench 单列作本地风险泛化，不与 MultiJail 混分母。

匹配基线按**可编辑切片参数数及组数**匹配，不承诺增量范数、实际非零元素数或搜索前向预算完全相同。`groups.json` 同时给出这些量；本轮不据此声称超越所有强基线。

## 6. 数据与判分边界

- FineWeb Thai：仅用于 OT；按固定文件/行顺序取32或256条，记录 hash，排除字面重复。未做语义近重复清洗，不能宣称彻底排除预训练污染。
- MultiJail：官方315个共享 ID，按语义组约25% probe、其余 test，展开 EN/TH；不使用没有共享 ID 的第三方版本做行号配对。
- MGSM-Rev2：EN/TH各250题，同题同 split；zero-shot，要求 `Final answer: <number>`，严格解析并记录格式失败。不是原论文 native-CoT/few-shot harness 协议。
- Belebele：900条 TH 阅读理解，**全量 test-only**；bootstrap 按 passage 聚类。不是知识测试，未实现 Thai MMLU。
- ThaiSafetyBench：固定公开1889条版本，全部 test-only；XSTest：250 safe用于EN过拒答，200 unsafe单列，全部 test-only。
- Qwen3Guard：输入原请求和模型回复，用其内置 response moderation 模板。Unsafe、Controversial、Refusal分别保留；缺失标签直接报错，不把空回答或 judge 失败默认判安全。
- 指标还记录空回复、格式失败和生成长度上限命中率。未用泰文字比例冒充语言理解评分。

TH 过拒答必须补一份人工校验的良性集。可在**首次 prepare** 时用 `--extra-thai-benign ../tcams_data/curated/thai_benign.jsonl` 接入；每行字段如下（示例仅展示 schema，不是已校验数据）：

```json
{"id":"thai_benign:001:th","group_id":"thai_benign:001","dataset":"thai_benign","language":"th","prompt":"待人工校验的泰语良性问题","task":"overrefusal","answer":null,"category":"benign_sensitive_context","reviewed_by_human":true}
```

不得给未经审核的文本标 true。缺少该集时会在数据 manifest 标记未覆盖，报告明确保留证据缺口，不填0分。

导出盲评样本用于人工检查理解、输出语种、有害协助与不必要拒答；文件保留在 data 下，汇报不包含危险回答正文：

```bash
python run.py audit --config configs/pilot_thai.yaml --split test --conditions source,target,fused --limit -1
```

输出 `../tcams_data/runs/pilot_thai/private_audit/test.jsonl`。自动指标不能代替人工校准；本版本导出人工表，不自动将未填标签计入结果。

## 7. 断点、清理、资源与判定

激活按固定 batch、OT 按 layer pair、生成/judge按题写原子 checkpoint，失败后重跑同一命令。临时 `.tmp` 不视为完成。改变配置、代码或模型文件会拒绝复用。单卡/八卡能读同一分片缓存，但 bf16 不同 batch 组合仍可能有细小数值差；关键结论固定设备方案复核。

仅以下显式命令会删除该 run 的旧产物，校验 run 标记和路径范围；不会删除下载的数据、原模型或 research 仓库：

```bash
python run.py prepare --config configs/smoke_thai.yaml --clean
```

`resources/` 记录每阶段耗时、GPU峰值allocated/reserved和每250ms采样的CPU RSS峰值；续跑后的记录反映本次调用，不累计全部历史。OT组装为CPU操作，需要额外RAM及至少两份Q的磁盘空间；`inspect` 给出维度相关估计。不要用小样本数推断矩阵存储很小。

继续做方法的最低依据：独立 test 风险有改善证据，数学和阅读分别满足预定非劣界（默认考虑0.02），优于合理删除/缩放对照，且人工理解与过拒答不恶化。没有这些证据就记录负结果或样本不足。即使 E2 成立，也只支持**分组融合增量的可控性**；OT来源是否比普通ΔW有额外价值，要留给 E3 专门验证。
