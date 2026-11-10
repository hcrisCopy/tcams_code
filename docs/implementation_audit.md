# T&M 免训练适配与实验实现审计

日期：2026-11-10。状态：已读官方源码并完成人工静态审查；未安装依赖或运行本项目代码。所有资源峰值、数值检查、模型得分均待服务器执行，不能写作“已通过”。

## 1. 代码来源与复用边界

官方仓库：[Cross-Architecture-Merging-for-Large-Language-Models](https://github.com/chenhangcuisg-code/Cross-Architecture-Merging-for-Large-Language-Models)，检查版本 `6a822ef33d386585c57fa4e7cdd79925454336a7`，位于 `../../tcams_research/transport_and_merge/`。运行时强制同一 commit 且工作区干净，外部代码不复制到本项目。

已阅读 README、REPRODUCE、MODELS、`run_activs_and_hot.py`、`activs_llama3_modules.py`、`core/hot_transport.py`、`core/hot_transport_chunk.py`、根目录及 core 的 `generate_hot_residual.py`、FineWeb chat builder，以及 `evaluation/thai/` 文档与脚本。论文原始公式另见 [T&M](https://arxiv.org/html/2602.05495)。

| 本项目部分 | 参考/调用 | 具体处理 |
|---|---|---|
| 激活 hooks | 官方 `activs_llama3_modules.py` | 改编 masked mean pre/post hooks；仅 Q/K/V/O；原生模型配置；不跳样本；保存 batch ID |
| 相关距离 | 官方 `core.hot_transport_chunk.corr_distance_matrix` | 直接调用；先检查样本数一致与输入有限，不允许内部 `min(T)` 截断掩盖错配 |
| feature OT | 官方 `sinkhorn_uniform_streaming` | 直接调用；明确传 epsilon、iters、chunk、fp32存储；不复写 Sinkhorn |
| layer OT | 官方 `compute_P_stable` | 直接调用；保留 median cost normalization；`max_eps_grow=0`，零矩阵/不收敛报错 |
| Q raw-space转换 | 官方 `run_activs_and_hot.py` | 同样 `sigma_A[:,None]*Q/sigma_B[None,:]`，边缘检查在转换前做 |
| 融合 | 官方 `fuse_attention_only_from_hot_dir` | 直接调用；A=target，B=source；direct weight fusion；不挂训练 residual |
| Delta干预 | 基于官方融合产物 | 比较实际checkpoint；按唯一参数切片分组；存原值并copy恢复，不反复浮点相减 |
| MGSM/Belebele评测 | 官方数据及 [lm-eval MGSM](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mgsm)、[Belebele零样本示例](https://github.com/facebookresearch/belebele/blob/main/sample_zero_shot_instructions.md) | 自定义已声明的zero-shot输出协议；不宣称官方harness复现 |
| 安全判分 | [Qwen3Guard官方模型卡](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)和tokenizer模板 | user+assistant回复判分；只解析新生成标签；Unsafe/Controversial/Refusal分开 |

T&M 当前仓库没有找到顶层 LICENSE 文件；公开可读不等于获得任意再许可。当前仅把它作为外部研究依赖，不为上游源码添加许可或作再分发承诺。模型/数据许可保存在 asset lock，详细核验见 [数据专题报告](../../tcams_research/related_work_evaluation.md)。

已通过HF连接器核验默认两只模型的公开元数据：均为Llama架构，均标记Llama 3.2许可。[3B源模型](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)、[Typhoon目标模型](https://huggingface.co/typhoon-ai/llama3.2-typhoon2-1b-instruct)。这只确认资源存在及发布元数据，不表示已经在本机下载、加载或验证能力优势。

## 2. 从源码发现的偏差与处理

| 发现 | 风险 | 本次实现 |
|---|---|---|
| `filter_nonempty_layer_pairs` 使用 `zip(X,Y)` | 层数不同可能截掉深源模型后半层 | 独立保存A/B全部层索引，计算完整L×M；属于明确工程修正 |
| 上游 `kept_layer_idx_B` 缺省复用A索引 | 不等深 bundle 解释错误 | 显式写 `kept_layer_idx` 和 `kept_layer_idx_B` |
| hook loader 将Llama3 RoPE改成dynamic | 实际模型与发布checkpoint的推理函数可能不同 | 保留原生config，不改RoPE |
| hook OOM降batch/跳样本，保存异常continue | A/B可能失配或不完整 | 适配层直接失败；已完成batch保留，不能自动改变batch |
| CLI `hot-chunk-cols` / `hot-dtype` 等曾仅记录 | 配置与实际求解不同 | 直接调低层solver，明确fp32存储；这与上游内部默认bf16 Q存储不同 |
| `compute_Q_and_layer_costs` 含NaN替换 | 非有限输入可能被隐藏 | 逐层对调用底层函数，并在前后严格检查；不调用该容错包装 |
| 外层P自适应增大epsilon、失败返回零矩阵 | 实验定义改变且可能继续 | 禁止增长并检查mass/marginal；失败停在OT阶段 |
| 融合可因top-neuron文件失败退回dense | 干预预算无意改变 | 预检所有索引，检查返回report中的fallback/skip，任何异常不保存完成标记 |
| fusion签名列出P/Q稀疏化参数 | 容易以为传参就生效 | 当前执行路径未调用相应稀疏化helper；不把这些值当有效实验旋钮 |
| O用post激活选择索引、融合却改列 | post-output语义与input列不一致 | 为保留官方选择行为暂不改为O_pre；报告明确该局限，禁止解释为精确输入神经元机制 |
| README对A/B称呼不一致 | 可能反向融合 | 从函数实参和权重写入核实A=小目标，输出保持A结构 |
| 论文Eq14与训练residual写法不同 | 中间张量不等于最终增量 | ΔW只用实际W_fused−W_target |

这些修正形成 **T&M-derived strict full-depth adapter**，不是逐行、逐数值复现原始脚本。原始校准chat builder含额外系统/assistant文本，本实现使用模型自己的user-only chat template；同样属于需记录的协议变化。若要对论文数字严格复现，须另建不变更协议的对照并解释原代码缺陷，不能把本配置分数直接与其表格相减。

## 3. 计算流程与资源边界

1. 每卡完整加载target采集自己的校准batch，再完整加载source采集相同batch；样本ID绑定。
2. 聚合固定顺序样本，保存每层Q/K/V/O pre/post特征。各层特征均为`样本数 × 特征维度`，不是把不同tokenizer的token序列硬拼成一一对应。
3. 对所有8类pre/post模块的全部L×M对计算相关代价、feature OT，并原子保存每个层对。
4. CPU组装每模块bundle、求P、保存两侧完整层索引；内存至少容纳当前模块Q列表。
5. 单卡调用官方融合两次：alpha=0的真实恒等检查，再alpha>0融合；不运行训练脚本。
6. 保存fused并逐参数检验；每组存target/fused切片与fp32增量，不保存每次消融的完整模型。

feature OT不会因为只用32条校准就变成32×32。以模型配置给出的投影维度计算，Q存储量为：

\[
4LM\left(\sum_{c\in\{Q,K,V,O\}}d^A_{out,c}d^B_{out,c}
+\sum_{c\in\{Q,K,V,O\}}d^A_{in,c}d^B_{in,c}\right)\ \text{bytes}.
\]

实际使用pair文件和assembled bundle两份存储，另有权重、激活和增量。`inspect`计算单份Q容量，不把它冒充GPU峰值。上游所谓streaming主要分块Sinkhorn归约，仍持有dense cost和Q；不是完全无需大矩阵。大RAM/磁盘与显存都要审计。

分片不用分布式collective；父进程检测任意worker非零退出就终止其他worker，不会继续汇总缺失结果。原子文件仅在完整保存后rename。融合checkpoint的中断粒度为整个融合阶段；已有激活和OT可重用。数据下载由HF续传，数据准备按确定性输入重建小JSONL；不是伪称每个函数都支持任意位置续算。

## 4. E0检查与源贡献审计

实现了：真实alpha=0全参数hash和probe logits相等；融合后logits有限；EN/TH简单问题可生成；alpha>0有实际参数变化；未选参数完全不变；target加所有分组可精确重构；异常退出也恢复内存patch。

上述是**实现的检查项，尚未执行**。量化/精度变更可能影响等式，当前禁止通过放宽容差掩盖错误。

重要额外对照：官方Q是概率质量计划，raw-space缩放后源项实际尺度不能凭公式想象。`delta/groups.json`记录fused切片与`(1-alpha)*target`的距离，`plans/shrink_control.json`提供源项为零的收缩对照。若fused与此对照几乎相同，必须先解释源贡献是否被精度或缩放压没；不能将单纯权重收缩效果称为能力迁移。当前保持官方融合语义，不擅自加维数重标定。

## 5. 数据、统计和已知未覆盖项

数据真实字段来自官方文件/card，详见 [评价报告](../../tcams_research/related_work_evaluation.md)。MultiJail官方共享ID用于EN/TH配对；MGSM-Rev2同题跨语言绑定；Belebele依据真实`ds`而非README中的`split`建立passage组。Belebele、ThaiSafetyBench、XSTest全部test-only；probe仅MGSM/MultiJail内部划分。确切重复文本的组会合并，任何与test-only集合连通的组整体进test。

统计包括逐题保存、同题配对差和95%cluster bootstrap，seed=42；各项能力分开报告。E2中的风险主指标固定为MultiJail，TH本地安全单列。单组效果不能相加代替联合评测，`joint_probe`与独立test计划都实际生成和判分。

未覆盖而不得声称完成的事项：

- GPU实测、测试运行、依赖完整安装验证、吞吐与资源实测。
- Thai MMLU知识指标、原论文few-shot/native-CoT协议复现。
- 自动可信的TH理解失败率或TH过拒答金标；提供人工审核数据入口与盲评导出，不能填默认0。
- 语义近重复/预训练污染彻底排除；当前只做ID隔离与字面去重。
- 随机删除的范数匹配/搜索成本完全匹配；当前匹配切片数与参数预算。
- 严格多重检验校正、跨模型/跨校准集复现实验、自动Go判定。
- SWE、SafeMERGE、SubSafe-Merge、LED原方法的完整异构适配及强方法比较。
- 精确source-layer贡献分解、OT边级干预、来源特征的额外价值证明。

## 6. 静态审查记录

检查了CLI入口、各阶段产物引用、schema与模型方向、只有下载阶段联网、seed=42、相对路径、无训练/隐式回退、分片与缓存文件命名、组干预恢复、风险符号和test-only约束。附带服务器测试覆盖配对split、judge标签、严格答案解析、异常时恢复、bf16差分重构、配对统计及矩形OT。按用户要求没有运行pytest、compileall、模型或实验；尚不能承诺无运行时缺陷。
