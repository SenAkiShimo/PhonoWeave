# PhonoWeave v5 — 连续 WAV 事件 → 用户原生单元的候选关联

## 这是实际运行的实验，不是音素分割完成宣告

使用前两轮**冻结**的 LynneQing 七音阶材料，将两个已有模块接起来：

1. **v4 自主事件提议**从完整 WAV 检出内部变化点，不读取 OTO 锚点、拼音或 IPA；
2. **v3 声学原生表示**（事先在开发源 WAV 模板上固定的 `late_context` 三段 80ms 倒谱）只用于给这些事件的前部声音提议现有用户标签；
3. 把事件末端接回 v4 的未命名候选片段，长度过长／没有有效内部变化／发声不足都必须弃判；
4. **推断完成后**，才使用此前冻结的 OTO 标签锚点做描述性后验对照；它不是准确的音素边界。没有任何新 alias、IPA 依赖、自动 split/merge，原始 ZIP 未修改。

### 本轮真实输出

| 项目 | 数量 |
|---|---:|
| 源 WAV | 175 |
| v4 全部内部变化事件 | 15,137 |
| 有可测量声学特征的事件 | 13,352 |
| 因缺少有效音频上下文弃判 | 1,785 |
| v4 原始候选区间 | 19,610 |
| 有**未经确认标签提示**的区间 | 12,197 |
| 保持未决的区间 | 7,413 |
| v3 用作开发标签原型的样本 | 5,816 |
| 事前冻结的 8 种留出源 WAV 模板的 OTO 参照行 | 2,044 |
| 参照点 ±100ms 内具有可测自主内部变化事件 | 1,281 / 2,044 = 62.67% |
| 最近事件标签与 OTO 左侧标签一致 | 491 / 2,044 = 24.02% |

上述百分比**不是**音素边界召回率／完整音素分割准确率，±100ms 是 OTO 中间位置的诊断范围，不是真实声学金标准。v3 知道 OTO 参考位置时对同样 2,044 行的标签识别率是 46.87%；v5 不再把这些时间位置当推断输入，因此显著变差。

### 关键负面发现

强迫每个自主声学变化点都映射到 `i/ir/i0/a/...` 这样一个用户标签会大量错配。有的变化是声调／音高／音色／辅音开始／能量改变，而不是一个可独立命名的单元出现。**“找到变化点”与“找到具体音素”不可等同**。这版未授权任何自动录音表拆分、合并或新标签生成。

当前最有针对性的下一步不是继续优化倒谱分类器：需要引入**用户给出的完整发音次序约束**以及**可跳过变化点的多对多序列对齐**，并以未参与调试的实际音素边界做验收。一个音素可能覆盖几个变化区域；一个候选区域也可能跨音素。不同语言依旧保留原始用户符号，不要求 IPA。

### 输出说明

- `results/event_unit_candidates.csv`：15,137 个 v4 内部事件的预测最相似用户标签、三项候选、距离 margin（**非概率**）、源 WAV PCM SHA-256；失败时弃判。
- `results/candidate_segment_unit_hints.csv`：19,610 个原始 v4 候选区间的建议标签及状态。每个标签都保留 `candidate_UNVERIFIED_not_a_phone` 标记。
- `results/heldout_posthoc_oto_diagnostics.csv`：仅用于诊断的 2,044 条参考位置对照；不用于推断或挑选候选点。
- `results/heldout_by_user_label.csv`、`heldout_by_template.csv`、`heldout_by_bank.csv`：失败模式分布。小样本行不可推断统计显著性。
- `results/summary.json`：总体结论。
- `source/phonoweave/event_unit_bridge.py`：可在 PhonoWeave 仓库中运行的 IPA 独立桥接模块。
- `source/tests/test_event_unit_bridge.py`：合成测试与来源保护。

### 实际运行命令

```bash
phonoweave link-user-units-to-events \
  --v3 /path/PhonoWeave_User_Native_Temporal_Context_v3.zip \
  --v4 /path/PhonoWeave_WholeAudio_Unanchored_Events_v4.zip \
  --manifest /path/PhonoWeave_User_Native_Batch_Discovery_v2.zip \
  --output /path/new_unused_folder
```

`--manifest` 使用仓库已有的 v2 私有银行配置，里面是原始 ZIP 在运行机器上的实际路径。原始声库 WAV **不上传 GitHub**。命令会核对 ZIP 和 PCM SHA-256，防止两版特征错配。输出目录已存在就拒绝覆盖。

说明：这是一个**用冻结 v3/v4 数据作输入的研究桥接器**，并非任意语言无训练数据就能使用的成品分类器。v4 自主事件检测语言无关，但用户原生标签识别需要带标签音频和可验证的发音顺序；自动发现新的内部符号仍须经重复性与合成验证。
