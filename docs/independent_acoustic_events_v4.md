# PhonoWeave v4 — OTO 无关的连续 WAV 声学变化点及候选分段

## 实际完成的工作

读取作者提供的 LynneQing 七个音阶（芒 G3/D4/G4/D5、露 C4/F4/B4）的全部 **175 条完整原始 WAV**，时长总计约 **9,815.043 秒**；完全不用 OTO、IPA、MFA 或用户转写来 *寻找* 声学变化位置。

- **15,137** 个持续发声内部的频谱变化候选（`internal_acoustic_change`）。
- **8,881** 个活动／非活动变化边缘（`activity_onset`、`activity_offset`）。
- **19,610** 个内部变化点与音频活动区间构成的未命名候选声学区间（`unclassified_acoustic_region`）。
- 175 个可在 Praat 打开的 `.TextGrid`，每个含独立事件点层和未命名区间层；提供精确到源 WAV 时间轴的 CSV，以及每条源 PCM／ZIP 的 SHA-256。

**所有这些数都不代表“已正确切出 N 个音素”。** 一个语言学音素可能产生多个声学事件，也可能没有强变化点。持续发声区间之间的静音不是音素边界。

## 算法（完全语言无关）

1. 原始 PCM16 只读，内部计算统一到 16 kHz；原声音频没有被覆盖。
2. 25 ms Hann STFT／10 ms hop，24-band log-mel 谱形消除总音量影响。
3. 比较时间点两侧各 40 ms 的局部谱形，得到无标签声学变化强度。
4. 通过**频带能量求和**（而非跨频带平均能量）测量活动区间。这项修正避免低估无声擦音中的高频能量。
5. 持续发声两侧均有活动、峰值强度足够且局部极大的时间点给出 `internal_acoustic_change`；活动起止另列，不伪装成内部音素变化。
6. 在活动区间内以内部峰值划分候选时间段；区间一律标为 *unclassified*；输出 TextGrid／CSV，不自动生成 `i1` 等别名，不改 OTO。

音频检测 `scan_archive` **没有任何读取 OTO 的代码路径**，可以处理 ZIP 里只有 WAV、完全没有 OTO 的情况。文档中的六条人工样例和 OTO 中点只由另一个 `acoustic_event_audit` 诊断命令**在分析完成之后**读取。

## 后验诊断，非盲测

本轮 D4 检测器曾因错误的音频能量定义而被修正；修正时检查过你之前人工复核的六条 D4 过渡。因此这六条是**开发样本**，不是独立测试集。修正后的固定版本对照这六段：

- 6 段中有 **4 段**的内部变化点位于人工过渡区间内部；另 **1 段**距离人工区间 **0.8 ms**；另 **1 段**明显漏检，偏离 **96.2 ms**。
- 纯静音活动起止的 8,881 个边缘中，D4 对应的六个过渡区间没有任何一个包含活动边缘（0/6）。这证实“静音切段”和“持续声音内的变化点”不是一回事。
- 这些人工标记本来是**过渡区间**，不是单一金标准音素边界；不能由重叠情况直接推断 phone boundary MAE。

利用相同 WAV 模板跨音阶对比（不是独立真人录音），D4 内部事件在其他音阶同名源 WAV 的 ±30 ms 内找到相应事件的比例为 **64.26%–73.75%**。把另一音阶事件人为整体错开 500 ms 的负控约 **9.40%–14.49%**。说明在 DiffSinger 同模板渲染中检测有重复性，但这些结果不能用于声学拆分／合并效益论证。

**独立核验过的完整音素起止边界：仍然是零。** 必须通过保留的、与调试不重合的新人工音素边界集才可能评估其误差。不能用用户 alias 标签分类准确率代替切分评估。

## 正式运行方式

```bash
phonoweave propose-acoustic-events \
  --archive /path/to/voicebank.zip \
  --root SomeBank/ \
  --output /path/to/NEW_EMPTY_OUTPUT_FOLDER
```

这个 CLI 不要求普通话拼音词典和 IPA。`--root` 必须与 ZIP 内源 WAV 文件夹完全匹配；输出目录必须尚不存在。完整运行可用于任意语言的 PCM16 单声道 WAV ZIP；是否适合不同语音／歌声，需要另行验证。

后验诊断（仅在持有真实人工标注时）：

```bash
python -m phonoweave.acoustic_event_audit \
  --results-root /path/to/seven_banks_scanned \
  --review-zip /path/to/review_package.zip \
  --reviewed-csv /path/to/author_annotations.csv \
  --output /path/to/new_audit_folder
```

`events.csv` 是变化点表，`candidate_segments.csv` 是可进一步处理、但没有音素身份的候选区间，`sources.csv` 提供原始 SHA-256，`textgrids` 可用 Praat 直接检查。

## 下一道真正的研究门槛

在**冻结这个检测器**之后，建立未被用于调试的普通话／其他语言音素边界参考，报告有多少已确认边界在 ±20／±40 ms 内找到了变化点，同时报告平均每秒误检候选数和漏检类别。若实际边界无法由声学变化点唯一判定，采用联合序列对齐／自监督模型，而不是强行把事件点直接解释为音素。
