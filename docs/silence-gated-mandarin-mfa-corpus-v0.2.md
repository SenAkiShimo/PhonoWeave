# PhonoWeave · 全库普通话单音节对齐准备 v0.2

本轮目标是用**同一套模型**对齐完整中文声库，而不是分别给 `/sh/`、`/m/` 等音素编写独立规则。此包已在作者提供的真实 LynneQing 五个 ZIP、七个音阶子库上运行；生成的片段真实来自原始 WAV，短片段切点只能位于源音频中测得的安静间隔。**未运行 MFA 声学模型，也没有生成声学模型估计的音素边界。**

## 实测结果

- 原始整库：**175 个 WAV / 7 个音阶子库 / 25 种源 WAV 模板**。
- 原始 OTO 草稿中能推断音节顺序的 119 个源 WAV：完成静音门控分割，**2,821 条约 1.87–2.46 秒的单音节 WAV + `.lab`**，每个音阶 403 条；7 个音阶中的总片段数并不表示 2,821 次独立真人录音。
- `ong.wav` 中重复的 `rong` OTO 锚点相差 23.366ms，定位为同一近重复音节标记；每个音阶去掉**一条重复音节候选**（共 7 条），保留原始 OTO 记录不改动。该判断仍需通过原始 DiffSinger 合成工程最终核实。
- 切点要有至少 80ms 连续安静段；119 个输入文件全部满足；遇到声学不安静或冲突的音节会弃判，绝不强行把响声切开。
- 56 个缺少完整音节序列的 VC/其他原始 WAV 仍然保留待处理，不会用 alias 猜造完整语料。
- 抽查 25 段裁切和各自原始 ZIP 的 PCM 样本逐字节一致。切分不使用音高变换、去噪、模型生成或重采样。标签仍是 OTO 重建草稿，不能当作已核实歌词。
- **真正运行 MFA 的段数：0；已验证音素边界：0。** 目前执行容器不能解析外部模型站点，也没有 MFA 声学权重。这是可执行环境限制，不是模型已验证成功。

## 文件

- `phonoweave_silence_chunks.py`：兼容 Python 3.11+、NumPy 的 read-only 整库短片段准备器与来源时间还原命令。
- `whole_utterance_mfa_bridge.py`：已有 PR #74 的 MFA `doctor`、`align`、`import` 执行与回执工具。
- `preflight.zip`：完整源 WAV 草稿索引、源 ZIP SHA-256、17 份待核实文本模板；不包含作者的原始大容量音频。
- `result_manifest/`：本次真实运行结果的 CSV、来源哈希与质量门禁结果，零虚构边界。
- `example_audio/`：少量已实际裁切的源 PCM WAV 和单音节 `.lab`，仅供直观检查。
- `test_silence_chunks.py`：**8 项**合成测试（不是音素准确率测试）。

## 在能联网的 Mac 上一键准备 + MFA

把此前上传的五个原始声库 ZIP 放到同一个目录（可与此工具包分开）。文件名须和原始一致：

```
LynneQingD4.zip
LynneQingD5.zip
LynneQingG3.zip
lynneqing_G4.zip
LynneQing_CVVChn_Lu.zip
```

首先安装 Miniforge / conda，创建 MFA 环境（模型下载需在你自己的网络正常的电脑上进行）。之后也可直接运行 `bash run_mfa_on_mac.command /你的/声库ZIP目录 /你的/新工作目录`，完成全部五步，重复运行时保留既有结果不覆盖：

```bash
conda create -n phonoweave-mfa -c conda-forge python=3.11 montreal-forced-aligner numpy
conda activate phonoweave-mfa
mfa --version
```

在本工具包的文件夹内执行以下步骤。替换示例路径；输出目录必须**还不存在**，防止覆盖原始数据：

```bash
# 1. 用真实源 WAV 安全生成 2,821 个单音节对齐输入（试验标注，非金标准）
python phonoweave_silence_chunks.py \
  --preflight-zip preflight.zip \
  --voicebank-dir /你的/五个声库ZIP所在文件夹 \
  --output /你的/phonoweave_syllable_corpus \
  --allow-unverified-pilot

# 2. 检查本机 MFA；无模型时不会伪造任何 TextGrid
python whole_utterance_mfa_bridge.py doctor

# 3. 真正运行普通话预训练声学模型（会在可用网络下下载 HF 模型）
python whole_utterance_mfa_bridge.py align \
  --stage /你的/phonoweave_syllable_corpus \
  --output /你的/phonoweave_mfa_textgrids --mode hf

# 4. 只读取有真实模型成功执行回执的 TextGrid
python whole_utterance_mfa_bridge.py import \
  --stage /你的/phonoweave_syllable_corpus \
  --textgrids /你的/phonoweave_mfa_textgrids \
  --output /你的/phonoweave_mfa_imported

# 5. 把真正模型输出的单音节局部时间还原为原始整条 WAV 的绝对时间
python phonoweave_silence_chunks.py lift \
  --stage /你的/phonoweave_syllable_corpus \
  --imported /你的/phonoweave_mfa_imported \
  --output /你的/phonoweave_original_phone_intervals.csv
```

**重要：** 若模型下载失败，请检查你的网络/MFA 版本，不要把 `result_manifest/clips.csv` 的起止时间当作音素边界。它们只是隔开音节的静音裁剪位置。上面 `align` 使用 MFA >=3.4 的 Hugging Face `mandarin_mfa` 模型，并为 Pinyin 启用 G2P 与 `--language unknown`；安装和运行选项以实际 MFA 版本为准。

## 闭环与下一关

短片段的取舍使通用 MFA 对齐器处理单个 `ma` 等音节而非 60–80 秒长的连续练声录音。MFA 的输出依然是**未经过人工证实的音素候选**，且其模型针对普通话语音，不保证适用 DiffSinger 歌声。需抽样检查人工边界，测量误差，再考虑接入 speaker profile 与录制方案。`/sh/` 等别名原样保留，不擅自转成 IPA 或修改 OTO。

后续若有原 DiffSinger 工程的音素持续时长/声学标签，它能比 OTO 草稿更好地确认各句文本和边界，可用于训练或独立验证，但不是运行本次对齐试验的前置条件。
