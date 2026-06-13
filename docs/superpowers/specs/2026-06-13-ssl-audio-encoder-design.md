# SSL 音频编码器设计：HuBERT/WavLM/wav2vec2（方向 ②）

- 日期：2026-06-13
- 分支：phase3-bc-i-optimization
- 状态：已评审通过，待生成实施计划
- 前置：建立在「方案 A 双声道 + 时序」之上（spec：[2026-06-13-audio-branch-dualchannel-temporal-design.md](2026-06-13-audio-branch-dualchannel-temporal-design.md)）

---

## 1. 背景与目标

①（双声道 + 时序，仍用 Whisper-large-v3）训练慢：`nvidia-smi` 实测 GPU-Util 70%/95%、功耗未顶、显存仅 14GB/49GB。两个原因：(a) 30s 全长 Whisper-large 编码器 ×双声道 = 每步 32 次重前向；(b) `WhisperFeatureExtractor` 的 mel 在 **CPU 同步** 跑、`GPU→CPU→GPU` 来回搬，制造气泡（`num_workers` 帮不上，因为它在 forward 里不在 DataLoader）。

**目标**：把音频编码器换成自监督语音模型（默认 **`TencentGameMate/chinese-hubert-large`**），同时解决速度与对口性：

- **更快**：HuBERT/WavLM/wav2vec2-large ≈ 316M（Whisper-large-v3 编码器 ~635M 的一半）；且**直接吃原始波形**，归一化是 GPU 上的 mean/std，**没有 CPU mel STFT** → 速度瓶颈消失。预期 ~2–2.5×。
- **更对口**：话轮预测靠韵律/停顿/说话人切换，自监督语音模型在这些上通常强于 ASR 导向的 Whisper；中文预训练（HuBERT/wav2vec2）声学更匹配方言数据。

不通过堆参数解决问题——总参数仍远低于官方 8B 上限。

---

## 2. 现状与可复用资产

- ① 的音频分支结构：`waveform [B,2,T]` → 每声道编码 → `gather_boundary_tail`（边界对齐尾部+掩码）→ `TemporalHead`（BiGRU/Transformer）→ `[B, proj_dim]` → LMF 融合。
- **关键：`gather_boundary_tail` 与 `TemporalHead`（[src/models/audio_temporal.py](../../../src/models/audio_temporal.py)）与编码器无关**——只吃 `hidden [2B, n_frames, D]`。换编码器只换"前端"，整条尾部/掩码/时序/融合链路原样复用。
- dataset 已输出 `audio_valid_samples`、波形已前置补零到 30s（边界在序列末端）。
- 编码器分支入口在 [multimodal_baseline.py:551](../../../src/models/multimodal_baseline.py)：`audio_type = cfg["audio_encoder"].get("type","cnn")`，现有 `whisper` / `cnn` 两支。`AutoModel` 已在文件内导入（文本编码器在用）。

---

## 3. 方案选择

**选定：通用 `SSLAudioEncoder`（走 `AutoModel`，可配置）。**
- HuBERT/WavLM/wav2vec2-large 三家共用同一 HF 接口：输入 `input_values`（原始波形）、`attention_mask`，输出 `last_hidden_state [B,T,1024]`，帧率 50Hz（20ms/帧，与 Whisper 一致）。换 `model_name` 即可 A/B，无需多写类。
- 不选「每模型一个类」（无谓重复）；不选「用 HF `Wav2Vec2FeatureExtractor` 做归一化」（又回 CPU，违背提速初衷）。

---

## 4. 架构设计（数据流）

改动仍只在音频分支前端；融合层及以下不变。

```
波形 [B,2,T]（已前置补零到 30s）
  └─ dual_channel=true → ch0/ch1；false → 两路都用均值混音（mono 消融，复用同一流水线）
  └─ 拼成 [2B, T]
  └─(SSL 专属) 构造末端对齐 attention_mask [2B,T]（前置补零置 0）
  └─(SSL 专属) GPU 上 per-utterance 零均值单位方差归一化（仅有效区间）—— 取代 CPU mel
  └─ AutoModel backbone（CNN 前端恒冻结，encoder 末 N 层可训）：input_values+attention_mask → [2B, n_frames, 1024]
  └─(复用①) gather_boundary_tail → 边界对齐尾部 [B,K,2048] + lengths + mask
  └─(复用①) TemporalHead（BiGRU/Transformer）→ 音频向量 [B, proj_dim]
  └─ LMF 融合（不变）+ text/context/hand → 5 标签事件头
```

---

## 5. 组件细节

### 5.1 `SSLAudioEncoder`（新增，置于 `multimodal_baseline.py`，与 `WhisperAudioEncoder` 并列）

```python
class SSLAudioEncoder(nn.Module):
    def __init__(self, model_name, sample_rate, proj_dim, freeze=True,
                 unfreeze_layers=0, dual_channel=True, tail_frames=400,
                 audio_len_samples=480000, temporal_head_cfg=None):
        self.backbone = AutoModel.from_pretrained(model_name)   # Hubert/WavLM/Wav2Vec2 Model
        # 冻结整个主干；CNN 特征前端恒冻结；解冻 encoder.layers 末 N 层
        if freeze: 全部 requires_grad=False
        self.backbone.feature_extractor 保持冻结（low-level 前端，标准做法）
        if unfreeze_layers>0: 解冻 self.backbone.encoder.layers[-N:]
        self.encoder_has_trainable_layers = any(p.requires_grad ...)
        hidden_size = self.backbone.config.hidden_size           # 1024
        self.dual_channel, self.tail_frames, self.audio_len_samples = ...
        cfg = temporal_head_cfg or {}
        self.temporal_head = TemporalHead(in_dim=2*hidden_size, model_dim=proj_dim,
                                          out_dim=proj_dim, head_type=cfg.get("type","gru"),
                                          hidden_dim=cfg.get("hidden_dim",256),
                                          num_layers=cfg.get("num_layers",1),
                                          dropout=cfg.get("dropout",0.1))
        self.out_dim = proj_dim
```

### 5.2 GPU 归一化（取代 CPU mel —— 提速核心）

自监督语音模型要 per-utterance 零均值单位方差的原始波形（大模型 `do_normalize=True`）。在 GPU 上对**有效区间**（前置补零不计）计算：

```python
def _normalize(self, x, attn):          # x:[2B,T], attn:[2B,T] (1=有效)
    m = attn.float()
    cnt = m.sum(1, keepdim=True).clamp_min(1.0)
    mean = (x * m).sum(1, keepdim=True) / cnt
    var = (((x - mean) * m) ** 2).sum(1, keepdim=True) / cnt
    return (x - mean) / torch.sqrt(var + 1e-7) * m     # 补零区间保持为 0
```

### 5.3 attention_mask（前置补零→末端对齐）

```python
def _end_aligned_mask(self, T, valid_samples_2b, device):   # → [2B,T] bool
    ar = torch.arange(T, device=device).unsqueeze(0)
    return ar >= (T - valid_samples_2b.clamp(1, T).unsqueeze(1))
```

传给 backbone：`self.backbone(input_values=x, attention_mask=mask)`。large 变体（`feat_extract_norm="layer"`，chinese-hubert-large / wavlm-large / chinese-wav2vec2-large 均是）会正确使用 mask，让 transformer 忽略前置补零。

### 5.4 forward（单一路径；mono 仅改输入构造）

```python
def forward(self, wave, valid_samples=None):
    B, _, T = wave.shape
    if self.dual_channel:
        ch0, ch1 = wave[:, 0, :], wave[:, 1, :]
    else:
        mono = wave.mean(dim=1); ch0 = ch1 = mono       # mono 消融：复用同一 dual 流水线
    x = torch.cat([ch0, ch1], dim=0)                     # [2B, T]
    if valid_samples is None:
        valid_samples = torch.full((B,), self.audio_len_samples, dtype=torch.long, device=wave.device)
    vs2 = torch.cat([valid_samples, valid_samples], dim=0)
    attn = self._end_aligned_mask(T, vs2, wave.device)   # [2B,T]
    x = self._normalize(x, attn)
    if self.freeze and not self.encoder_has_trainable_layers:
        with torch.no_grad():
            hidden = self.backbone(input_values=x, attention_mask=attn).last_hidden_state
    else:
        hidden = self.backbone(input_values=x, attention_mask=attn).last_hidden_state   # [2B,n_frames,1024]
    tail, lengths, mask = gather_boundary_tail(hidden, B, valid_samples, self.tail_frames, self.audio_len_samples)
    return self.temporal_head(tail, lengths, mask)
```

> 帧率：16kHz conv 总步长 320 → 50Hz；30s≈1499–1500 帧。`gather_boundary_tail` 用 `hidden.shape[1]` 当 `n_frames`、按 `valid_samples/audio_len_samples` 比例算有效帧，对 ±1 帧不敏感，无需改动。

---

## 6. 接口改动

| 文件 | 改动 |
|------|------|
| `src/models/multimodal_baseline.py` | 新增 `SSLAudioEncoder`；`MultimodalTurnTakingModel.__init__` 加 `elif audio_type == "ssl":` 分支（传 `model_name/proj_dim/freeze/unfreeze_layers/dual_channel/tail_frames/audio_len_samples/temporal_head_cfg`）；`forward` 把 `isinstance(self.audio_encoder, WhisperAudioEncoder)` 改为 **`not isinstance(self.audio_encoder, AudioEncoder)`**（仅 CNN 不传 `valid_samples`，Whisper/SSL 统一传） |
| `configs/` | 新增 `whisper_qwen0_6b_ssl_hubert.yaml`（`type: ssl` + `model_name: TencentGameMate/chinese-hubert-large`） |

无需改 dataset / train / infer / tune_threshold —— 它们已透传 `audio_valid_samples`。

---

## 7. 配置 schema（新增）

```yaml
audio_encoder:
  type: ssl
  model_name: TencentGameMate/chinese-hubert-large   # 换 microsoft/wavlm-large 或 chinese-wav2vec2-large 即 A/B
  proj_dim: 512
  freeze: true
  unfreeze_layers: 2
  dual_channel: true
  tail_frames: 400
  temporal_head:
    type: gru
    hidden_dim: 256
    num_layers: 1
    dropout: 0.1
```

其余字段（数据/标签/融合/损失/训练）沿用 `whisper_qwen0_6b_dualchannel_temporal.yaml`（含已调好的 `batch_size: 16` 等）。

---

## 8. 复用不变的契约

- `gather_boundary_tail`、`TemporalHead`、`AttentionPooling`、dataset 的 `audio_valid_samples`、`MultimodalFusion`、`MultiLabelFocalLoss`、阈值搜索、5 标签事件头 —— 全部不动。
- Whisper（`type: whisper`）与 CNN（`type: cnn`）路径保留 → 支持 Whisper vs HuBERT vs WavLM 的 A/B。
- 音频分支对外仍输出单一 `[B, proj_dim]` 向量。

---

## 9. 测试策略（TDD，先红后绿，不下载大模型）

**单测（桩 `AutoModel`，避免拉 chinese-hubert-large）**
1. `_end_aligned_mask`：前置补零→有效位在末端、计数=`min(valid_samples,T)`、边界位有效。
2. `_normalize`：有效区间零均值单位方差；补零区间输出严格为 0；对前置补零内容不变（不污染统计量）。
3. dual forward：桩 backbone 返回 `[2B,1500,1024]`（带 `config.hidden_size`、`encoder.layers`、`feature_extractor`），断言 `SSLAudioEncoder(wave, valid_samples)` → `[B, proj_dim]`，`out_dim==proj_dim`，且 `valid_samples=None` 可跑。
4. `dual_channel=false`：两路用均值混音，仍 `[B, proj_dim]`。

**集成路由测试**
5. `MultimodalTurnTakingModel.forward` 的 isinstance 改动：CNN 路径不收 `valid_samples`、Whisper/SSL 路径收 —— 用桩编码器断言三类 `type` 都能前向出 `[B,5]`。

---

## 10. 风险与取舍

- **归一化精确匹配**：不像 Whisper mel 那样敏感（SSL 主干较鲁棒，且我们还微调末 N 层 + 时序头）；按 HF `zero_mean_unit_var_norm` 复刻即可。
- **attention_mask 使用**：仅对 large 变体（`feat_extract_norm="layer"`）确定生效；本设计三个候选均满足。若换到 base 变体需复核。
- **mono 回退用 2× 计算**：`dual_channel=false` 复用 dual 流水线（两路混音相同），浪费一倍算力，但零额外代码、消融更干净；仅消融时用，可接受。
- **首次下载**：chinese-hubert-large ~1.2GB，走 `HF_ENDPOINT=hf-mirror.com`。
- **窗口仍 30s**：②的提速来自"无 mel + 半参数"；"30s→12s 窗口"是正交后续杠杆。

---

## 11. 不在本次范围

- 音频窗口缩短（30s→~12s）：正交的进一步提速，另做。
- WavLM/wav2vec2 的实际 A/B 训练：靠换 `model_name`，属运行而非编码。
- 文本分支升级、VAP 先验等：更早 spec 已列后续。

---

## 12. 验收标准

- 全部单测 + 路由测试通过。
- 端到端训练可跑通（短 epoch 冒烟），logits `[B,5]`，可反向/存取 checkpoint。
- 训练速度较 ①（Whisper 双声道）明显改善（目标 ~2×，以 s/it 与 GPU-Util 为准）。
- 验证集 `macro_best_f1`：HuBERT 路径与 Whisper 路径 A/B 对比；并与公榜 0.736767 对比。

---

## 13. 保持不变的契约

- 输出仍为 event-level 5 标签（C/T/BC/I/NA），匹配 `pred_test1.csv` 提交格式。
- 训练/推理/阈值脚本的调用契约不变（已透传 `audio_valid_samples`）。
- `type: whisper` / `type: cnn` 旧路径保留。
