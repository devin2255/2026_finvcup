# 音频分支重构设计：双声道 + 时序保留（方案 A）

- 日期：2026-06-13
- 分支：phase3-bc-i-optimization
- 状态：已评审通过，待生成实施计划
- 当前提交分数（macro-F1）：0.736767

---

## 1. 背景与目标

赛题为多方言中文对话的**话轮交互建模（Turn-Taking）**：给定过去 ≤30s 的双声道对话音频 + ASR 文本 + 历史 chunk 标签，预测未来 2s（25×80ms chunk）内是否出现 5 类事件（C/T/BC/I/NA）。提交为 event-level 多标签（每段 5 个 0/1 列），评测指标为 **macro-F1（逐标签搜最优阈值，`macro_best_f1`）**，见 `src/utils.py:101` `compute_multilabel_metrics`。

**目标**：在不改任务形式、不换更大模型的前提下，修复当前音频分支丢失关键信号的两个结构性问题，预期提升 macro-F1。本设计**不**通过堆参数量解决问题——当前模型仅用约 1.2B 参数（Whisper-large-v3 encoder + Qwen3-0.6B，均大体冻结），远低于官方 8B 上限，参数量不是瓶颈。

---

## 2. 现状分析

### 2.1 当前架构（`src/models/multimodal_baseline.py`）

四模态 → 低秩张量融合（LMF）→ 5 标签 sigmoid 头：

- 音频：`WhisperAudioEncoder`，Whisper-large-v3 encoder（冻结，仅解冻末 2 层）
- 文本：`TextEncoder`，Qwen3-0.6B（全冻结）
- 上下文标签：`ContextLabelEncoder`（C/T/BC/I/NA 历史序列）
- 手工特征：`HandcraftedFeatures`（从上下文标签算转移/衰减/间隔等统计量）

上下文标签在训练与测试均可得（测试集提供 `context/<id>.npy`），已编码"谁在何时持有话权"。

### 2.2 三个结构性问题（本次要修）

1. **双声道被 mono 平均**：`WhisperAudioEncoder._build_input_features` 在 `multimodal_baseline.py:150` 执行 `mono = wave.mean(dim=1)`，把两个说话人声道平均成单声道。话轮预测的核心是"A 还是 B 在说话"，mono 后 Whisper 无法把声学内容归属到具体说话人。数据本身是双声道（`dataset.py:204-208`，强制 2 通道）。

2. **时序被池化成单点**：`WhisperAudioEncoder.forward`（`multimodal_baseline.py:159-174`）对尾部帧做注意力池化压成一个 512 向量。决定 T/BC/I 的恰是预测边界前 1~2s 的精细韵律/重叠/尾音，单点池化抹平了时序分辨率。

3. **动态上下文下"看 padding"（隐藏 bug）**：Whisper feature extractor 把音频补零到固定 30s（内容在前、补零在后），而代码取 `hidden[:, -tail:]`（末端）。满 30s（公榜）正常；但私榜上下文是动态长度 (0,30]s，训练里也有 50% 概率走动态长度（`dataset.py:242`）。此时"尾部"取到的是**末尾补零帧**，真实的预测边界反而落在序列中间——音频分支在动态样本上基本失效。

> 经验佐证：最近一次 0.7285→0.7368 的提升全部来自 stride/loss/特征工程，没有一项来自模型规模。信号瓶颈在"信息丢失"，不在参数量。

---

## 3. 方案选择

**约束**：训练硬件为 2×48GB L20（DDP，约 96GB 显存），显存不是约束；当前 `batch_size=1`、`grad_accum=8` 属保守设置，有充足上抬空间。

考虑过的方案：

- **方案 A（选定）**：每声道 Whisper（共享权重）+ 带掩码的时序尾部头。最干净地同时修复 2.2 的三个问题，融合层及以下不动。
- 方案 B（最小改动）：mono Whisper 时序头 + 廉价逐声道能量/VAD 活动分支。在显存富裕时其"省算力"优势无意义，且 Whisper 仍不区分说话人。
- 方案 C（信号最全）：A + 廉价逐声道活动分支。作为 A 增益遇瓶颈后的可加性扩展（活动分支不推翻 A），不进首版。

**选 A 的理由**：2×L20 让"每声道 Whisper"从奢侈变成默认更优选项；它一次性命中"修双声道 + 保留时序"，并顺手关闭 padding bug；改动收敛在音频分支内部，回归面小。

---

## 4. 架构设计（数据流）

改动只收敛在音频分支内部；LMF 融合、文本/上下文/手工特征分支、损失、阈值搜索全部不变。

```
波形 [B,2,T]
  └─(改) 前置补零到固定 30s，边界对齐到序列末端；输出 audio_valid_samples
  └─(改) 按声道切分 ch0 / ch1（不再 mono 平均）
        └─ Whisper 编码器（共享权重，解冻末2层）：一次前向跑 [2B,128,3000] → [2B,1500,1280]
        └─ split 回 ch0/ch1 隐状态 [B,1500,1280] ×2
  └─(改) 边界对齐尾部 + 掩码：取末端 K 帧，按真实长度右对齐 mask → [B,K,1280] ×2
  └─(改) 逐帧拼接两声道 [B,K,2560] → 线性投影 [B,K,512]
  └─(改) 时序头（BiGRU/Transformer + mask）→ 末帧 + 注意力池化 → 音频向量 [B,512]
  └─ 进入 LMF 多模态融合（不变），与 text/context/hand 四模态低秩交互
  └─ 5 标签事件头 → C/T/BC/I/NA（sigmoid）
```

音频分支输出 `[B,512]`，**接口与原 `audio_encoder.out_dim` 一致**，无缝喂回现有 `MultimodalFusion`。

---

## 5. 组件细节

### 5.1 边界对齐 + 掩码（核心，且修 2.2-③）

在 dataset 把波形**前置补零**到固定 30s（`context_chunks×chunk_ms = 375×80ms = 480000 samples @16kHz`），并多输出 `audio_valid_samples`：

```python
# _load_wave_segment 末尾：补在前面，目标长度固定
target = context_chunks * chunk_ms * sample_rate // 1000      # 480000
content = wave.shape[1]
if content < target:
    wave = F.pad(wave, (target - content, 0))                 # 前置补零
else:
    wave = wave[:, -target:]                                  # 取最近 30s
audio_valid_samples = min(content, target)
```

内容永远落在末端 → Whisper 输出 `[B,1500,1280]` 的最后一帧即最近时刻，取 `hidden[:, -K:]` 天然对齐预测边界。掩码按真实长度右对齐：

```python
HOP = 320                                   # Whisper encoder 帧 = 20ms = 320 samples @16kHz
valid_frames = round(audio_valid_samples / HOP)
valid_frames = clamp(valid_frames, 1, 1500)
n_valid_tail = min(K, valid_frames)
# 末端 K 帧中，最后 n_valid_tail 个为有效，其余（前置 pad）置 0
tail_mask[:, K - n_valid_tail : K] = 1
```

动态/短上下文样本不再把补零当信号 → 2.2-③ 关闭。

> Whisper 帧率推导：feature extractor `hop_length=160`（10ms）→ 30s 得 3000 帧；encoder 第二层 conv `stride=2` → 1500 帧 → 20ms/帧 → 320 samples/帧。掩码对舍入不敏感。

### 5.2 每声道 Whisper 前向（一次前向跑 2B）

```python
# wave: [B, 2, T]，已前置补零到 30s
arrays = [wave[b, 0].cpu().numpy() for b in range(B)] + \
         [wave[b, 1].cpu().numpy() for b in range(B)]
feats = feature_extractor(arrays, sampling_rate=sr, return_tensors="pt")["input_features"]  # [2B,128,3000]
hidden = encoder(input_features=feats.to(device)).last_hidden_state                         # [2B,1500,1280]
h0, h1 = hidden[:B], hidden[B:]
tail = torch.cat([h0[:, -K:, :], h1[:, -K:, :]], dim=-1)    # [B,K,2560]
x = proj_in(tail)                                           # Linear(2560 → 512) → [B,K,512]  (512 = proj_dim)
```

feature extraction 仍在 fp32（`autocast` 关闭，沿用现状）。冻结主干在 `no_grad` 下前向、仅末 2 层带梯度，沿用 `encoder_has_trainable_layers` 逻辑。

### 5.3 时序头 + 汇聚

- **默认 BiGRU**：`input_size=512`、1 层、`hidden_dim=256`、双向（输出 512），带 mask；汇聚取「边界帧 `out[:, -1, :]`（512）+ 掩码注意力池化（复用 `AttentionPooling`，传 mask）（512）」拼接（512+512=1024）→ `Linear(1024 → proj_dim=512)`。
- 可配 **Transformer**：输入 `d_model=512`、2 层、8 头、learned positional encoding、`src_key_padding_mask=~tail_mask`；汇聚方式同上（1024 → 512）。
- 选 GRU 当默认：尾部序列短，循环天然把"边界前历史"压进末帧，鲁棒、参数少、无需位置编码。

边界帧恒为 index `K-1`（只要 `valid_frames≥1` 即有效），故"末帧"= `head_out[:, -1, :]`，简单可靠。

---

## 6. 接口改动

| 文件 | 改动 |
|------|------|
| `src/data/dataset.py` | `_load_wave_segment` 改前置补零到固定 target；`TurnTakingTrainDataset.__getitem__` 与 `TurnTakingTestDataset.__getitem__` 输出 `audio_valid_samples`（标量张量）；`CollateFn.__call__` 在 train/test 两支都把 `audio_valid_samples` stack 成 `[B]` 放入 batch。所有波形定长后，collate 内 per-batch 波形 padding 变为 no-op（保留以防御）。 |
| `src/models/multimodal_baseline.py` | `WhisperAudioEncoder.forward(self, wave, valid_samples=None)` 实现 5.1–5.3；新增子模块 `proj_in` 与时序头；`MultimodalTurnTakingModel.forward(..., audio_valid_samples=None)` 透传给音频编码器；CNN 路径 `AudioEncoder.forward` 忽略 `valid_samples`（签名兼容）。 |
| `src/train.py` | `model(...)` 调用处补 `audio_valid_samples=batch["audio_valid_samples"]`。 |
| `src/infer_test.py` | 同上，推理调用处补传 `audio_valid_samples`。 |

`valid_samples=None` 时按"全有效"处理（向后兼容 / CNN 路径）。

---

## 7. 配置 schema（新增，均可开关以便消融）

```yaml
audio_encoder:
  type: whisper
  model_name: openai/whisper-large-v3
  proj_dim: 512
  freeze: true
  unfreeze_layers: 2
  dual_channel: true          # 新增：false 退回 mono（保留旧路径做对照）
  tail_frames: 400            # 新增：保留末端 K 帧(~8s)，替代 tail_ratio
  temporal_head:              # 新增
    type: gru                 # gru | transformer
    hidden_dim: 256
    num_layers: 1
    dropout: 0.1
```

- 未提供 `tail_frames` 时回退 `tail_ratio`（向后兼容）。
- `dual_channel` 与 `temporal_head` 独立开关，支持「mono vs 双声道」「单点池化 vs 时序头」A/B 消融。
- 音频定长由 `context_chunks×chunk_ms` 推导（30s），可选 `audio_len_ms` 覆盖。

---

## 8. 训练 / 推理触点

- EMA、AMP、checkpoint、early-stop 均不受影响；Whisper feature extraction 维持 fp32。
- 架构已变 → **必须重训，旧 checkpoint 不复用**（预期内）。
- 显存富裕，建议重训时把 `batch_size` 从 1 上抬（顺带稳住 focal/pos_weight 的梯度估计）——属调参，不写进架构核心，但在新配置里给出推荐值。

---

## 9. 测试策略（TDD，先红后绿，不依赖下载大模型）

**纯逻辑单测**
1. 前置补零：短波形内容落在末端、长度=target、`audio_valid_samples` 正确；超长则取最近 30s。
2. 尾部掩码：满 30s / 10s / 1s / <1 帧 各情形，断言右对齐、有效计数 `min(K,valid_frames)` 正确、边界帧有效。
3. 时序头不变性：改动被 mask 掉的 pad 区，音频向量输出不变（证明掩码生效）。
4. collate：batch 含 `audio_valid_samples` 且形状 `[B]`（train/test 两支）。

**关键回归测试（直接证明 2.2-③ 修好）**
- 同一满-30s 样本 vs 人为加前置补零并相应设置 `valid_samples`，两者音频向量应**完全一致**（边界对齐 + 掩码 → 对前置 pad 不变）。

**集成 smoke 测试**
- 用 **stub 编码器**（返回 `[2B,1500,1280]` 随机张量，暴露 `.config.d_model`）替换 Whisper，跑通 `MultimodalTurnTakingModel.forward` → 断言 logits `[B,5]` 且可反向，避免拉取 whisper-large-v3。

---

## 10. 风险与取舍

- 电话每声道存在对方串音/回声 → 仍比 mono 信息多，模型可学，低风险。
- CPU 端 feature extraction 变 2× → 先观察 dataloader 吞吐；若成瓶颈，再上"log-mel 磁盘缓存"或 GPU 端 mel（列为可选优化，不进首版）。
- 时序头类型与 `tail_frames` 取值可经配置消融，不锁死。

---

## 11. 不在本次范围（YAGNI / 后续）

- 文本分支升级（Qwen3-0.6B→4B）：文本是弱信号且冻结，低 ROI，暂不做。
- VAP/MaAI 话轮先验特征接入（赛题清单含 `vap_mc_ch_kyoto`/`vap_bc_ch`）：列为后续"②/③"方向。
- 改为 per-chunk 序列预测再聚合：改任务形式，属更大重构，另开 spec。
- 廉价逐声道活动分支（方案 C 增量）：A 遇瓶颈后再加。
- mel 特征缓存优化：仅在 CPU 预处理成瓶颈时启用。

---

## 12. 验收标准

- 全部单测 + 回归测试 + 集成 smoke 测试通过。
- 满-30s 公榜样本上，新音频向量对"额外前置补零"严格不变（回归测试绿）。
- 端到端训练可跑通（短 epoch 冒烟），logits 形状 `[B,5]`，可反向、可保存/加载 checkpoint。
- 在验证集上对比 `dual_channel=false`（旧路径）与 `true`（新路径）的 `macro_best_f1`，新路径不劣于旧路径；提交公榜验证相对 0.736767 的变化。

---

## 13. 保持不变的契约

- 输出仍为 event-level 5 标签（C/T/BC/I/NA），匹配 `pred_test1.csv` 提交格式。
- `MultimodalFusion`、文本分支、上下文标签分支、手工特征、损失（`MultiLabelFocalLoss`）、阈值搜索全部不动。
- `audio_encoder.type: cnn` 旧路径保留，仅改 `whisper` 路径。
- 音频分支对外仍输出单一 `[B, proj_dim]` 向量。
