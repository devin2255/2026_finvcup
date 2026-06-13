# Phase 1 + Phase 2 优化实施说明

## 📋 已实施的优化

### Phase 1: 立即见效优化 (配置文件级别)

#### 1. 数据增强
- ✅ **stride: 2** (从5改为2) - 样本密度提升2.5倍
- ✅ **动态上下文长度训练** - 适应私榜测试集的动态长度
  - `min_context_chunks: 125` (10秒)
  - `max_context_chunks: 375` (30秒)
  - `context_prob: 0.5` (50%概率使用动态长度)

#### 2. 损失函数优化
- ✅ **focal_gamma: 2.0** (从1.0提升) - 更关注难样本
- ✅ **pos_weight_cap: 8.0** (从5.0提升) - 更重视少数类
- ✅ **label_smoothing: 0.05** (新增) - 防止过拟合

#### 3. 训练策略
- ✅ **learning_rate: 3.0e-5** (从5e-5降低) - 更稳定的训练
- ✅ **warmup_ratio: 0.05** (从0.01提升) - 更平滑的启动
- ✅ **ema_decay: 0.995** (从0.98提升) - 更好的模型平滑
- ✅ **eval_valid_sample_count: 8000** (从4000提升) - 更准确的阈值

#### 4. 模型微调
- ✅ **unfreeze_layers: 2** (从1提升) - Whisper解冻更多层
- ✅ **tail_ratio: 0.25** (从0.2提升) - 更关注尾部音频

---

### Phase 2: 特征工程增强

#### 1. Context Encoder 增强
```yaml
context_encoder:
  embed_dim: 24        # 从16增加到24
  channels: [48, 96]   # 从[32,64]增加到[48,96]
  tail_k: 75           # 从50增加到75
```

#### 2. Fusion 增强
```yaml
fusion:
  hidden_dim: 320      # 从256增加到320
  bilinear_rank: 64    # 从48增加到64
  dropout: 0.25        # 从0.2增加到0.25
```

#### 3. HandcraftedFeatures 扩展
新增特征维度从 `19` 增加到 `64` (投影后):

**原有特征 (19维):**
- 标签分布 (tail25/50/100): 15维
- 距离特征: 1维
- 最近标签: 3维

**新增特征 (45维):**
- ✅ **转移模式** (5维): 最近50个chunk的标签转移概率
- ✅ **时间衰减分布** (10维): 近期(指数衰减) + 中期(线性衰减)
- ✅ **统计特征** (5维): 均值/标准差/最大值/最小值/变化率
- ✅ **话轮间隔** (3维): 距离最近话权转移/转移频率/平均话轮长度

---

## 🚀 使用方法

### 1. 训练优化模型

```bash
# Windows (PowerShell)
python -m src.train --config configs/whisper_qwen0_6b_lmf_8g_optimized.yaml

# Linux
bash scripts/run_train_optimized.sh
```

### 2. 对比基线模型

训练完成后，你会得到两个模型：

| 模型 | 配置文件 | 输出目录 | 预期分数 |
|------|---------|---------|---------|
| 基线模型 | `whisper_qwen0_6b_lmf_8g.yaml` | `outputs/lmf_8g/` | 0.728 |
| 优化模型 | `whisper_qwen0_6b_lmf_8g_optimized.yaml` | `outputs/lmf_8g_optimized/` | 0.75~0.77 |

### 3. 推理测试集

```bash
# 优化模型推理
bash scripts/run_infer.sh \
    outputs/lmf_8g_optimized/checkpoints/best_lmf_optimized.pt \
    pred_test1_optimized.csv \
    /path/to/test \
    configs/whisper_qwen0_6b_lmf_8g_optimized.yaml
```

---

## 📊 预期效果

### 训练时间对比
- **基线模型**: ~80 epochs × 20000 steps/epoch = 1.6M steps
- **优化模型**: 
  - stride=2 导致样本数增加2.5倍
  - 但max_steps_per_epoch=20000限制了每个epoch的步数
  - 实际训练时间相近

### 显存占用
- **基线模型**: ~7.5GB
- **优化模型**: ~7.8GB (增加约300MB)
  - Context encoder: +100MB
  - Fusion: +100MB
  - HandcraftedFeatures: +50MB
  - 其他: +50MB

### 性能提升预期
| 优化项 | 预期提升 |
|--------|---------|
| stride=2 + 损失函数 | +1.5~2% |
| 动态上下文 | +1~1.5% |
| 特征工程 | +0.5~1% |
| 模型微调 | +0.5~1% |
| **总计** | **+3.5~5.5%** |

**预期最终分数: 0.728 → 0.76~0.77**

---

## 🔍 训练监控

### TensorBoard
```bash
tensorboard --logdir outputs/lmf_8g_optimized/logs/tb --host 0.0.0.0 --port 6006
```

### 关键指标
- `train/loss`: 训练损失 (应该逐渐下降)
- `valid/macro_f1`: 验证集宏平均F1 (主要指标)
- `valid/macro_best_f1`: 最优阈值下的F1 (保存模型的依据)
- `valid/{label}_f1`: 每个标签的F1分数
  - 重点关注 `bc_f1` 和 `i_f1` (少数类)

### 预期训练曲线
- **Loss**: 从 ~0.5 降到 ~0.15-0.20
- **Macro F1**: 从 ~0.60 升到 ~0.75-0.78
- **BC F1**: 从 ~0.40 升到 ~0.55-0.60 (难度最大)
- **I F1**: 从 ~0.65 升到 ~0.75-0.78

---

## ⚠️ 注意事项

### 1. 显存不足时的应对
如果训练时OOM，可以尝试：

```yaml
# 方案1: 减少上下文长度
context_chunks: 312  # 从375减到312 (25秒)

# 方案2: 增加梯度累积
train:
  gradient_accumulation_steps: 16  # 从8增加到16

# 方案3: 降低文本长度
text_encoder:
  max_length: 192  # 从256减到192
```

### 2. 训练速度慢时的优化
```yaml
# 减少验证样本数
train:
  eval_valid_sample_count: 4000  # 从8000减到4000

# 减少每个epoch的步数
train:
  max_steps_per_epoch: 10000  # 从20000减到10000
```

### 3. 过拟合时的处理
如果验证集F1不再提升：
- 增加 `dropout: 0.3`
- 增加 `label_smoothing: 0.1`
- 增加 `weight_decay: 0.02`
- 启用更多数据增强

---

## 📝 代码修改清单

### 修改的文件
1. ✅ `configs/whisper_qwen0_6b_lmf_8g_optimized.yaml` - 新配置文件
2. ✅ `src/data/dataset.py` - 添加动态上下文支持
3. ✅ `src/models/multimodal_baseline.py` - 扩展HandcraftedFeatures
4. ✅ `src/train.py` - 添加标签平滑支持
5. ✅ `scripts/run_train_optimized.sh` - 新训练脚本

### 核心改动
- **TurnTakingTrainDataset.__init__**: 添加动态上下文参数
- **TurnTakingTrainDataset.__getitem__**: 实现动态上下文逻辑
- **HandcraftedFeatures**: 从19维扩展到64维(投影后)
- **MultiLabelFocalLoss**: 添加标签平滑支持

---

## 🎯 下一步优化方向

如果这版效果好，可以继续尝试：

### Phase 3: 模型集成 (预期+1~2%)
```bash
# 训练3个不同seed的模型
for seed in 42 123 456; do
    python -m src.train \
        --config configs/whisper_qwen0_6b_lmf_8g_optimized.yaml \
        --seed $seed
done

# 推理时ensemble
python ensemble_predict.py \
    --models model1.pt model2.pt model3.pt \
    --output pred_ensemble.csv
```

### Phase 4: 后处理优化
- 时序平滑: 相邻预测的平滑
- 逻辑约束: C和T互斥等规则
- 上下文一致性检查

---

## 📞 问题排查

### Q1: 训练loss不下降
- 检查学习率是否过小
- 检查梯度是否正常 (grad_clip_norm)
- 尝试降低label_smoothing

### Q2: 验证集F1很低
- 检查阈值是否合理 (best_threshold)
- 检查样本是否平衡
- 尝试调整pos_weight_cap

### Q3: BC和I的F1特别低
- 这两个是少数类，本身就难
- 尝试增加focal_gamma到3.0
- 尝试增加pos_weight_cap到10.0

---

**祝训练顺利！有问题随时问我。** 🚀
