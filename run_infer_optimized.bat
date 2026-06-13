@echo off
REM 推理脚本 - 优化模型

echo ========================================
echo 设置环境变量（使用镜像源）
echo ========================================

set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=D:/bisai/2026_finvcup_baseline/.cache/huggingface
set TRANSFORMERS_CACHE=D:/bisai/2026_finvcup_baseline/.cache/huggingface
set TORCH_HOME=D:/bisai/2026_finvcup_baseline/.cache/torch

echo ========================================
echo 开始推理测试集
echo ========================================

python -m src.infer_test ^
    --config configs/whisper_qwen0_6b_lmf_8g_optimized.yaml ^
    --checkpoint outputs/lmf_8g_optimized/checkpoints/best_lmf_optimized.pt ^
    --test_root test ^
    --output_csv pred_test_optimized.csv

echo ========================================
echo 推理完成！
echo 输出文件: pred_test_optimized.csv
echo ========================================

pause
