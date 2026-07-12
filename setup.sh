#!/bin/bash
set -e

echo "=== Шаг 1/5: PyTorch + CUDA ==="
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo "=== Шаг 2/5: Python-зависимости (без трогания torch) ==="
pip install -r requirements_ds_server.txt

echo "=== Шаг 3/5: flash-attn (пересборка под текущий torch) ==="
pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall --no-deps

echo "=== Шаг 4/5: flash-linear-attention (без зависимостей) ==="
pip install flash-linear-attention[cuda]==0.5.1 --no-deps

echo "=== Шаг 5/5: causal-conv1d (сборка с текущим torch) ==="
pip install causal-conv1d --no-build-isolation

echo ""
echo "=== Проверка ==="
python -c "
import torch
print(f'torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.is_available()}')
import flash_attn; print(f'flash-attn: {flash_attn.__version__}')
import fla; print(f'fla: OK')
import causal_conv1d; print(f'causal-conv1d: OK')
"
echo ""
echo "=== Готово ==="