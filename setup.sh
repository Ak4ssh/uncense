#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# RunPod A100 Setup Script — Gemma 2 QLoRA Fine-Tuning
# ══════════════════════════════════════════════════════════════════════════════
# Run this ONCE after connecting to your RunPod instance:
#   bash setup.sh
#
# Recommended RunPod template: RunPod PyTorch 2.4 / CUDA 12.4
# ══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Gemma 2 QLoRA Setup — RunPod A100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── GPU Info ──────────────────────────────────────────────────────────────────
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq \
    git curl wget unzip tmux htop nvtop \
    libssl-dev libffi-dev python3-dev > /dev/null
echo "✓ System packages"

# ── Unsloth (fastest path — installs torch + transformers automatically) ──────
echo "Installing Unsloth + dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
pip install --no-deps trl peft accelerate bitsandbytes -q
echo "✓ Unsloth installed"

# ── Additional packages ───────────────────────────────────────────────────────
pip install \
    datasets \
    wandb \
    scipy \
    pyyaml \
    einops \
    sentencepiece \
    protobuf \
    -q
echo "✓ Additional packages installed"

# ── Flash Attention 2 (needed for full speed on A100) ────────────────────────
pip install flash-attn --no-build-isolation -q
echo "✓ Flash Attention 2 installed"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python3 -c "
import torch, transformers, unsloth, peft, trl, bitsandbytes
print(f'  torch       : {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')
print(f'  transformers: {transformers.__version__}')
print(f'  trl         : {trl.__version__}')
print(f'  peft        : {peft.__version__}')
print(f'  bitsandbytes: {bitsandbytes.__version__}')
print(f'  unsloth     : OK')
"

# ── Create data directory ─────────────────────────────────────────────────────
mkdir -p data output

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Upload your dataset:  scp data.jsonl root@<pod>:/workspace/data/"
echo "  2. Prepare data:         python prepare_data.py --input data/data.jsonl --split 0.9"
echo "  3. Edit config.yaml      (set model, epochs, etc.)"
echo "  4. Start training:       tmux new -s train"
echo "                           python train.py --config config.yaml"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
