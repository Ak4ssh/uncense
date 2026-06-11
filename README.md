# Uncensored LLM Fine-Tuning — Llama 3 8B + QLoRA

Fine-tune a fully uncensored Llama 3 8B model using Unsloth + QLoRA on RunPod A100.  
No restrictions. No filters. Export to GGUF for Ollama deployment.

## Stack
- **Model**: Llama 3 8B base (no RLHF)
- **Method**: QLoRA (4-bit, rank 64)
- **Framework**: Unsloth + TRL + PEFT
- **GPU**: A100 40GB (RunPod)
- **Export**: GGUF → Ollama

## Quick Start (RunPod)

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/uncensored-llm.git
cd uncensored-llm

# 2. Install dependencies
bash setup.sh

# 3. (Optional) Add your own data
# Upload your JSONL to data/train.jsonl
# Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# 4. Configure (edit model, epochs, datasets)
nano config.yaml

# 5. Train (inside tmux — survives disconnects)
tmux new -s train
python train.py

# 6. Chat with result
python infer.py --adapter ./output/uncensored-llama3/lora_adapter
```

## Data Format (JSONL)

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "response": "..."}
{"prompt": "...", "response": "..."}
```

Run `prepare_data.py` to convert any format automatically:
```bash
python prepare_data.py --input raw_data.jsonl --split 0.9
```

## Deploy with Ollama

```bash
# After training completes:
ollama create my-model -f Modelfile
ollama run my-model
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `config.yaml` | Hyperparameters, model, datasets |
| `prepare_data.py` | Data format converter |
| `setup.sh` | RunPod environment setup |
| `infer.py` | Interactive chat inference |
| `Modelfile` | Ollama deployment config |
| `data/train.jsonl` | Sample training data |
