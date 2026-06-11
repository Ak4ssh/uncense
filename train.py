"""
Uncensored Gemma 2 QLoRA Fine-Tuning — RunPod A100
====================================================
Strategy: RAW base model (no RLHF) + uncensored SFT data.
          The base model has zero built-in restrictions.
          SFT data teaches the model to respond without refusals.

Supports:
  - Multi-source datasets (local JSONL + HuggingFace)
  - Automatic format normalization
  - Response-only loss masking (trains on completions only)
  - GGUF export for Ollama

Usage:
  python train.py                          # use config.yaml
  python train.py --config config.yaml
  python train.py --resume-from-checkpoint
"""

import os, sys, json, yaml, logging, argparse, random
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Try Unsloth first (2× faster, less VRAM) ─────────────────────────────────
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    UNSLOTH = True
    log.info("✓ Unsloth detected — using optimised CUDA kernels")
except ImportError:
    UNSLOTH = False
    log.warning("⚠ Unsloth not found — falling back to standard HF + PEFT")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    m     = cfg["model"]
    lora  = cfg["lora"]

    if UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = m["name"],
            max_seq_length = m["max_seq_length"],
            dtype          = None,       # auto → bf16 on A100
            load_in_4bit   = True,       # 4-bit QLoRA
        )
        log.info(f"Loaded: {m['name']}  |  4-bit QLoRA via Unsloth")

        model = FastLanguageModel.get_peft_model(
            model,
            r                          = lora["rank"],
            target_modules             = lora["target_modules"],
            lora_alpha                 = lora["alpha"],
            lora_dropout               = lora.get("dropout", 0.0),
            bias                       = "none",
            use_gradient_checkpointing = "unsloth",
            random_state               = 42,
        )
        # Auto-detect chat template from model name
        model_name_lower = m["name"].lower()
        if "llama-3" in model_name_lower or "llama3" in model_name_lower or "dolphin" in model_name_lower or "lexi" in model_name_lower or "hermes" in model_name_lower:
            chat_template = "llama-3"
        elif "mistral" in model_name_lower:
            chat_template = "mistral"
        elif "qwen" in model_name_lower:
            chat_template = "qwen-2.5"
        elif "gemma" in model_name_lower:
            chat_template = "gemma"
        else:
            chat_template = "chatml"

        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        log.info(f"Chat template: {chat_template}")

    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        tokenizer = AutoTokenizer.from_pretrained(m["name"])
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            m["name"],
            quantization_config = bnb,
            device_map          = "auto",
            torch_dtype         = torch.bfloat16,
            attn_implementation = "flash_attention_2",
        )
        peft_cfg = LoraConfig(
            r              = lora["rank"],
            lora_alpha     = lora["alpha"],
            target_modules = lora["target_modules"],
            lora_dropout   = lora.get("dropout", 0.0),
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.0f}M  "
             f"({100*trainable/total:.2f}%)")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset building
# ─────────────────────────────────────────────────────────────────────────────

def normalize(example: dict, sys_prompt: str = "") -> dict | None:
    """Convert any format → {messages: [...]}"""
    msgs = None

    if "messages" in example:
        msgs = example["messages"]

    elif "conversations" in example:
        rmap = {"human": "user", "gpt": "assistant", "system": "system",
                "Human": "user", "Assistant": "assistant"}
        msgs = [{"role": rmap.get(m.get("from", ""), m.get("from", "user")),
                 "content": m.get("value", m.get("content", ""))}
                for m in example["conversations"]]

    elif "instruction" in example:
        msgs = [
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example.get("response",
                                              example.get("output", ""))},
        ]
        if example.get("system"):
            msgs.insert(0, {"role": "system", "content": example["system"]})

    elif "prompt" in example and "response" in example:
        msgs = [
            {"role": "user",      "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]

    elif "question" in example and "answer" in example:
        msgs = [
            {"role": "user",      "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]

    if msgs is None:
        return None

    # Prepend system prompt if no system message present
    if sys_prompt and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": sys_prompt}] + msgs

    return {"messages": msgs}


def load_local(path: str, sys_prompt: str, max_samples: int = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        log.warning(f"Local file not found: {path}")
        return []
    examples = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = normalize(json.loads(line), sys_prompt)
                if ex:
                    examples.append(ex)
            except Exception:
                pass
    log.info(f"  Local  {path}: {len(examples)} examples")
    return examples[:max_samples] if max_samples else examples


def load_hf(name: str, split: str, sys_prompt: str,
            max_samples: int = None) -> list[dict]:
    from datasets import load_dataset
    log.info(f"  HF     {name} [{split}] — loading up to {max_samples}...")
    try:
        ds = load_dataset(name, split=split, streaming=True)
    except Exception as e:
        log.error(f"  Failed to load {name}: {e}")
        return []
    examples = []
    for ex in ds:
        n = normalize(ex, sys_prompt)
        if n:
            examples.append(n)
        if max_samples and len(examples) >= max_samples:
            break
    log.info(f"  HF     {name}: {len(examples)} examples loaded")
    return examples


def build_datasets(cfg: dict, tokenizer):
    from datasets import Dataset as HFDataset

    sys_prompt = cfg.get("system_prompt", "").strip()
    data_cfg   = cfg["data"]
    sources    = data_cfg.get("sources", [])

    all_examples = []
    for src in sources:
        if not src.get("enabled", True):
            continue
        if src["type"] == "local":
            all_examples += load_local(
                src["path"], sys_prompt, src.get("max_samples"))
        elif src["type"] == "huggingface":
            all_examples += load_hf(
                src["name"], src.get("split", "train"),
                sys_prompt, src.get("max_samples"))

    # Fallback: old-style single train_file key
    if not all_examples and "train_file" in data_cfg:
        all_examples = load_local(data_cfg["train_file"], sys_prompt)

    if not all_examples:
        log.error("No training data found! Check config.yaml data.sources")
        sys.exit(1)

    log.info(f"Total examples before dedup: {len(all_examples)}")
    random.shuffle(all_examples)

    # Apply chat template → text field
    def to_text(ex):
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        except Exception:
            return {"text": ""}

    texts = [to_text(ex) for ex in all_examples]
    texts = [t for t in texts if len(t["text"]) > 50]
    log.info(f"Total examples after formatting: {len(texts)}")

    # Train / val split
    val_ratio = data_cfg.get("val_split", 0.05)
    cut = int(len(texts) * (1 - val_ratio))
    train_texts = texts[:cut]
    val_texts   = texts[cut:]

    train_ds = HFDataset.from_list(train_texts)
    val_ds   = HFDataset.from_list(val_texts) if val_texts else None

    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(cfg: dict, model, tokenizer, train_ds, val_ds):
    from trl import SFTTrainer, SFTConfig

    t       = cfg["training"]
    out_dir = cfg.get("output_dir", "./output/uncensored-gemma2")
    os.makedirs(out_dir, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir                  = out_dir,
        run_name                    = cfg.get("wandb", {}).get("run_name", "uncensored-sft"),
        num_train_epochs            = t["epochs"],
        per_device_train_batch_size = t["batch_size"],
        per_device_eval_batch_size  = t.get("eval_batch_size", t["batch_size"]),
        gradient_accumulation_steps = t["grad_accumulation_steps"],
        learning_rate               = t["max_lr"],
        lr_scheduler_type           = "cosine",
        warmup_ratio                = t.get("warmup_ratio", 0.05),
        bf16                        = True,
        fp16                        = False,
        optim                       = "adamw_8bit",
        weight_decay                = t.get("weight_decay", 0.01),
        max_grad_norm               = t.get("grad_clip", 1.0),
        max_seq_length              = cfg["model"]["max_seq_length"],
        dataset_text_field          = "text",
        packing                     = t.get("packing", True),
        logging_steps               = t.get("logging_steps", 10),
        save_steps                  = t.get("save_steps", 100),
        save_total_limit            = 3,
        eval_strategy               = "steps" if val_ds else "no",
        eval_steps                  = t.get("eval_steps", 100) if val_ds else None,
        load_best_model_at_end      = bool(val_ds),
        report_to                   = "wandb" if cfg.get("wandb", {}).get("enabled") else "none",
        seed                        = 42,
        dataloader_num_workers      = 4,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        args          = sft_cfg,
    )

    # Only compute loss on ASSISTANT tokens — model learns uncensored responses
    if UNSLOTH:
        model_name_lower = cfg["model"]["name"].lower()
        if any(k in model_name_lower for k in ["llama-3", "llama3", "dolphin", "lexi", "hermes"]):
            # Llama 3 / Dolphin / Lexi / Hermes all use Llama 3 header format
            instr_part = "<|start_header_id|>user<|end_header_id|>\n\n"
            resp_part  = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in model_name_lower:
            instr_part = "[INST] "
            resp_part  = " [/INST]"
        elif "gemma" in model_name_lower:
            instr_part = "<start_of_turn>user\n"
            resp_part  = "<start_of_turn>model\n"
        else:
            instr_part = "<|im_start|>user\n"
            resp_part  = "<|im_start|>assistant\n"

        trainer = train_on_responses_only(trainer, instr_part, resp_part)
        log.info(f"Response-only masking: '{resp_part.strip()}'")

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export(cfg: dict, model, tokenizer):
    exp     = cfg.get("export", {})
    out_dir = cfg.get("output_dir", "./output/uncensored-gemma2")

    # Always save adapter
    adapter_path = os.path.join(out_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info(f"✓ LoRA adapter → {adapter_path}")

    if not UNSLOTH:
        log.warning("Merged/GGUF export requires Unsloth. Skipping.")
        return

    # Merged 16-bit (HuggingFace format)
    if exp.get("merge_weights", True):
        merged_path = os.path.join(out_dir, "merged_model")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        log.info(f"✓ Merged 16-bit model → {merged_path}")

    # GGUF for Ollama
    if exp.get("gguf", False):
        quant      = exp.get("gguf_quantization", "q4_k_m")
        gguf_dir   = os.path.join(out_dir, "gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
        log.info(f"✓ GGUF ({quant}) → {gguf_dir}")
        log.info("  Deploy: ollama create my-model -f Modelfile")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # WandB
    if cfg.get("wandb", {}).get("enabled"):
        import wandb
        wandb.init(
            project = cfg["wandb"]["project"],
            name    = cfg["wandb"].get("run_name",
                        f"uncensored-{datetime.now():%Y%m%d-%H%M}"),
            config  = cfg,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    log.info("=" * 60)
    log.info(f"  Model  : {cfg['model']['name']}")
    log.info(f"  LoRA r : {cfg['lora']['rank']}  alpha: {cfg['lora']['alpha']}")
    log.info(f"  Epochs : {cfg['training']['epochs']}")
    log.info(f"  MaxSeq : {cfg['model']['max_seq_length']}")
    log.info("=" * 60)

    model, tokenizer       = load_model_and_tokenizer(cfg)
    train_ds, val_ds       = build_datasets(cfg, tokenizer)
    trainer                = build_trainer(cfg, model, tokenizer, train_ds, val_ds)

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    log.info("Exporting model...")
    export(cfg, model, tokenizer)
    log.info("✓ Done.")


if __name__ == "__main__":
    main()
