"""
Inference script — chat with your fine-tuned Gemma 2 model.

Usage (LoRA adapter):
  python infer.py --adapter ./output/gemma2-sft/lora_adapter

Usage (merged model):
  python infer.py --model ./output/gemma2-sft/merged_model

Usage (GGUF via Ollama):
  ollama create my-gemma -f Modelfile
  ollama run my-gemma
"""

import argparse
import sys

def load_model(adapter_path=None, model_path=None):
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        UNSLOTH = True
    except ImportError:
        UNSLOTH = False

    if UNSLOTH and adapter_path:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name   = adapter_path,
            max_seq_length = 2048,
            dtype        = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    elif UNSLOTH and model_path:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name   = model_path,
            max_seq_length = 2048,
            dtype        = None,
            load_in_4bit = False,
        )
        FastLanguageModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        base_path = model_path or adapter_path
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model     = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    return model, tokenizer


def chat(model, tokenizer, system_prompt: str, max_new_tokens: int = 512):
    import torch

    print("\n" + "━"*50)
    print("  Gemma 2 Chat — type 'exit' to quit")
    print("━"*50 + "\n")

    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            history,
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens      = max_new_tokens,
                temperature         = 0.7,
                top_p               = 0.9,
                repetition_penalty  = 1.1,
                do_sample           = True,
                use_cache           = True,
            )

        response = tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        history.append({"role": "assistant", "content": response})
        print(f"\nGemma: {response}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--model",   default=None, help="Merged model path")
    parser.add_argument("--system",  default="You are an intelligent assistant.",
                        help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if not args.adapter and not args.model:
        print("Error: provide --adapter or --model path")
        sys.exit(1)

    model, tokenizer = load_model(args.adapter, args.model)
    chat(model, tokenizer, args.system, args.max_tokens)


if __name__ == "__main__":
    main()
