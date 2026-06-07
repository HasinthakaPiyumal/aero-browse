#!/usr/bin/env python3
"""
🚀 Aero Browse — End-to-End Training & Optimization Pipeline
Fine-tunes SmolVLM on Multimodal-Mind2Web, merges adapters, quantizes to GGUF, and uploads to Hugging Face.
"""

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Aero Browse Training & Quantization Pipeline")
    
    # Model and Repo Configurations
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolVLM-Instruct", help="HF base model ID")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face repo ID to push the merged model and adapter")
    parser.add_argument("--hf_gguf_repo_id", type=str, default=None, help="Hugging Face repo ID to push the quantized GGUF models")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA dimension rank r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max context sequence length")
    parser.add_argument("--subset_size", type=int, default=None, help="If set, trains on a subset of the dataset (for debugging)")
    
    # Output directories
    parser.add_argument("--output_dir", type=str, default="./aero-browse-sft-output", help="Directory to save checkpoints")
    parser.add_argument("--adapter_dir", type=str, default="./aero-browse_lora_adapter", help="Directory to save LoRA adapter")
    parser.add_argument("--merged_dir", type=str, default="./aero-browse-merged", help="Directory to save merged weights")
    parser.add_argument("--gguf_dir", type=str, default="./aero-browse-gguf", help="Directory to save GGUF files")
    
    # W&B Configurations
    parser.add_argument("--wandb_project", type=str, default="aero-browse", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="smolvlm-lora-mind2web", help="Weights & Biases run name")
    
    # Actions
    parser.add_argument("--skip_gguf", action="store_true", help="Skip GGUF quantization step")
    parser.add_argument("--skip_push", action="store_true", help="Skip pushing to Hugging Face Hub")
    
    return parser.parse_args()

def run_pipeline(args):
    # Defer imports of heavy ML libraries until after argument parsing
    print("📦 Loading deep learning libraries...")
    import json
    import gc
    import subprocess
    import torch
    try:
        import torchvision
    except ImportError:
        print("❌ Error: 'torchvision' library is missing! It is required by SmolVLM's image processor.")
        print("   Please install it in your environment (e.g. pip install torchvision)")
        sys.exit(1)
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    from transformers import AutoProcessor, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset, logging as ds_logging

    # Add local path to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.vla_tokenizer import VLATokenizerConfig
    from src.dataset import BrowserAgentDataset

    # Set dataset verbosity to info
    ds_logging.set_verbosity_info()

    # Define the callback inside so it has access to TrainerCallback
    class ConsoleProgressCallback(TrainerCallback):
        """Callback to log progress live to W&B and console tqdm."""
        def __init__(self, max_steps):
            self.pbar = None
            self.max_steps = max_steps

        def on_train_begin(self, args, state, control, **kwargs):
            self.pbar = tqdm(total=self.max_steps, desc="Fine-tuning VLA", dynamic_ncols=True)

        def on_step_end(self, args, state, control, **kwargs):
            if self.pbar:
                self.pbar.n = state.global_step
                self.pbar.refresh()
            
            # Log progress live to W&B on every step
            progress_pct = (state.global_step / self.max_steps) * 100
            import wandb
            wandb.log({
                "progress_percent": progress_pct,
                "global_step_counter": state.global_step
            })

        def on_train_end(self, args, state, control, **kwargs):
            if self.pbar:
                self.pbar.close()

    # ── Check Authentication & Environment ──
    hf_token = os.environ.get("HF_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    
    if not wandb_api_key:
        print("⚠️  WANDB_API_KEY environment variable not set. W&B logging might prompt for login or fail.")
    if not hf_token and not args.skip_push:
        print("❌ Error: HF_TOKEN environment variable is required to upload models to Hugging Face. "
              "Set HF_TOKEN or run with --skip_push.")
        sys.exit(1)

    # Log in to HF Hub early so all downloads are authenticated
    if hf_token:
        try:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token, add_to_git_credential=False)
            print("✅ Logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"⚠️  HF login failed (token may be invalid/expired): {e}")
            print("   Continuing without authentication (public models still work).")
            hf_token = None
        
    if not torch.cuda.is_available():
        print("❌ Error: CUDA-enabled GPU is required for training.")
        sys.exit(1)
        
    print(f"🚀 Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Initialize W&B ──
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        tags=["aero-browse", "smolvlm", "vla", "browser-agent"]
    )
    print(f"✅ W&B initialized: {wandb.run.url}")

    # ── 1. Load Tokenizer & Configs ──
    print("\n[1/5] Injecting action tokens into processor...")
    vla_config = VLATokenizerConfig()

    # SmolVLM's preprocessor_config.json can break AutoProcessor auto-detection
    # in some transformers versions. Try explicit processor classes first.
    processor = None
    for proc_cls_name in ["SmolVLMProcessor", "Idefics3Processor"]:
        try:
            import transformers
            proc_cls = getattr(transformers, proc_cls_name)
            processor = proc_cls.from_pretrained(args.base_model, token=hf_token)
            print(f"      Loaded processor via {proc_cls_name}.")
            break
        except (AttributeError, ImportError, Exception) as e:
            print(f"      {proc_cls_name} unavailable: {e}")
            continue
    if processor is None:
        processor = AutoProcessor.from_pretrained(args.base_model, token=hf_token)
        print("      Loaded processor via AutoProcessor.")

    tokenizer = processor.tokenizer
    num_added = tokenizer.add_tokens(vla_config.all_custom_tokens)
    print(f"      Added {num_added} custom action & coordinate tokens.")

    # ── 2. Load & Filter Dataset ──
    print("\n[2/5] Loading Multimodal-Mind2Web dataset from HF...")
    hf_dataset = load_dataset("osunlp/Multimodal-Mind2Web", split="train")

    def has_bounding_box(example):
        try:
            candidates = example["pos_candidates"]
            if not candidates:
                return False
            for c_str in candidates:
                c = json.loads(c_str) if isinstance(c_str, str) else c_str
                attrs = c.get("attributes", {})
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                if attrs.get("bounding_box_rect"):
                    return True
            return False
        except Exception:
            return False

    print("      Filtering dataset for bounding boxes sequentially (low memory)...")
    filtered_dataset = hf_dataset.filter(has_bounding_box)
    print(f"      Filtered dataset size: {len(filtered_dataset)} samples")

    if args.subset_size is not None and args.subset_size < len(filtered_dataset):
        print(f"      Selecting a subset of {args.subset_size} samples...")
        filtered_dataset = filtered_dataset.select(range(args.subset_size))

    train_dataset = BrowserAgentDataset(filtered_dataset, processor)

    # ── 3. Load Base Model (4-bit NF4) ──
    print("\n[3/5] Loading base model in 4-bit NF4 with SDPA attention...")
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForVision2Seq

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        _attn_implementation="sdpa",
        token=hf_token,
    )
    model.resize_token_embeddings(len(tokenizer))

    # ── 4. Apply LoRA ──
    print("\n[4/5] Constructing LoRA adapter...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── 5. Fine-Tuning SFTTrainer ──
    print("\n[5/5] Launching SFT training loop...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=1,
        max_steps=args.max_steps,
        optim="adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        max_length=args.max_seq_length,
        report_to="wandb",
        run_name=args.wandb_run_name,
        warmup_steps=20,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        disable_tqdm=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    def collate_fn(examples):
        input_ids = [e["input_ids"] for e in examples]
        attention_mask = [e["attention_mask"] for e in examples]
        labels = [e["labels"] for e in examples]
        pixel_values = [e["pixel_values"] for e in examples]
        pixel_attention_mask = [e["pixel_attention_mask"] for e in examples]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Pad vision patches along dimension 0
        max_patches = max(p.shape[0] for p in pixel_values)
        padded_pixel_values = []
        padded_pixel_attention_mask = []

        for p, p_mask in zip(pixel_values, pixel_attention_mask):
            num_patches = p.shape[0]
            p_pad = torch.nn.functional.pad(p, (0, 0, 0, 0, 0, 0, 0, max_patches - num_patches), value=0)
            padded_pixel_values.append(p_pad)
            
            m_pad = torch.nn.functional.pad(p_mask, (0, 0, 0, 0, 0, max_patches - num_patches), value=0)
            padded_pixel_attention_mask.append(m_pad)

        pixel_values = torch.stack(padded_pixel_values)
        pixel_attention_mask = torch.stack(padded_pixel_attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask
        }

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
        callbacks=[ConsoleProgressCallback(max_steps=args.max_steps)]
    )

    train_result = trainer.train()

    # Save adapter
    print(f"\n💾 Saving adapter to {args.adapter_dir}...")
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    
    metrics = train_result.metrics
    wandb.log({"final_train_loss": metrics.get("train_loss", None)})
    print("✅ Training complete. Adapter saved successfully!")

    # ── 6. Merge LoRA Weights ──
    print("\n🔗 Starting LoRA Weight Merge...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("      Loading base model in fp16 on CPU for merge...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=hf_token,
    )
    
    # Match vocabulary size with the adapter config
    base_model.resize_token_embeddings(len(tokenizer))
    
    print("      Loading LoRA adapter...")
    model_with_lora = PeftModel.from_pretrained(base_model, args.adapter_dir)
    
    print("      Merging weights...")
    merged_model = model_with_lora.merge_and_unload()
    
    print(f"      Saving merged model to {args.merged_dir}...")
    merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print("✅ LoRA weights merged successfully!")

    # Free up memory
    del base_model, model_with_lora, merged_model
    gc.collect()

    # ── 7. Quantization to GGUF (llama.cpp) ──
    if not args.skip_gguf:
        print("\n⚙️  Starting GGUF conversion with llama.cpp...")
        llama_cpp_dir = "./llama.cpp"
        
        if not os.path.exists(llama_cpp_dir):
            print("      Cloning llama.cpp...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                check=True
            )
            
        print("      Installing llama.cpp dependencies...")
        subprocess.run(
            ["pip", "install", "-q", "-r", os.path.join(llama_cpp_dir, "requirements.txt")],
            check=True
        )

        os.makedirs(args.gguf_dir, exist_ok=True)
        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
        gguf_fp16_path = os.path.join(args.gguf_dir, "aero-browse-f16.gguf")

        print("      Converting to FP16 GGUF...")
        res = subprocess.run(
            ["python", convert_script, args.merged_dir, "--outfile", gguf_fp16_path, "--outtype", "f16"],
            capture_output=True, text=True
        )
        
        if res.returncode != 0:
            print(f"⚠️  GGUF FP16 conversion failed:\n{res.stderr}")
            print("   Note: SmolVLM might require a custom or updated llama.cpp converter.")
        else:
            size_mb = os.path.getsize(gguf_fp16_path) / (1024 * 1024)
            print(f"✅ FP16 GGUF saved: {gguf_fp16_path} ({size_mb:.0f} MB)")
            
            # Quantize to Q4_K_M
            quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
            gguf_q4_path = os.path.join(args.gguf_dir, "aero-browse-q4_k_m.gguf")
            
            if not os.path.exists(quantize_bin):
                print("      Building llama.cpp quantizer...")
                build_dir = os.path.join(llama_cpp_dir, "build")
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
                subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"], cwd=build_dir, check=True)
                
            if os.path.exists(quantize_bin):
                print("      Quantizing to Q4_K_M...")
                subprocess.run([quantize_bin, gguf_fp16_path, gguf_q4_path, "Q4_K_M"], check=True)
                q4_size = os.path.getsize(gguf_q4_path) / (1024 * 1024)
                print(f"✅ Q4_K_M GGUF saved: {gguf_q4_path} ({q4_size:.0f} MB)")
            else:
                print("⚠️  Quantize binary not compiled; skipping Q4_K_M step.")

    # ── 8. Upload to Hugging Face Hub ──
    if not args.skip_push and args.hf_repo_id:
        print("\n🤗 Pushing artifacts to Hugging Face Hub...")
        from huggingface_hub import HfApi, login
        login(token=hf_token)
        api = HfApi()
        
        # 1. Push merged model
        print(f"      Uploading merged model to hf.co/{args.hf_repo_id}...")
        api.create_repo(args.hf_repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=args.merged_dir,
            repo_id=args.hf_repo_id,
            commit_message="Upload Aero Browse VLA - Merged weights"
        )
        
        # 2. Push adapter
        adapter_repo = f"{args.hf_repo_id}-lora"
        print(f"      Uploading adapter to hf.co/{adapter_repo}...")
        api.create_repo(adapter_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=args.adapter_dir,
            repo_id=adapter_repo,
            commit_message="Upload Aero Browse VLA - LoRA adapter only"
        )
        
        # 3. Push GGUF
        if not args.skip_gguf and args.hf_gguf_repo_id and os.path.exists(args.gguf_dir):
            print(f"      Uploading GGUF to hf.co/{args.hf_gguf_repo_id}...")
            api.create_repo(args.hf_gguf_repo_id, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=args.gguf_dir,
                repo_id=args.hf_gguf_repo_id,
                commit_message="Upload Aero Browse VLA - GGUF quantized models"
            )
            
        print("✅ Hugging Face Hub upload completed successfully!")

    # ── Cleanup & Finalize ──
    def get_dir_size_mb(path):
        total = 0
        if os.path.exists(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 * 1024)

    summary = {
        "training_samples": len(train_dataset),
        "max_steps": args.max_steps,
        "adapter_size_mb": get_dir_size_mb(args.adapter_dir),
        "merged_model_size_mb": get_dir_size_mb(args.merged_dir),
    }
    if not args.skip_gguf:
        summary["gguf_dir_size_mb"] = get_dir_size_mb(args.gguf_dir)
        
    wandb.log(summary)
    wandb.finish()
    print("\n🎉 Aero Browse Pipeline Completed Successfully!")

if __name__ == "__main__":
    # If they run python train_pipeline.py --help, parse_args() executes first,
    # prints help, and exits. Deep learning libraries are never imported.
    args = parse_args()
    run_pipeline(args)
