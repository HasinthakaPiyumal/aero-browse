import torch
import sys
try:
    import torchvision
except ImportError:
    print("❌ Error: 'torchvision' library is missing! It is required by SmolVLM's image processor.")
    print("   Please install it in your environment: pip install torchvision")
    sys.exit(1)
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from src.vla_tokenizer import VLATokenizerConfig
from src.dataset import BrowserAgentDataset
import numpy as np

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

def train_vla():
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    
    print("[Training] Loading Tokenizer and Custom Configs...")
    vla_config = VLATokenizerConfig()
    
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    
    num_added_toks = tokenizer.add_tokens(vla_config.all_custom_tokens)
    print(f"Added {num_added_toks} new action/coordinate tokens to vocabulary.")

    from datasets import load_dataset
    import json

    print("[Training] Loading Multimodal-Mind2Web dataset from HF Hub...")
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

    print("[Training] Filtering dataset for valid bounding boxes sequentially...")
    filtered_dataset = hf_dataset.filter(has_bounding_box)
    print(f"[Training] Filtered dataset size: {len(filtered_dataset)}")

    print("[Training] Splitting dataset into 95% train and 5% validation...")
    split_ds = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = BrowserAgentDataset(split_ds["train"], processor)
    eval_dataset = BrowserAgentDataset(split_ds["test"], processor)
    print(f"[Training] Train samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        _attn_implementation="sdpa"
    )

    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"], 
        task_type="CAUSAL_LM",
        ensure_weight_tying=True
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # Use a subset of the dataset if you want to test the pipeline quickly
    # E.g., train_dataset = BrowserAgentDataset(filtered_dataset.select(range(100)), processor)

    training_args = SFTConfig(
        output_dir="./aero-browse-sft-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, 
        optim="adamw_8bit",
        bf16=True, 
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        max_length=1024,
        gradient_checkpointing=True,
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
        
        # Pad pixel_values and pixel_attention_mask along patch dimension
        max_patches = max(p.shape[0] for p in pixel_values)
        padded_pixel_values = []
        padded_pixel_attention_mask = []
        
        for p, p_mask in zip(pixel_values, pixel_attention_mask):
            num_patches = p.shape[0]
            # Pad pixel_values: shape [num_patches, 3, 384, 384] -> [max_patches, 3, 384, 384]
            # padding: (0,0) for dim 3, (0,0) for dim 2, (0,0) for dim 1, (0, max_patches - num_patches) for dim 0
            p_pad = torch.nn.functional.pad(p, (0, 0, 0, 0, 0, 0, 0, max_patches - num_patches), value=0)
            padded_pixel_values.append(p_pad)
            
            # Pad pixel_attention_mask: shape [num_patches, 384, 384] -> [max_patches, 384, 384]
            # padding: (0,0) for dim 2, (0,0) for dim 1, (0, max_patches - num_patches) for dim 0
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
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=collate_fn,
    )

    print("[Training] Starting SFT Training Loop on Colab...")
    trainer.train()
    
    trainer.model.save_pretrained("./aero-browse_lora_adapter")
    print("[Training] Fine-tuning completed and Adapter saved successfully!")

if __name__ == "__main__":
    train_vla()