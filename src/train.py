import torch
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
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

    print("[Training] Filtering dataset for valid bounding boxes...")
    filtered_dataset = hf_dataset.filter(has_bounding_box)
    print(f"[Training] Filtered dataset size: {len(filtered_dataset)}")

    # Use a subset of the dataset if you want to test the pipeline quickly
    # E.g., train_dataset = BrowserAgentDataset(filtered_dataset.select(range(100)), processor)
    train_dataset = BrowserAgentDataset(filtered_dataset, processor)



    training_args = SFTConfig(
        output_dir="./aero-browse-sft-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, 
        optim="paged_adamw_8bit",
        bf16=True, 
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=50,
        max_length=1024,
    )

    def collate_fn(examples):
        input_ids = [e["input_ids"] for e in examples]
        attention_mask = [e["attention_mask"] for e in examples]
        labels = [e["labels"] for e in examples]
        pixel_values = [e["pixel_values"] for e in examples]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        pixel_values = torch.stack(pixel_values)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
    )

    print("[Training] Starting SFT Training Loop on Colab...")
    trainer.train()
    
    trainer.model.save_pretrained("./aero-browse_lora_adapter")
    print("[Training] Fine-tuning completed and Adapter saved successfully!")

if __name__ == "__main__":
    train_vla()