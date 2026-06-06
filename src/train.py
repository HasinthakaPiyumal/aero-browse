import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from src.vla_tokenizer import VLATokenizerConfig
from src.dataset import BrowserAgentDataset
import numpy as np

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
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    mock_raw_data = [
        {
            "image": np.zeros((448, 448, 3), dtype=np.uint8),
            "goal": "Click the sign in button",
            "action": "click",
            "x": 420,
            "y": 85
        }
    ] * 10
    
    train_dataset = BrowserAgentDataset(mock_raw_data, processor)

    training_args = TrainingArguments(
        output_dir="./velovla-sft-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        optim="paged_adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=50
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=1024,
    )

    print("[Training] Starting SFT Training Loop on Colab...")
    trainer.train()
    
    trainer.model.save_pretrained("./velovla_lora_adapter")
    print("[Training] Fine-tuning completed and Adapter saved successfully!")

if __name__ == "__main__":
    train_vla()