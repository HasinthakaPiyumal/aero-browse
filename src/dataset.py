import torch
from torch.utils.data import Dataset
from src.vla_tokenizer import VLATokenizerConfig

class BrowserAgentDataset(Dataset):
    def __init__(self, raw_web_data, processor):
        self.data = raw_web_data
        self.processor = processor
        self.vla_config = VLATokenizerConfig()
        self.column_names = ["input_ids", "pixel_values", "labels"]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = item["image"]  # This will eventually connect with our frame buffer formats
        user_goal = item["goal"] # e.g., "Click on the login button"
        
        prompt = f"<image>\nUser Objective: {user_goal}\nNext Action:"
        
        target_action_str = self.vla_config.format_target_action(
            action_type=item["action"],
            x=item.get("x"),
            y=item.get("y"),
            text_content=item.get("text")
        ) + "<terminate>"
        
        full_text = prompt + target_action_str
        
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask the prompt tokens so the model only calculates loss on the target_action_str
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }