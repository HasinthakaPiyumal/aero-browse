import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from src.vla_tokenizer import VLATokenizerConfig

class BrowserAgentDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor
        self.vla_config = VLATokenizerConfig()
        self.column_names = ["input_ids", "attention_mask", "pixel_values", "labels"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        
        # Get the instruction/goal
        user_goal = row["confirmed_task"]
        
        # Parse the operation
        op = row["operation"]
        if isinstance(op, str):
            op = json.loads(op)
        
        op_type = op.get("op", "CLICK")
        value = op.get("value", "")
        
        # Parse candidates to find target bounding box
        candidates = row["pos_candidates"]
        target_cand = None
        
        # 1. Look for original target
        for c_str in candidates:
            c = json.loads(c_str) if isinstance(c_str, str) else c_str
            if c.get("is_original_target"):
                target_cand = c
                break
                
        # 2. Look for top level target
        if not target_cand:
            for c_str in candidates:
                c = json.loads(c_str) if isinstance(c_str, str) else c_str
                if c.get("is_top_level_target"):
                    target_cand = c
                    break
                    
        # 3. Fallback to first
        if not target_cand and candidates:
            c_str = candidates[0]
            target_cand = json.loads(c_str) if isinstance(c_str, str) else c_str
            
        # Get bounding box coordinates
        scaled_x, scaled_y = 500, 500  # Default fallback coordinates
        if target_cand:
            attrs = target_cand.get("attributes", {})
            if isinstance(attrs, str):
                attrs = json.loads(attrs)
            
            bbox_str = attrs.get("bounding_box_rect")
            if bbox_str:
                try:
                    parts = [float(x) for x in bbox_str.split(",")]
                    if len(parts) == 4:
                        x, y, w, h = parts
                        cx = x + w / 2
                        cy = y + h / 2
                        
                        # Scale coordinates relative to screenshot size
                        screenshot = row["screenshot"]
                        W, H = screenshot.size
                        scaled_x = int((cx / W) * 1000)
                        scaled_y = int((cy / H) * 1000)
                        
                        # Clip to bounds
                        scaled_x = max(0, min(1000, scaled_x))
                        scaled_y = max(0, min(1000, scaled_y))
                except Exception:
                    pass
        
        # Map action type
        if op_type == "TYPE":
            action = "type"
            text_content = value
        elif op_type == "SELECT":
            action = "click"  # Dropdowns mapped to click action
            text_content = None
        else:
            action = "click"
            text_content = None

        prompt = f"<image>\nUser Objective: {user_goal}\nNext Action:"
        
        target_action_str = self.vla_config.format_target_action(
            action_type=action,
            x=scaled_x,
            y=scaled_y,
            text_content=text_content
        ) + "<terminate>"
        
        full_text = prompt + target_action_str
        
        image = row["screenshot"]
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask the prompt tokens so the model only calculates loss on the target_action_str
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }