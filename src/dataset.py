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
        # Get raw screenshot and dimensions
        screenshot = row["screenshot"]
        W, H = screenshot.size
        H_crop = 1024
        
        # Bounding box and coordinates extraction
        scaled_x, scaled_y = 500, 500
        image = screenshot
        
        # Extract target center if candidate exists
        has_valid_coords = False
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
                        has_valid_coords = True
                except Exception:
                    pass
                    
        # Apply vertical cropping if the screenshot is too tall (e.g. 5000+ pixels)
        # to prevent OOM errors and massive patch dimensions in SmolVLM.
        if H > H_crop:
            if has_valid_coords:
                y_start = int(cy - H_crop / 2)
                y_start = max(0, min(H - H_crop, y_start))
                image = screenshot.crop((0, y_start, W, y_start + H_crop)).copy()
                new_cy = cy - y_start
            else:
                y_start = 0
                image = screenshot.crop((0, 0, W, H_crop)).copy()
                new_cy = 500
            
            actual_crop_height = H_crop
        else:
            y_start = 0
            image = screenshot.copy()
            new_cy = cy if has_valid_coords else 500
            actual_crop_height = H
            
        if has_valid_coords:
            scaled_x = int((cx / W) * 1000)
            scaled_y = int((new_cy / actual_crop_height) * 1000)
            scaled_x = max(0, min(1000, scaled_x))
            scaled_y = max(0, min(1000, scaled_y))
        
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
        
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Tokenize only target_action_str to find its token length.
        # This avoids running the expensive vision processor twice.
        target_inputs = self.processor.tokenizer(target_action_str, add_special_tokens=False)
        target_len = len(target_inputs["input_ids"])
        
        prompt_len = max(0, input_ids.shape[0] - target_len)
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "pixel_attention_mask": inputs["pixel_attention_mask"].squeeze(0),
            "labels": labels
        }