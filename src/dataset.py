import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from src.vla_tokenizer import VLATokenizerConfig

def generate_simple_goal(op_type, value, target_cand):
    if not target_cand:
        return "Interact with the element"
        
    tag = target_cand.get("tag", "element").strip()
    
    # Extract visible text
    text = target_cand.get("text", "").strip()
    if len(text) > 40:
        text = text[:37] + "..."
        
    attrs = target_cand.get("attributes", {})
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            attrs = {}
            
    # Look for descriptive attributes
    desc = ""
    for attr_name in ["placeholder", "aria-label", "title", "name", "id", "class"]:
        val = attrs.get(attr_name, "")
        if val:
            if isinstance(val, list):
                val = " ".join(val)
            val = str(val).strip()
            if val:
                desc = val
                if len(desc) > 30:
                    desc = desc[:27] + "..."
                break
                
    # Combine tag and description/text
    element_repr = f"{tag}"
    if text:
        element_repr += f" '{text}'"
    elif desc:
        element_repr += f" ({desc})"
        
    if op_type == "TYPE":
        val_str = str(value)
        if len(val_str) > 30:
            val_str = val_str[:27] + "..."
        return f"Type '{val_str}' into the {element_repr}"
    elif op_type == "SELECT":
        return f"Select '{value}' from the {element_repr} dropdown"
    else:
        return f"Click on the {element_repr}"

class BrowserAgentDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor
        self.vla_config = VLATokenizerConfig()
        self.column_names = ["input_ids", "attention_mask", "pixel_values", "labels"]
        
        # Pre-index dataset into flat training steps (incorporating scrolling)
        self.steps = []
        print("🛠️ Pre-indexing dataset into flat training steps (with scrolling)...")
        
        for idx in range(len(self.dataset)):
            row = self.dataset[idx]
            
            # Parse candidates to find target bounding box
            candidates = row["pos_candidates"]
            target_cand = None
            for c_str in candidates:
                c = json.loads(c_str) if isinstance(c_str, str) else c_str
                if c.get("is_original_target"):
                    target_cand = c
                    break
            if not target_cand:
                for c_str in candidates:
                    c = json.loads(c_str) if isinstance(c_str, str) else c_str
                    if c.get("is_top_level_target"):
                        target_cand = c
                        break
            if not target_cand and candidates:
                c_str = candidates[0]
                target_cand = json.loads(c_str) if isinstance(c_str, str) else c_str
                
            # Parse coordinates
            has_valid_coords = False
            cx, cy = None, None
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
                    except:
                        pass
                        
            W, H = row["screenshot"].size
            H_crop = 1024
            
            # Map action type and value
            op = row["operation"]
            if isinstance(op, str):
                op = json.loads(op)
            op_type = op.get("op", "CLICK")
            value = op.get("value", "")
            
            # Generate the simple goal
            simple_goal = generate_simple_goal(op_type, value, target_cand)
            
            if has_valid_coords and H > H_crop:
                # Target is below the fold, generate scrolling steps
                y_start = 0
                while cy >= y_start + H_crop:
                    self.steps.append({
                        "row_idx": idx,
                        "y_start": y_start,
                        "simple_goal": simple_goal,
                        "target_action": "<scroll_down><terminate>"
                    })
                    y_start += 800
                    if y_start > H - H_crop:
                        y_start = H - H_crop
                        break
                
                # Create the final click/type action at the scrolled viewport offset
                new_cy = cy - y_start
                scaled_x = int((cx / W) * 1000)
                scaled_y = int((new_cy / H_crop) * 1000)
                scaled_x = max(0, min(1000, scaled_x))
                scaled_y = max(0, min(1000, scaled_y))
                
                if op_type == "TYPE":
                    action = "type"
                    text_content = value
                elif op_type == "SELECT":
                    action = "click"
                    text_content = None
                else:
                    action = "click"
                    text_content = None
                    
                target_str = self.vla_config.format_target_action(
                    action_type=action,
                    x=scaled_x,
                    y=scaled_y,
                    text_content=text_content
                ) + "<terminate>"
                
                self.steps.append({
                    "row_idx": idx,
                    "y_start": y_start,
                    "simple_goal": simple_goal,
                    "target_action": target_str
                })
            else:
                # Target is visible in the initial viewport, or no valid coords
                scaled_x, scaled_y = 500, 500
                if has_valid_coords:
                    scaled_x = int((cx / W) * 1000)
                    scaled_y = int((cy / H) * 1000)
                    scaled_x = max(0, min(1000, scaled_x))
                    scaled_y = max(0, min(1000, scaled_y))
                    
                if op_type == "TYPE":
                    action = "type"
                    text_content = value
                elif op_type == "SELECT":
                    action = "click"
                    text_content = None
                else:
                    action = "click"
                    text_content = None
                    
                target_str = self.vla_config.format_target_action(
                    action_type=action,
                    x=scaled_x,
                    y=scaled_y,
                    text_content=text_content
                ) + "<terminate>"
                
                self.steps.append({
                    "row_idx": idx,
                    "y_start": 0,
                    "simple_goal": simple_goal,
                    "target_action": target_str
                })
                
        print(f"✅ Pre-indexed {len(self.steps)} flat training steps (expanded from {len(self.dataset)} raw samples).")

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        step_info = self.steps[idx]
        row = self.dataset[step_info["row_idx"]]
        y_start = step_info["y_start"]
        simple_goal = step_info["simple_goal"]
        target_action_str = step_info["target_action"]
        
        screenshot = row["screenshot"]
        W, H = screenshot.size
        H_crop = 1024
        
        if H > H_crop:
            image = screenshot.crop((0, y_start, W, y_start + H_crop)).copy()
        else:
            image = screenshot.copy()
            
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        max_img_size = 768
        if max(image.size) > max_img_size:
            image.thumbnail((max_img_size, max_img_size), Image.Resampling.LANCZOS)
            
        prompt = f"<image>\nUser Objective: {simple_goal}\nNext Action:"
        full_text = prompt + target_action_str
        
        inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
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