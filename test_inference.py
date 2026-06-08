#!/usr/bin/env python3
"""
🚀 Aero Browse — VLA Inference & Action Prediction Tester
Loads the fine-tuned model, runs inference on a sample web screenshot,
and draws the predicted click/type coordinates on the image.
"""

import os
import re
import argparse
import torch
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test Aero Browse VLA Action Prediction")
    parser.add_argument("--model_dir", type=str, default="./aero-browse-merged", help="Directory of the merged model or base model")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Optional path to LoRA adapter (if using unmerged base model)")
    parser.add_argument("--image_path", type=str, default=None, help="Path to local screenshot image (if None, loads a sample from Mind2Web)")
    parser.add_argument("--goal", type=str, default="Click on the search bar", help="User objective/instruction for the VLA agent")
    parser.add_argument("--output_path", type=str, default="./predicted_action.png", help="Path to save the annotated output screenshot")
    return parser.parse_args()

def load_vla_components(model_dir, adapter_dir=None):
    from src.vla_tokenizer import VLATokenizerConfig
    from transformers import AutoProcessor, AutoModelForImageTextToText as AutoModelForVision2Seq
    
    vla_config = VLATokenizerConfig()
    
    print(f"📦 Loading processor and tokenizer from {model_dir}...")
    try:
        from transformers import Idefics3ImageProcessor, Idefics3Processor
        image_processor = Idefics3ImageProcessor.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        processor = Idefics3Processor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_dir)
        
    tokenizer = processor.tokenizer
    tokenizer.add_tokens(vla_config.all_custom_tokens)
    
    print(f"🤖 Loading VLA model (using device: cuda if available, else cpu)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if adapter_dir:
        from peft import PeftModel
        print(f"   Loading base model {model_dir} with LoRA adapter {adapter_dir}...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        print(f"   Loading merged model from {model_dir}...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        model.resize_token_embeddings(len(tokenizer))
        
    model.eval()
    return processor, model, device

def parse_predicted_action(output_text):
    """Parses action type and coordinates from token output string."""
    action_type = "unknown"
    x, y = None, None
    text_content = ""
    
    if "<click>" in output_text:
        action_type = "click"
    elif "<type>" in output_text:
        action_type = "type"
    elif "<hover>" in output_text:
        action_type = "hover"
    elif "<scroll_down>" in output_text:
        action_type = "scroll_down"
    elif "<scroll_up>" in output_text:
        action_type = "scroll_up"
        
    # Match location tokens like <loc_123> or separate <x_123><y_456>
    locs = re.findall(r"<loc_(\d+)>", output_text)
    if len(locs) >= 2:
        x = int(locs[0])
        y = int(locs[1])
    else:
        x_match = re.search(r"<x_(\d+)>", output_text)
        y_match = re.search(r"<y_(\d+)>", output_text)
        if x_match and y_match:
            x = int(x_match.group(1))
            y = int(y_match.group(1))
            
    # Extract type text if present
    if action_type == "type":
        # Look for <text:"something"> pattern
        text_match = re.search(r'<text:"([^"]+)"\>', output_text)
        if text_match:
            text_content = text_match.group(1)
        else:
            # Fallback to splitting by the last location token
            split_token = f"<y_{y}>" if y is not None and f"<y_{y}>" in output_text else (f"<loc_{locs[1]}>" if len(locs) >= 2 else "")
            if split_token:
                parts = output_text.split(split_token)
                if len(parts) > 1:
                    text_content = parts[1].replace("<terminate>", "").strip()
                    # Clean up <text:"..."> wrapper
                    text_content = re.sub(r'^<text:"(.*)">$', r'\1', text_content)
                    text_content = text_content.strip('"')
            
    return action_type, x, y, text_content

def run_test():
    args = parse_args()
    
    # Load VLA
    processor, model, device = load_vla_components(args.model_dir, args.adapter_dir)
    
    # Load image
    if args.image_path and os.path.exists(args.image_path):
        print(f"🖼️ Loading local image from {args.image_path}...")
        image = Image.open(args.image_path)
    else:
        print("🖼️ No local image provided. Fetching a sample screenshot from osunlp/Multimodal-Mind2Web...")
        from datasets import load_dataset
        import json
        ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train")
        sample = ds[0]
        image = sample["screenshot"]
        
        # Check if user did not provide a custom goal on the command line
        if args.goal == "Click on the search bar":
            # Auto-generate simple goal to align with training format
            candidates = sample["pos_candidates"]
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
                
            op = sample["operation"]
            if isinstance(op, str):
                op = json.loads(op)
            op_type = op.get("op", "CLICK")
            value = op.get("value", "")
            
            from src.dataset import generate_simple_goal
            args.goal = generate_simple_goal(op_type, value, target_cand)
            print(f"   Auto-generated simple goal: '{args.goal}'")
        else:
            print(f"   Using provided goal: '{args.goal}'")
        
    # Preprocess Image
    W, H = image.size
    H_crop = 1024
    if H > H_crop:
        print(f"   Cropping tall image ({W}x{H}) to top {H_crop}px...")
        image = image.crop((0, 0, W, H_crop)).copy()
        H = H_crop
        
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Keep a copy for drawing
    draw_image = image.copy()
    
    # Resize input for SmolVLM vision encoder
    max_img_size = 768
    if max(image.size) > max_img_size:
        image.thumbnail((max_img_size, max_img_size), Image.Resampling.LANCZOS)
        
    # Build prompt
    prompt = f"<image>\nUser Objective: {args.goal}\nNext Action:"
    
    print("🧠 Preparing inputs and running inference...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.encode("<terminate>")[0]
        )
        
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    predicted_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\n🔮 Model Prediction Output: {predicted_text}")
    
    action, x_norm, y_norm, text_val = parse_predicted_action(predicted_text)
    print(f"   Parsed Action: {action.upper()}")
    
    if action in ["scroll_down", "scroll_up"]:
        print(f"   Parsed action is scrolling. Visualizing scroll direction.")
        # Draw visual feedback for scrolling
        draw = ImageDraw.Draw(draw_image)
        label_text = f" ACTION: {action.upper()} "
        
        # Draw an indicator box in the middle of the screen
        box_w, box_h = 300, 60
        left = (W - box_w) // 2
        top = (H - box_h) // 2
        draw.rectangle(
            [(left, top), (left + box_w, top + box_h)],
            fill="black", outline="red", width=3
        )
        draw.text((left + 35, top + 20), label_text, fill="white")
        
        # Draw some directional arrows
        arrow_y_offset = 80
        if action == "scroll_down":
            # Arrow pointing down
            draw.line([(W // 2, top + box_h), (W // 2, top + box_h + arrow_y_offset)], fill="red", width=5)
            draw.line([(W // 2 - 15, top + box_h + arrow_y_offset - 15), (W // 2, top + box_h + arrow_y_offset), (W // 2 + 15, top + box_h + arrow_y_offset - 15)], fill="red", width=5)
        else:
            # Arrow pointing up
            draw.line([(W // 2, top), (W // 2, top - arrow_y_offset)], fill="red", width=5)
            draw.line([(W // 2 - 15, top - arrow_y_offset + 15), (W // 2, top - arrow_y_offset), (W // 2 + 15, top - arrow_y_offset + 15)], fill="red", width=5)
            
        draw_image.save(args.output_path)
        print(f"✅ Annotated screenshot (scrolling action) saved successfully to: {args.output_path}")
    elif x_norm is not None and y_norm is not None:
        # Denormalize coordinates (0-1000) back to actual cropped image pixels
        actual_x = int((x_norm / 1000) * W)
        actual_y = int((y_norm / 1000) * H)
        print(f"   Predicted Coordinates: ({actual_x}, {actual_y}) [Normalized: ({x_norm}, {y_norm})]")
        if action == "type" and text_val:
            print(f"   Typed Text: '{text_val}'")
            
        # Draw on the screenshot
        draw = ImageDraw.Draw(draw_image)
        
        # Draw target dot (red circle)
        radius = 15
        draw.ellipse(
            [(actual_x - radius, actual_y - radius), (actual_x + radius, actual_y + radius)],
            outline="red", width=4
        )
        # Draw outer crosshair lines
        draw.line([(actual_x - 30, actual_y), (actual_x + 30, actual_y)], fill="red", width=2)
        draw.line([(actual_x, actual_y - 30), (actual_x, actual_y + 30)], fill="red", width=2)
        
        # Add label box
        label_text = f" ACTION: {action.upper()} ({x_norm}, {y_norm}) "
        if action == "type" and text_val:
            label_text += f"\n TEXT: '{text_val}' "
        
        # Simple draw text background box
        draw.rectangle(
            [(actual_x + 20, actual_y - 20), (actual_x + 280, actual_y + 20)],
            fill="black", outline="red"
        )
        draw.text((actual_x + 25, actual_y - 15), label_text, fill="white")
        
        draw_image.save(args.output_path)
        print(f"✅ Annotated screenshot saved successfully to: {args.output_path}")
    else:
        print("⚠️  Failed to parse valid coordinates from the model's generated text.")

if __name__ == "__main__":
    run_test()
