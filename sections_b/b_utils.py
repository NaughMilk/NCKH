import torch
import torchvision
from PIL import Image
from typing import List, Tuple

def to_tensor_img(pil_or_np) -> torch.Tensor:
    """Convert PIL Image or numpy array to tensor"""
    if isinstance(pil_or_np, Image.Image):
        return torchvision.transforms.ToTensor()(pil_or_np)
    return torchvision.transforms.ToTensor()(Image.fromarray(pil_or_np))

def _apply_separate_thresholds(boxes, logits, phrases, combined_caption):
    """Apply separate thresholds based on prompt type"""
    # Import log functions and config from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, CFG
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        class CFG:
            current_box_prompt_thr = 0.3
            current_hand_detection_thr = 0.3
            current_qr_items_thr = 0.3
    
    if boxes is None or len(boxes) == 0:
        return boxes, logits, phrases
    
    filtered_boxes = []
    filtered_logits = []
    filtered_phrases = []
    
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        phrase_lower = phrase.lower().strip()
        
        # Determine threshold based on phrase type
        if any(keyword in phrase_lower for keyword in ["box", "container", "tray", "bin", "crate", "hộp", "thùng"]):
            threshold = CFG.current_box_prompt_thr
            threshold_type = "box_prompt"
        elif any(keyword in phrase_lower for keyword in ["hand", "finger", "palm", "thumb", "wrist"]):
            threshold = CFG.current_hand_detection_thr
            threshold_type = "hand_detection"
        else:
            # Assume it's a QR item (fruit)
            threshold = CFG.current_qr_items_thr
            threshold_type = "qr_items"
        
        # Apply threshold
        if logit >= threshold:
            filtered_boxes.append(box)
            filtered_logits.append(logit)
            filtered_phrases.append(phrase)
            _log_info("GDINO Filter", f"Kept {phrase} (confidence: {logit:.3f}, threshold: {threshold:.3f}, type: {threshold_type})")
        else:
            _log_info("GDINO Filter", f"Filtered {phrase} (confidence: {logit:.3f} < {threshold:.3f}, type: {threshold_type})")
    
    return (torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4)),
            torch.stack(filtered_logits) if filtered_logits else torch.empty((0,)),
            filtered_phrases)
