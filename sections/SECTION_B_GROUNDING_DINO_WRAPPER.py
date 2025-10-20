# ========================= SECTION B: GROUNDING DINO WRAPPER ========================= #
# ========================= SECTION B: GROUNDING DINO WRAPPER ========================= #

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional



import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
def to_tensor_img(pil_or_np) -> torch.Tensor:
    """Convert PIL Image or numpy array to tensor"""
    if isinstance(pil_or_np, Image.Image):
        return torchvision.transforms.ToTensor()(pil_or_np)
    return torchvision.transforms.ToTensor()(Image.fromarray(pil_or_np))

def _ensure_gdino_on_path(cfg: Config) -> None:
    _log_info("GroundingDINO", "Checking GroundingDINO availability...")
    
    def _try_import():
        try:
            from groundingdino.util.inference import load_model, predict, annotate  # noqa
            import groundingdino  # noqa
            _log_success("GroundingDINO Import", "Successfully imported from installed package")
            return True
        except ImportError as e:
            _log_warning("GroundingDINO Import", f"Package not found: {e}")
            return False
        except Exception as e:
            _log_error("GroundingDINO Import", e, "Unexpected error during import")
            return False
    
    if _try_import():
        return
    
    _log_info("GroundingDINO", "Trying local repositories...")
    for i, p in enumerate(cfg.gdino_repo_candidates):
        _log_info("GroundingDINO", f"Checking candidate {i+1}/{len(cfg.gdino_repo_candidates)}: {p}")
        
        if not os.path.exists(p):
            _log_warning("GroundingDINO Path", f"Path does not exist: {p}")
            continue
            
        if not os.path.isdir(p):
            _log_warning("GroundingDINO Path", f"Path is not a directory: {p}")
            continue
            
        try:
            sys.path.insert(0, os.path.normpath(p))
            if _try_import():
                _log_success("GroundingDINO", f"Using local repo: {p}")
                return
            else:
                _log_warning("GroundingDINO", f"Import failed from: {p}")
        except Exception as e:
            _log_error("GroundingDINO Path", e, f"Error processing path: {p}")
    
    _log_error("GroundingDINO", RuntimeError("GroundingDINO not found"), 
               f"Checked {len(cfg.gdino_repo_candidates)} candidates: {cfg.gdino_repo_candidates}")
    raise RuntimeError("Không tìm thấy GroundingDINO. Hãy cài repo và cập nhật đường dẫn trong Config.")


def _resolve_gdino_cfg_and_weights(cfg: Config) -> Tuple[str, str]:
    _log_info("GroundingDINO Config", "Resolving config and weights files...")
    
    try:
        import groundingdino as _gd
        pkg_dir = os.path.dirname(_gd.__file__)
        _log_info("GroundingDINO Config", f"Package directory: {pkg_dir}")
    except Exception as e:
        _log_error("GroundingDINO Config", e, "Failed to import groundingdino package")
        raise
    
    # Tìm config file
    cand = [os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")]
    for base in cfg.gdino_repo_candidates:
        if os.path.exists(base):
            cand.append(os.path.join(base, cfg.gdino_cfg_rel))
    
    _log_info("GroundingDINO Config", f"Checking {len(cand)} config candidates...")
    gdino_cfg = None
    for i, c in enumerate(cand):
        _log_info("GroundingDINO Config", f"Checking config {i+1}/{len(cand)}: {c}")
        if os.path.isfile(c):
            gdino_cfg = os.path.normpath(c)
            _log_success("GroundingDINO Config", f"Found config: {gdino_cfg}")
            break
        else:
            _log_warning("GroundingDINO Config", f"Config not found: {c}")
    
    if gdino_cfg is None:
        _log_error("GroundingDINO Config", FileNotFoundError("Config not found"), 
                   f"Checked {len(cand)} candidates: {cand}")
        raise FileNotFoundError("Không tìm thấy GroundingDINO_SwinT_OGC.py")
    
    # Tìm weights file
    _log_info("GroundingDINO Weights", f"Checking {len(cfg.gdino_weights_candidates)} weight candidates...")
    for i, w in enumerate(cfg.gdino_weights_candidates):
        _log_info("GroundingDINO Weights", f"Checking weight {i+1}/{len(cfg.gdino_weights_candidates)}: {w}")
        if os.path.isfile(w):
            weight_path = os.path.normpath(w)
            _log_success("GroundingDINO Weights", f"Found weights: {weight_path}")
            return gdino_cfg, weight_path
        else:
            _log_warning("GroundingDINO Weights", f"Weight not found: {w}")
    
    _log_error("GroundingDINO Weights", FileNotFoundError("Weights not found"), 
               f"Checked {len(cfg.gdino_weights_candidates)} candidates: {cfg.gdino_weights_candidates}")
    raise FileNotFoundError("Không tìm thấy 'groundingdino_swint_ogc.pth'. Cập nhật Config.gdino_weights_candidates")

# ========================= SECTION B: GROUNDINGDINO WRAPPER ========================= #

class GDINO:
    def __init__(self, cfg: Config):
        _log_info("GDINO Init", "Starting GroundingDINO initialization...")
        
        try:
            _ensure_gdino_on_path(cfg)
            _log_success("GDINO Init", "GroundingDINO path verified")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to ensure GroundingDINO path")
            raise
        
        try:
            from groundingdino.util.inference import load_model, predict, annotate
            self._predict = predict
            self._annotate = annotate
            _log_success("GDINO Init", "GroundingDINO functions imported")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to import GroundingDINO functions")
            raise
        
        try:
            cfg_path, w_path = _resolve_gdino_cfg_and_weights(cfg)
            _log_info("GDINO Init", f"Loading model from config: {cfg_path}")
            _log_info("GDINO Init", f"Loading weights from: {w_path}")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to resolve config and weights")
            raise
        
        try:
            self.model = load_model(cfg_path, w_path)
            _log_success("GDINO Init", "Model loaded successfully")
        except Exception as e:
            _log_error("GDINO Init", e, f"Failed to load model from {cfg_path}, {w_path}")
            raise
        
        try:
            self.model = self.model.to(CFG.device).eval()
            self.short_side = cfg.gdino_short_side
            self.max_size = cfg.gdino_max_size
            
            # GPU memory optimization
            if CFG.device == "cuda":
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(CFG.gpu_memory_fraction)
                _log_info("GDINO Init", f"GPU memory fraction set to {CFG.gpu_memory_fraction}")
            
            _log_success("GDINO Init", f"Model ready on {CFG.device} | short_side={self.short_side}, max_size={self.max_size}")
        except Exception as e:
            _log_error("GDINO Init", e, f"Failed to move model to {CFG.device}")
            raise

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray, caption: str = None, box_thr: float = None, text_thr: float = None, qr_items: List[str] = None) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        GroundingDINO inference với khả năng detect cả box tổng và vật thể bên trong từ QR
        """
        
        # Validate input
        if frame_bgr is None:
            raise ValueError("Input frame is None")
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame_bgr.shape}, expected (H, W, 3)")
        
        h, w = frame_bgr.shape[:2]
        
        # Set defaults
        if caption is None:
            caption = CFG.current_prompt if CFG.current_prompt else CFG.gdino_prompt
        if box_thr is None:
            box_thr = CFG.current_box_thr
        if text_thr is None:
            text_thr = CFG.current_text_thr
        
        # Resize image (train-like settings for better stability)
        scale = self.short_side / float(min(h, w))
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor_img(rgb)
        
        # Use caption directly (no more complex prompt combination)
        _log_info("GDINO", f"Using prompt: {caption}")
        _log_info("GDINO", f"Thresholds - Box: {box_thr}, Text: {text_thr}")
        
        # Run inference with autocast (COPY Y CHANG từ DEMO GROUNDING DINO.py)
        if CFG.device == "cuda":
            ctx = torch.amp.autocast('cuda')
        else:
            ctx = torch.amp.autocast('cpu')
        
        with ctx:
            boxes, logits, phrases = self._predict(
                model=self.model,
                image=image_tensor,             # ← tensor, predict() sẽ .to(device)
                caption=caption,
                box_threshold=float(box_thr),
                text_threshold=float(text_thr)
            )
        
        # Debug nhanh (COPY Y CHANG từ DEMO GROUNDING DINO.py)
        try:
            if len(boxes) > 0:
                b = boxes[0].detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)[0]
                _log_info("GDINO Debug", f"First raw box from predict(): {b}")
                
                # Debug: Log all raw boxes from GroundingDINO
                _log_info("GDINO Debug", f"All raw boxes from predict(): {[box.tolist() for box in boxes]}")
        except Exception:
            pass
        _log_info("GDINO Debug", f"Original: {frame_bgr.shape}, Resized: {frame_resized.shape}, Scale: {scale:.4f}")
        _log_info("GDINO Debug", f"Detected: {len(boxes)} objects")
        
        # Debug: Log raw detections before filtering
        if boxes is not None and len(boxes) > 0:
            _log_info("GDINO Raw", f"Raw detections: {len(boxes)} boxes, {len(phrases)} phrases")
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                _log_info("GDINO Raw", f"Raw {i}: phrase='{phrase}', confidence={logit:.3f}, bbox={box.tolist()}")
        
        # Return original detections (no separate threshold filtering - like NCC_PROCESS.py)
        return boxes, logits, phrases, frame_resized
    
    @torch.inference_mode()
    def infer_two_stage(self, frame_bgr: np.ndarray, qr_items: List[str] = None, box_thr: float = None, text_thr: float = None) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        2-stage GroundingDINO inference:
        Stage 1: Detect box container
        Stage 2: Detect items from QR code
        """
        import time
        start_time = time.time()
        
        # Validate input
        if frame_bgr is None:
            raise ValueError("Input frame is None")
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame_bgr.shape}, expected (H, W, 3)")
        
        h, w = frame_bgr.shape[:2]
        
        # Set defaults
        if box_thr is None:
            box_thr = CFG.current_box_thr
        if text_thr is None:
            text_thr = CFG.current_text_thr
        
        # Resize image (train-like settings for better stability)
        scale = self.short_side / float(min(h, w))
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor_img(rgb)
        
        # Stage 1: Detect box container
        _log_info("GDINO Two-Stage", "Stage 1: Detecting box container...")
        box_caption = "box ."
        
        if CFG.device == "cuda":
            ctx = torch.amp.autocast('cuda')
        else:
            ctx = torch.amp.autocast('cpu')
        
        with ctx:
            boxes_box, logits_box, phrases_box = self._predict(
                model=self.model,
                image=image_tensor,
                caption=box_caption,
                box_threshold=box_thr,
                text_threshold=text_thr,
            )
        
        _log_info("GDINO Two-Stage", f"Stage 1 result: {len(boxes_box)} box detections")
        
        # Stage 2: Detect items from QR (if available)
        boxes_items = torch.empty((0, 4))
        logits_items = torch.empty((0,))
        phrases_items = []
        
        if qr_items and len(qr_items) > 0:
            _log_info("GDINO Two-Stage", f"Stage 2: Detecting QR items: {qr_items}")
            items_prompt = " . ".join(qr_items) + " ."
            
            with ctx:
                boxes_items, logits_items, phrases_items = self._predict(
                    model=self.model,
                    image=image_tensor,
                    caption=items_prompt,
                    box_threshold=box_thr,
                    text_threshold=text_thr,
                )
            
            _log_info("GDINO Two-Stage", f"Stage 2 result: {len(boxes_items)} item detections")
        else:
            _log_info("GDINO Two-Stage", "Stage 2: No QR items provided, skipping")
        
        # Combine results
        if len(boxes_box) > 0 and len(boxes_items) > 0:
            all_boxes = torch.cat([boxes_box, boxes_items])
            all_logits = torch.cat([logits_box, logits_items])
            all_phrases = phrases_box + phrases_items
        elif len(boxes_box) > 0:
            all_boxes = boxes_box
            all_logits = logits_box
            all_phrases = phrases_box
        elif len(boxes_items) > 0:
            all_boxes = boxes_items
            all_logits = logits_items
            all_phrases = phrases_items
        else:
            all_boxes = torch.empty((0, 4))
            all_logits = torch.empty((0,))
            all_phrases = []
        
        total_time = time.time() - start_time
        _log_info("GDINO Two-Stage", f"Combined result: {len(all_boxes)} total detections")
        _log_info("GDINO Two-Stage", f"Phrases: {all_phrases}")
        _log_success("GDINO Timing", f"Two-stage inference completed in {total_time*1000:.1f}ms")
        
        return all_boxes, all_logits, all_phrases, frame_resized
    
    def _apply_separate_thresholds(self, boxes, logits, phrases, combined_caption):
        """Apply separate thresholds based on prompt type"""
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

