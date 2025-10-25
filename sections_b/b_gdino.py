import torch
import cv2
import numpy as np
import time
from typing import List, Tuple

class GDINO:
    def __init__(self, cfg):
        # Import functions from other modules (will be available after all sections are loaded)
        try:
            from sections_b.b_path import _ensure_gdino_on_path, _resolve_gdino_cfg_and_weights
            from sections_a.a_config import _log_info, _log_success, _log_error, CFG
        except ImportError:
            # Fallback if functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
            class CFG:
                device = "cpu"
                gpu_memory_fraction = 0.8
        
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
        # Import functions from other modules (will be available after all sections are loaded)
        try:
            from sections_b.b_utils import to_tensor_img
            from sections_a.a_config import _log_info, CFG
        except ImportError:
            # Fallback if functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            class CFG:
                current_prompt = "plastic box ."
                gdino_prompt = "plastic box ."
                current_box_thr = 0.5
                current_text_thr = 0.35
                device = "cpu"
        
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
        
        # Run inference with autocast
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
        
        # Debug nhanh
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
        # Import functions from other modules (will be available after all sections are loaded)
        try:
            from sections_b.b_utils import to_tensor_img
            from sections_a.a_config import _log_info, _log_success, CFG
        except ImportError:
            # Fallback if functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            class CFG:
                current_box_thr = 0.5
                current_text_thr = 0.35
                device = "cpu"
        
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
        
        # Resize image (train-like settings for better stability) - LIKE NCC_PIPELINE_NEW.py
        scale = self.short_side / float(min(h, w))
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor_img(rgb)
        
        # Stage 1: Detect plastic box container (use exact phrase)
        _log_info("GDINO Two-Stage", "Stage 1: Detecting plastic box container...")
        box_caption = "plastic box ."
        
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
        
        _log_info("GDINO Two-Stage", f"Stage 1 result: {len(boxes_box)} plastic box detections")
        
        # Stage 2: Detect items from QR (if available)
        boxes_items = torch.empty((0, 4))
        logits_items = torch.empty((0,))
        phrases_items = []
        
        if qr_items and len(qr_items) > 0:
            _log_info("GDINO Two-Stage", f"Stage 2: Detecting QR items: {qr_items}")
            # FIXED: Use correct prompt format like NCC_PIPELINE_NEW.py
            items_prompt = " . ".join(qr_items) + " ."
            
            with ctx:
                # FIXED: Use lower thresholds for Stage 2 (fruit detection)
                boxes_items, logits_items, phrases_items = self._predict(
                    model=self.model,
                    image=image_tensor,
                    caption=items_prompt,
                    box_threshold=box_thr * 0.8,  # Lower box threshold for fruits
                    text_threshold=text_thr * 0.8,  # Lower text threshold for fruits
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
