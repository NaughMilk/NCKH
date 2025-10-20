import os
import sys
import json
import cv2
import numpy as np
import torch
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

def _pick_box_bbox(boxes, phrases, qr_points, img_shape):
    """Pick best box bbox from detections"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, CFG
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        class CFG:
            device = "cpu"
    
    if boxes is None or len(boxes) == 0:
        return None
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, device=CFG.device)
    H, W = img_shape[:2]

    _log_info("Bbox Selection", f"Image shape: {img_shape}, Processing {len(boxes)} detections")

    # bbox QR (nếu có)
    qr_bonus_box = None
    if qr_points is not None and len(qr_points) >= 4:
        xq1, yq1 = qr_points.min(axis=0)
        xq2, yq2 = qr_points.max(axis=0)
        qr_bonus_box = (int(xq1), int(yq1), int(xq2), int(yq2))
        _log_info("Bbox Selection", f"QR bonus box: {qr_bonus_box}")

    best, best_box = -1e9, None
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.tolist()
        
        # Boxes are already in pixel coordinates from _to_pixel_xyxy
        # No need to convert from normalized coordinates
        
        # Ensure proper bbox coordinates (x1 < x2, y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Clamp to image bounds
        x1 = max(0, min(W-1, x1))
        y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2))
        y2 = max(0, min(H-1, y2))
        
        area = max(0., (x2-x1)) * max(0., (y2-y1))
        phrase = (phrases[i] if i < len(phrases) else "").lower()
        kw = 1.0 if any(k in phrase for k in ("box","container","tray","bin","crate","hộp","thùng")) else 0.0

        qr_iou = 0.0
        if qr_bonus_box:
            xa1, ya1, xa2, ya2 = map(int, [x1,y1,x2,y2])
            xb1, yb1, xb2, yb2 = qr_bonus_box
            inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
            uni = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter + 1e-6
            qr_iou = inter / uni

        # FIXED: Improved scoring to prioritize proper box selection
        # Base score from area, bonus for box keywords, bonus for QR overlap
        base_score = area
        keyword_bonus = 0.3 * area * kw  # Reduced from 0.5 to 0.3 for better balance
        qr_bonus = 0.2 * area * qr_iou   # Increased from 0.15 to 0.2 for QR importance
        
        # Additional penalty for very small boxes
        size_penalty = 0.0
        if area < 1000:  # Less than 1000 pixels
            size_penalty = 0.1 * (1000 - area)
        
        score = base_score + keyword_bonus + qr_bonus - size_penalty
        
        _log_info("Bbox Selection", f"Detection {i}: phrase='{phrase}', bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), area={area:.0f}, kw={kw}, qr_iou={qr_iou:.3f}, score={score:.0f}")
        
        if score > best:
            best, best_box = score, (int(x1), int(y1), int(x2), int(y2))
            _log_info("Bbox Selection", f"New best: {best_box} (score: {score:.0f})")
    
    _log_info("Bbox Selection", f"Final selected bbox: {best_box}")
    return best_box

def _save_rejected_image(frame_bgr, boxes, phrases, selected_bbox, reason, rejected_dir):
    """Save rejected image with bbox visualization for debugging"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    try:
        # Create visualization
        vis = frame_bgr.copy()
        
        # Draw all detected bboxes
        if boxes is not None and len(boxes) > 0:
            boxes_pixel = _to_pixel_xyxy(boxes, frame_bgr.shape[1], frame_bgr.shape[0])
            for i, (box, phrase) in enumerate(zip(boxes_pixel, phrases)):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for first, red for others
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{phrase}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw selected bbox (the one that was rejected)
        if selected_bbox:
            x1, y1, x2, y2 = map(int, selected_bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for selected
            cv2.putText(vis, "SELECTED", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add reason text
        cv2.putText(vis, f"REJECTED: {reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save image
        timestamp = int(time.time() * 1000)
        filename = f"rejected_{reason}_{timestamp}.jpg"
        filepath = os.path.join(rejected_dir, filename)
        
        cv2.imwrite(filepath, vis)
        _log_info("Rejected Image", f"Saved rejected image: {filename}")
        
    except Exception as e:
        _log_warning("Rejected Image", f"Failed to save rejected image: {e}")

def _to_pixel_xyxy(boxes_tensor, img_w, img_h):
    """Convert normalized boxes to pixel coordinates"""
    if boxes_tensor is None or len(boxes_tensor) == 0:
        return []
    
    if isinstance(boxes_tensor, torch.Tensor):
        boxes = boxes_tensor.cpu().numpy()
    else:
        boxes = np.array(boxes_tensor)
    
    # Convert from normalized [0,1] to pixel coordinates
    pixel_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_pixel = int(x1 * img_w)
        y1_pixel = int(y1 * img_h)
        x2_pixel = int(x2 * img_w)
        y2_pixel = int(y2 * img_h)
        pixel_boxes.append([x1_pixel, y1_pixel, x2_pixel, y2_pixel])
    
    return pixel_boxes

def _get_fruit_class_id(phrase: str, qr_items_dict: Dict[str, int]) -> int:
    """Get class ID for fruit based on phrase and QR items"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
        from sections_e.e_qr_utils import _map_fruit_name_to_qr_item
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _map_fruit_name_to_qr_item(fruit_name, qr_items): return fruit_name
    
    phrase_lower = phrase.lower()
    
    # Check if phrase contains any fruit from QR
    for qr_fruit in qr_items_dict.keys():
        if qr_fruit.lower() in phrase_lower:
            _log_info("Fruit Class ID", f"Matched '{phrase}' to QR fruit '{qr_fruit}'")
            return _get_class_id_for_fruit(qr_fruit)
    
    # If no match, try to map the phrase to a QR fruit
    mapped_fruit = _map_fruit_name_to_qr_item(phrase, qr_items_dict)
    if mapped_fruit != phrase:
        _log_info("Fruit Class ID", f"Mapped '{phrase}' to '{mapped_fruit}'")
        return _get_class_id_for_fruit(mapped_fruit)
    
    # Default to generic fruit class
    _log_warning("Fruit Class ID", f"No specific match for '{phrase}', using generic fruit class")
    return 21  # Generic fruit class ID

def _get_class_id_for_fruit(fruit_name: str) -> int:
    """Get class ID for specific fruit name"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Fruit name to class ID mapping
    fruit_class_mapping = {
        'tangerine': 1, 'orange': 2, 'banana': 3, 'apple': 4, 'mango': 5,
        'grape': 6, 'strawberry': 7, 'kiwi': 8, 'pineapple': 9, 'watermelon': 10,
        'lemon': 11, 'lime': 12, 'peach': 13, 'pear': 14, 'cherry': 15,
        'blueberry': 16, 'raspberry': 17, 'blackberry': 18, 'coconut': 19, 'avocado': 20
    }
    
    fruit_lower = fruit_name.lower().strip()
    class_id = fruit_class_mapping.get(fruit_lower, 21)  # Default to generic fruit
    
    _log_info("Fruit Class ID", f"'{fruit_name}' -> class ID {class_id}")
    return class_id

def update_class_names(self):
    """Update class names in dataset"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    
    # Update dataset class names
    self.ds.class_names = self.ds.class_names[:1] + list(self.detected_classes)
    self.ds.class_id_counter = len(self.ds.class_names)
    
    # Update YAML files
    self.ds._create_initial_yaml_files()
    
    _log_success("Class Update", f"Updated class names: {self.ds.class_names}")
    _log_info("Class Update", f"Total classes: {len(self.ds.class_names)}")

def process_frame(self, frame_bgr: np.ndarray, preview_only: bool = False, save_dataset: bool = True, return_both_visualizations: bool = False):
    """Process single frame through the pipeline"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
        from sections_e.e_qr_utils import parse_qr_payload, check_hand_detection, validate_qr_detection
        from sections_c.c_segmentation import segment_box_by_boxprompt, segment_object_by_point
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
        def parse_qr_payload(s): return {}
        def check_hand_detection(phrases): return False, "No hand detected"
        def validate_qr_detection(qr_items, phrases): return False, "No validation"
        def segment_box_by_boxprompt(bg_removal, img, bbox): return None
        def segment_object_by_point(bg_removal, img, point): return None
    
    self.processed_frames += 1
    _log_info("Frame Processing", f"Processing frame {self.processed_frames}")
    
    try:
        # Step 1: QR Detection
        qr_data, qr_points = self.qr.decode(frame_bgr)
        qr_items = {}
        if qr_data:
            qr_payload = parse_qr_payload(qr_data)
            qr_items = qr_payload.get('fruits', {})
            _log_info("QR Detection", f"Detected QR with {len(qr_items)} fruit types")
        
        # Step 2: GroundingDINO Detection
        boxes, logits, phrases, img_resized = self.gd.infer_two_stage(frame_bgr, qr_items)
        
        if boxes is None or len(boxes) == 0:
            _log_warning("Frame Processing", "No objects detected")
            return None
        
        # Step 3: Hand Detection Check
        hand_detected, hand_reason = check_hand_detection(phrases)
        if hand_detected:
            _log_warning("Frame Processing", f"Hand detected: {hand_reason}")
            if save_dataset:
                _save_rejected_image(frame_bgr, boxes, phrases, None, "hand_detected", self.rejected_dir)
                self.rejected_samples += 1
            return None
        
        # Step 4: QR Validation
        if qr_items:
            qr_valid, qr_reason = validate_qr_detection(qr_items, phrases)
            if not qr_valid:
                _log_warning("Frame Processing", f"QR validation failed: {qr_reason}")
                if save_dataset:
                    _save_rejected_image(frame_bgr, boxes, phrases, None, "qr_validation_failed", self.rejected_dir)
                    self.rejected_samples += 1
                return None
        
        # Step 5: Box Selection
        selected_bbox = _pick_box_bbox(boxes, phrases, qr_points, frame_bgr.shape)
        if selected_bbox is None:
            _log_warning("Frame Processing", "No suitable box found")
            if save_dataset:
                _save_rejected_image(frame_bgr, boxes, phrases, None, "no_suitable_box", self.rejected_dir)
                self.rejected_samples += 1
            return None
        
        # Step 6: Segmentation
        if self.cfg.use_white_ring_seg:
            # Use white-ring segmentation
            from sections_a.a_white_ring import process_white_ring_segmentation
            mask, rect_pts, process_time = process_white_ring_segmentation(frame_bgr, self.cfg, selected_bbox)
        else:
            # Use AI-based segmentation
            if self.bg_removal is None:
                _log_error("Frame Processing", "Background removal model not initialized")
                return None
            
            mask_box = segment_box_by_boxprompt(self.bg_removal, frame_bgr, selected_bbox)
            if mask_box is None:
                _log_warning("Frame Processing", "Segmentation failed")
                return None
            
            # Point-prompt for fruits and MERGE
            mask = mask_box.copy()
            for phrase in phrases:
                if any(fruit in phrase.lower() for fruit in qr_items.keys()):
                    # Find center of fruit detection
                    fruit_boxes = [box for i, box in enumerate(boxes) if i < len(phrases) and phrase == phrases[i]]
                    if fruit_boxes:
                        box = fruit_boxes[0]
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        bx1, by1, bx2, by2 = selected_bbox
                        
                        m_obj = segment_object_by_point(self.bg_removal, frame_bgr, (cx, cy), box_hint=(bx1, by1, bx2, by2))
                        mask = (mask | m_obj).astype(np.uint8)
        
        # Step 7: Save to Dataset
        if save_dataset and not preview_only:
            # Generate box ID
            box_id = f"box_{self.processed_frames}_{int(time.time())}"
            
            # Create metadata
            metadata = {
                'timestamp': time.time(),
                'session_id': self.session_id,
                'supplier_id': self.supplier_id,
                'qr_data': qr_data,
                'qr_items': qr_items,
                'phrases': phrases,
                'selected_bbox': selected_bbox
            }
            
            # Add to dataset
            success = self.ds.add_sample(frame_bgr, mask, metadata, box_id, qr_items)
            if success:
                self.saved_samples += 1
                _log_success("Frame Processing", f"Saved sample {box_id}")
            else:
                self.rejected_samples += 1
                _log_warning("Frame Processing", f"Failed to save sample {box_id}")
        
        # Step 8: Create Visualization
        if return_both_visualizations:
            gdino_vis = _create_gdino_visualization(img_resized, boxes, logits, phrases)
            return {
                'gdino_visualization': gdino_vis,
                'selected_bbox': selected_bbox,
                'mask': mask,
                'qr_items': qr_items,
                'phrases': phrases
            }
        else:
            return {
                'selected_bbox': selected_bbox,
                'mask': mask,
                'qr_items': qr_items,
                'phrases': phrases
            }
    
    except Exception as e:
        _log_error("Frame Processing", e, f"Failed to process frame {self.processed_frames}")
        return None

def _create_gdino_visualization(img_resized, boxes_original, logits, phrases):
    """Create GroundingDINO visualization"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    if boxes_original is None or len(boxes_original) == 0:
        return img_resized
    
    vis = img_resized.copy()
    boxes_pixel = _to_pixel_xyxy(boxes_original, img_resized.shape[1], img_resized.shape[0])
    
    for i, (box, phrase, logit) in enumerate(zip(boxes_pixel, phrases, logits)):
        x1, y1, x2, y2 = map(int, box)
        
        # Color based on confidence
        confidence = float(logit)
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{phrase} ({confidence:.2f})", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    _log_info("Visualization", f"Created GroundingDINO visualization with {len(boxes_pixel)} detections")
    return vis

def _to_normalized_xyxy(boxes, img_w, img_h):
    """Convert pixel boxes to normalized coordinates"""
    if boxes is None or len(boxes) == 0:
        return []
    
    normalized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_norm = x1 / img_w
        y1_norm = y1 / img_h
        x2_norm = x2 / img_w
        y2_norm = y2 / img_h
        normalized_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_boxes
