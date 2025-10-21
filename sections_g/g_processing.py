import os
import cv2
import numpy as np
import torch
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Import log functions at module level
try:
    from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG
except ImportError:
    # Fallback if log functions not available yet
    def _log_info(context, message): print(f"[INFO] {context}: {message}")
    def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
    class CFG:
        device = "cpu"

def _pick_box_bbox(self, boxes, phrases, qr_points, img_shape):
    """Pick best box bbox from detections"""
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
        score = area * 0.001 + kw * 1000 + qr_iou * 500
        
        _log_info("Bbox Selection", f"Box {i}: phrase='{phrase}', area={area:.0f}, kw={kw}, qr_iou={qr_iou:.3f}, score={score:.1f}")
        
        if score > best:
            best, best_box = score, (x1, y1, x2, y2)
            _log_info("Bbox Selection", f"New best box: {best_box} (score={score:.1f})")

    if best_box is not None:
        _log_success("Bbox Selection", f"Selected box: {best_box} (score={best:.1f})")
    else:
        _log_warning("Bbox Selection", "No suitable box found")
    
    return best_box

def _save_rejected_image(self, frame_bgr, boxes, phrases, selected_bbox, reason):
    """Save rejected image with metadata"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_warning
        from sections_a.a_utils import ensure_dir
    except ImportError:
        # Fallback if log functions not available yet
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def ensure_dir(d): pass
    
    try:
        # Create rejected images directory
        rejected_dir = os.path.join(self.cfg.project_dir, self.cfg.rejected_images_dir)
        ensure_dir(rejected_dir)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"rejected_{reason}_{timestamp}.jpg"
        filepath = os.path.join(rejected_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, frame_bgr)
        
        # Save metadata
        meta_filepath = filepath.replace('.jpg', '.json')
        metadata = {
            'reason': reason,
            'timestamp': timestamp,
            'boxes': boxes.tolist() if boxes is not None else [],
            'phrases': phrases,
            'selected_bbox': list(selected_bbox) if selected_bbox is not None else None
        }
        
        import json
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        _log_warning("Rejected Image", f"Saved rejected image: {filename}")
        
    except Exception as e:
        _log_warning("Rejected Image", f"Failed to save rejected image: {e}")

def _to_pixel_xyxy(self, boxes, img_w, img_h):
    """Convert normalized boxes to pixel coordinates"""
    if boxes is None or len(boxes) == 0:
        return boxes
    
    # Ensure boxes is numpy array
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    elif not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    
    # Convert from normalized to pixel
    pixel_boxes = boxes.copy().astype(np.float32)
    pixel_boxes[:, [0, 2]] = pixel_boxes[:, [0, 2]] * img_w  # x coordinates
    pixel_boxes[:, [1, 3]] = pixel_boxes[:, [1, 3]] * img_h  # y coordinates
    
    return pixel_boxes

def _get_fruit_class_id(self, phrase: str, qr_items_dict: Dict[str, int]) -> int:
    """Get class ID for fruit based on phrase and QR items"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Map fruit names to class IDs
    fruit_class_mapping = {
        'apple': 1, 'banana': 2, 'orange': 3, 'mango': 4, 'tangerine': 5,
        'grape': 6, 'strawberry': 7, 'kiwi': 8, 'pineapple': 9, 'watermelon': 10,
        'lemon': 11, 'lime': 12, 'peach': 13, 'pear': 14, 'cherry': 15,
        'blueberry': 16, 'raspberry': 17, 'blackberry': 18, 'coconut': 19, 'avocado': 20
    }
    
    fruit_lower = phrase.lower().strip()
    class_id = fruit_class_mapping.get(fruit_lower, 21)  # Default to generic fruit
    
    _log_info("Fruit Class ID", f"'{phrase}' -> class ID {class_id}")
    return class_id

def _get_class_id_for_fruit(self, fruit_name: str) -> int:
    """Get class ID for fruit name"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Map fruit names to class IDs
    fruit_class_mapping = {
        'apple': 1, 'banana': 2, 'orange': 3, 'mango': 4, 'tangerine': 5,
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
    # Import helper functions
    try:
        from sections_e.e_qr_utils import parse_qr_payload, check_hand_detection, validate_qr_detection
        from sections_c.c_segmentation import segment_box_by_boxprompt, segment_object_by_point
    except ImportError:
        # Fallback if functions not available yet
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
        qr_payload = {} # Initialize qr_payload
        if qr_data:
            qr_payload = parse_qr_payload(qr_data)
            qr_items = qr_payload.get('fruits', {})
            _log_info("QR Detection", f"Detected QR with {len(qr_items)} fruit types")
        else:
            _log_info("QR Detection", "No QR code detected")
        
        # Step 2: GroundingDINO Detection
        boxes, logits, phrases, img_resized = self.gd.infer_two_stage(frame_bgr, qr_items)
        
        if boxes is None or len(boxes) == 0:
            _log_warning("Frame Processing", "No objects detected")
            if return_both_visualizations:
                return None, None, {}, None, None
            else:
                return None, {}, None, None
        
        # Step 3: Hand Detection Check
        hand_detected, hand_reason = check_hand_detection(phrases)
        if hand_detected:
            _log_warning("Frame Processing", f"Hand detected: {hand_reason}")
            if save_dataset:
                self._save_rejected_image(frame_bgr, boxes, phrases, None, "hand_detected")
                self.rejected_samples += 1
            if return_both_visualizations:
                return None, None, {}, None, None
            else:
                return None, {}, None, None
        
        # Step 4: QR Validation
        if qr_items:
            qr_valid, qr_reason = validate_qr_detection(qr_items, phrases)
            if not qr_valid:
                _log_warning("Frame Processing", f"QR validation failed: {qr_reason}")
                if save_dataset:
                    self._save_rejected_image(frame_bgr, boxes, phrases, None, "qr_validation_failed")
                    self.rejected_samples += 1
                # Return proper tuple format for both visualization modes
                if return_both_visualizations:
                    return None, None, {}, None, None
                else:
                    return None, {}, None, None
        
        # Step 5: Box Selection
        selected_bbox = self._pick_box_bbox(boxes, phrases, qr_points, frame_bgr.shape)
        if selected_bbox is None:
            _log_warning("Frame Processing", "No suitable box found")
            if save_dataset:
                self._save_rejected_image(frame_bgr, boxes, phrases, None, "no_suitable_box")
                self.rejected_samples += 1
            if return_both_visualizations:
                return None, None, {}, None, None
            else:
                return None, {}, None, None
        
        # Step 6: Segmentation
        if self.cfg.use_white_ring_seg:
            # Use white-ring segmentation
            from sections_a.a_white_ring import process_white_ring_segmentation
            mask, rect_pts, process_time = process_white_ring_segmentation(frame_bgr, self.cfg)
        else:
            # Use AI-based segmentation
            if self.bg_removal is None:
                _log_error("Frame Processing", "Background removal model not initialized")
                if return_both_visualizations:
                    return None, None, {}, None, None
                else:
                    return None, {}, None, None
            
            mask_box = segment_box_by_boxprompt(self.bg_removal, frame_bgr, selected_bbox)
            if mask_box is None:
                _log_warning("Frame Processing", "Segmentation failed")
                if return_both_visualizations:
                    return None, None, {}, None, None
                else:
                    return None, {}, None, None
            
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
            
            # Extract segment_square_corners (4 points) from rect_pts
            segment_square_corners = None
            if rect_pts is not None and len(rect_pts) >= 4:
                try:
                    # Use minAreaRect to get 4 corner points of bounding rectangle
                    rect_pts_np = np.array(rect_pts, dtype=np.float32)
                    
                    # Handle nested format [[[x,y]]]
                    if rect_pts_np.ndim == 3 and rect_pts_np.shape[1] == 1:
                        rect_pts_np = rect_pts_np[:, 0, :]
                    
                    min_rect = cv2.minAreaRect(rect_pts_np)
                    box_points = cv2.boxPoints(min_rect)  # Returns 4 corner points
                    
                    # Clip corners to image bounds
                    H, W = frame_bgr.shape[:2]
                    box_points[:, 0] = np.clip(box_points[:, 0], 0, W - 1)
                    box_points[:, 1] = np.clip(box_points[:, 1], 0, H - 1)
                    
                    segment_square_corners = box_points.tolist()
                    _log_info("Segment Square", f"Extracted 4 corners from {len(rect_pts)} segment points (clipped to image bounds)")
                except Exception as e:
                    _log_warning("Segment Square", f"Failed to extract square corners: {e}")
            
            # Create metadata
            metadata = {
                'timestamp': time.time(),
                'session_id': self.session_id,
                'supplier_id': self.supplier_id,
                'qr_data': qr_data,
                'qr_items': qr_items,
                'qr_corners': qr_points.tolist() if qr_points is not None else None,
                'segment_corners': rect_pts.tolist() if rect_pts is not None else None,
                'segment_square_corners': segment_square_corners,  # NEW: 4 corner points
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
        
        # Step 8: Create Visualization and Return Results
        if return_both_visualizations:
            gdino_vis = self._create_gdino_visualization(img_resized, boxes, logits, phrases)
            
            # Create metadata
            meta = {
                'qr': {
                    'data': qr_data,
                    'parsed': qr_payload,
                    'items': qr_items,
                    'corners': qr_points.tolist() if qr_points is not None else None
                },
                'detection': {
                    'boxes': boxes.tolist() if boxes is not None else [],
                    'phrases': phrases,
                    'logits': logits.tolist() if logits is not None else []
                },
                'selected_bbox': list(selected_bbox) if selected_bbox is not None else None,
                'segment_corners': rect_pts.tolist() if rect_pts is not None else None,
                'mask_shape': mask.shape if mask is not None else None
            }
            
            # Define paths (placeholder for now)
            img_path = f"processed_frame_{self.processed_frames}.jpg"
            lab_path = f"processed_frame_{self.processed_frames}.txt"
            
            # Return tuple format expected by handle_capture
            return gdino_vis, mask, meta, img_path, lab_path
        else:
            # Create metadata for single visualization
            meta = {
                'qr': {
                    'data': qr_data,
                    'parsed': qr_payload,
                    'items': qr_items,
                    'corners': qr_points.tolist() if qr_points is not None else None
                },
                'detection': {
                    'boxes': boxes.tolist() if boxes is not None else [],
                    'phrases': phrases,
                    'logits': logits.tolist() if logits is not None else []
                },
                'selected_bbox': list(selected_bbox) if selected_bbox is not None else None,
                'segment_corners': rect_pts.tolist() if rect_pts is not None else None,
                'mask_shape': mask.shape if mask is not None else None
            }
            
            # Define paths (placeholder for now)
            img_path = f"processed_frame_{self.processed_frames}.jpg"
            lab_path = f"processed_frame_{self.processed_frames}.txt"
            
            return mask, meta, img_path, lab_path
    
    except Exception as e:
        _log_error("Frame Processing", e, f"Failed to process frame {self.processed_frames}")
        # Return tuple instead of None to avoid unpacking errors
        if return_both_visualizations:
            return None, None, {}, None, None
        else:
            return None, {}, None, None

def _create_gdino_visualization(self, img_resized, boxes_original, logits, phrases):
    """Create GroundingDINO visualization - COPY Y CHANG từ NCC_PROCESS.py"""
    try:
        # Convert BGR to RGB for GroundingDINO (COPY Y CHANG từ NCC_PROCESS.py)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        if boxes_original is not None and len(boxes_original) > 0:
            _log_info("GDINO Visualization", f"Creating visualization with {len(boxes_original)} boxes")
            
            # Sử dụng annotate() để có visualization đẹp như NCC_PROCESS.py
            try:
                # Sử dụng cùng logic như NCC_PROCESS.py để có màu đẹp
                annotated_img = self.gd._annotate(
                    image_source=img_rgb,
                    boxes=boxes_original,
                    logits=logits,
                    phrases=phrases
                )
                # Đảm bảo annotated_img là RGB (GroundingDINO có thể trả về BGR)
                if annotated_img is not None and len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                    vis_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                else:
                    vis_rgb = annotated_img if annotated_img is not None else img_rgb
                
                _log_success("GDINO Visualization", "Successfully created annotated image using _annotate()")
                
            except Exception as e:
                _log_warning("GDINO Visualization", f"Annotate failed, using manual visualization: {e}")
                # Fallback: manual visualization - chỉ hiển thị tất cả detections
                vis_rgb = img_rgb.copy()
                H, W = img_rgb.shape[:2]
                boxes_pix = self._to_pixel_xyxy(boxes_original, W, H)
                
                for i, (box, logit, phrase) in enumerate(zip(boxes_pix, logits, phrases)):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    # Tất cả boxes đều cùng màu tím (không phân biệt selected)
                    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (128, 0, 128), 2)
                    cv2.putText(vis_rgb, phrase, (x1, max(0, y1-8)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
        else:
            vis_rgb = img_rgb
            _log_warning("GDINO Visualization", "No boxes provided")
            
        return vis_rgb
        
    except Exception as e:
        _log_error("GDINO Visualization", e, "Failed to create GroundingDINO visualization")
        # Fallback to original image
        vis_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return vis_rgb

def _to_normalized_xyxy(self, boxes, img_w, img_h):
    """Convert pixel boxes to normalized coordinates"""
    if boxes is None or len(boxes) == 0:
        return []
    
    normalized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        norm_box = [x1/img_w, y1/img_h, x2/img_w, y2/img_h]
        normalized_boxes.append(norm_box)
    
    return np.array(normalized_boxes)
