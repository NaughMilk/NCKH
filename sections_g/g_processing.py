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
    from sections_e.e_qr_utils import validate_yolo_label
except ImportError:
    # Fallback if log functions not available yet
    def _log_info(context, message): print(f"[INFO] {context}: {message}")
    def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
    def validate_yolo_label(class_id, x_center, y_center, width, height): return True  # fallback
    class CFG:
        device = "cpu"

def _pick_box_bbox(self, boxes, phrases, qr_points, img_shape):
    """Pick best box bbox from detections"""
    if boxes is None or len(boxes) == 0:
        return None
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, device=CFG.device)
    
    # GPU memory optimization
    if CFG.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
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
        
        # Boxes expected in pixel coordinates here
        
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
        
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
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

def _to_pixel_xyxy(self, boxes, img_w, img_h, scale_x=1.0, scale_y=1.0):
    """Convert GroundingDINO normalized (cx, cy, w, h) -> pixel (x1, y1, x2, y2) on original image."""
    if boxes is None or len(boxes) == 0:
        return boxes
    
    # Ensure boxes is torch tensor for clamp operations
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    elif not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    
    # GroundingDINO outputs normalized (cx, cy, w, h) on resized image
    # Convert to original image coordinates
    cx = boxes[:, 0] * img_w * scale_x
    cy = boxes[:, 1] * img_h * scale_y
    w  = boxes[:, 2] * img_w * scale_x
    h  = boxes[:, 3] * img_h * scale_y
    
    # Convert to (x1, y1, x2, y2) format
    x1 = (cx - w / 2).clamp(0, img_w - 1)  # Clamp negative coordinates
    y1 = (cy - h / 2).clamp(0, img_h - 1)  # Clamp negative coordinates
    x2 = (cx + w / 2).clamp(0, img_w - 1)  # Clamp negative coordinates
    y2 = (cy + h / 2).clamp(0, img_h - 1)  # Clamp negative coordinates
    
    # Handle swapped coordinates (x1 > x2 or y1 > y2)
    temp_x = torch.clone(x1)
    temp_y = torch.clone(y1)
    x1 = torch.where(x1 > x2, x2, x1)
    x2 = torch.where(temp_x > x2, temp_x, x2)
    y1 = torch.where(y1 > y2, y2, y1)
    y2 = torch.where(temp_y > y2, temp_y, y2)
    
    # Stack and convert back to numpy
    pixel_boxes = torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()
    
    return pixel_boxes

def _xyxy_to_yolo(self, x1, y1, x2, y2, W, H):
    """Convert xyxy pixel coordinates to normalized YOLO (xc, yc, w, h)."""
    try:
        # ensure ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # compute normalized center and size
        xc = ((x1 + x2) / 2.0) / float(max(W, 1))
        yc = ((y1 + y2) / 2.0) / float(max(H, 1))
        w  = (x2 - x1) / float(max(W, 1))
        h  = (y2 - y1) / float(max(H, 1))
        # clamp to [0, 1]
        xc = 0.0 if xc < 0.0 else 1.0 if xc > 1.0 else xc
        yc = 0.0 if yc < 0.0 else 1.0 if yc > 1.0 else yc
        w  = 0.0 if w  < 0.0 else 1.0 if w  > 1.0 else w
        h  = 0.0 if h  < 0.0 else 1.0 if h  > 1.0 else h
        return xc, yc, w, h
    except Exception:
        # fall back safe values
        return 0.5, 0.5, 0.0, 0.0

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
    
    # Update dataset class names (support multiple classes)
    # Ensure 'plastic box' is always first
    if "plastic box" not in self.ds.class_names:
        self.ds.class_names.insert(0, "plastic box")
    
    # Add all detected fruit classes
    for fruit_class in self.detected_classes:
        if fruit_class not in self.ds.class_names:
            self.ds.class_names.append(fruit_class)
    
    self.ds.class_id_counter = len(self.ds.class_names)
    _log_info("Class Update", f"Updated to {len(self.ds.class_names)} classes: {self.ds.class_names}")
    
    # Update YAML files
    self.ds._create_initial_yaml_files()
    
    _log_success("Class Update", f"Updated class names: {self.ds.class_names}")
    _log_info("Class Update", f"Total classes: {len(self.ds.class_names)}")

def process_frame(self, frame_bgr: np.ndarray, preview_only: bool = False, save_dataset: bool = True, return_both_visualizations: bool = False):
    """Process single frame through the pipeline"""
    # GPU memory optimization at start
    if CFG.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        qr_items = {}
        if qr_data:
            qr_payload = parse_qr_payload(qr_data)
            _log_info("QR Detection", f"Parsed QR payload: {qr_payload}")
            
            # Load QR metadata from JSON file (like NCC_PIPELINE_NEW.py)
            qr_id = qr_payload.get("_qr")
            if qr_id:
                try:
                    import json  # Import json here
                    # Use self.cfg instead of importing CFG
                    qr_meta_dir = os.path.join(self.cfg.project_dir, "qr_meta")
                    qr_meta_path = os.path.join(qr_meta_dir, qr_id, f"{qr_id}.json")
                    
                    if os.path.exists(qr_meta_path):
                        with open(qr_meta_path, 'r', encoding='utf-8') as f:
                            qr_meta_data = json.load(f)
                            qr_items = qr_meta_data.get("fruits", {})
                            _log_success("QR Detection", f"Loaded QR metadata for ID '{qr_id}' with {len(qr_items)} fruit types")
                    else:
                        _log_warning("QR Detection", f"No QR metadata file found for ID '{qr_id}' at {qr_meta_path}")
                except Exception as e:
                    _log_error("QR Detection", e, f"Failed to load QR metadata for ID '{qr_id}'")
            else:
                _log_warning("QR Detection", "No QR ID found in payload")
            
            if not qr_items:
                _log_warning("QR Detection", "No QR items (fruits) found after processing QR data")
                # Skip image if QR decoded but no fruits metadata
                _log_warning("Frame Processing", "Skipping image: QR decoded but no fruits metadata")
                if return_both_visualizations:
                    return None, None, None, {}, None, None
                else:
                    return None, None, {}, None, None
            else:
                _log_success("QR Detection", f"Detected QR with {len(qr_items)} fruit types: {list(qr_items.keys())}")
        else:
            _log_info("QR Detection", "No QR code detected")
            # Skip image if no QR code detected
            _log_warning("Frame Processing", "Skipping image: No QR code detected")
            if return_both_visualizations:
                return None, None, None, {}, None, None
            else:
                return None, None, {}, None, None
        
        # Step 2: GroundingDINO Detection
        boxes, logits, phrases, img_resized = self.gd.infer_two_stage(frame_bgr, qr_items)
        
        # Calculate scale factor for coordinate conversion
        h_orig, w_orig = frame_bgr.shape[:2]
        h_resized, w_resized = img_resized.shape[:2]
        scale_x = w_orig / w_resized
        scale_y = h_orig / h_resized
        
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
        # FIXED: Convert normalized coordinates to pixel coordinates for bbox selection
        if boxes is not None and len(boxes) > 0:
            H_resized, W_resized = img_resized.shape[:2]
            boxes_pix = self._to_pixel_xyxy(boxes, W_resized, H_resized, 1.0, 1.0)
            selected_bbox = self._pick_box_bbox(boxes_pix, phrases, qr_points, img_resized.shape)
        else:
            selected_bbox = None
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
            # FIXED: Use img_resized instead of frame_bgr to match with dataset
            mask, rect_pts, process_time = process_white_ring_segmentation(img_resized, self.cfg)
        else:
            # Use AI-based segmentation
            if self.bg_removal is None:
                _log_error("Frame Processing", "Background removal model not initialized")
                if return_both_visualizations:
                    return None, None, {}, None, None
                else:
                    return None, {}, None, None
            
            # FIXED: Use img_resized instead of frame_bgr to match with dataset
            mask_box = segment_box_by_boxprompt(self.bg_removal, img_resized, selected_bbox)
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
                        
                        # FIXED: Use img_resized instead of frame_bgr to match with dataset
                        m_obj = segment_object_by_point(self.bg_removal, img_resized, (cx, cy), box_hint=(bx1, by1, bx2, by2))
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
                    # FIXED: Use img_resized instead of frame_bgr to match with dataset
                    H, W = img_resized.shape[:2]
                    box_points[:, 0] = np.clip(box_points[:, 0], 0, W - 1)
                    box_points[:, 1] = np.clip(box_points[:, 1], 0, H - 1)
                    
                    segment_square_corners = box_points.tolist()
                    _log_info("Segment Square", f"Extracted 4 corners from {len(rect_pts)} segment points (clipped to image bounds)")
                except Exception as e:
                    _log_warning("Segment Square", f"Failed to extract square corners: {e}")
            
            # DYNAMIC: Create class names from QR data (TRULY DYNAMIC - no minimum constraint)
            # Get fruit names from QR data (support multiple fruits)
            if qr_items and len(qr_items) > 0:
                # Add all fruit types from QR to class names
                for fruit_name in qr_items.keys():
                    if fruit_name not in self.ds.class_names:
                        self.ds.class_names.append(fruit_name)
                        self.ds.class_id_counter += 1
                        _log_info("Frame Processing", f"Added new fruit class: '{fruit_name}'")
            else:
                # Only add default fruit if we have no QR data AND no existing classes
                if len(self.ds.class_names) == 0:
                    self.ds.class_names.append("fruit")
                    self.ds.class_id_counter += 1
                    _log_info("Frame Processing", f"Added default fruit class (no QR data)")
            
            _log_info("Frame Processing", f"DYNAMIC class names: {self.ds.class_names} (total: {len(self.ds.class_names)})")

            # Create YOLO detections based on selected plastic box and fruits inside it
            yolo_detections = []
            try:
                # Use ORIGINAL image dimensions for YOLO format
                H_orig, W_orig = frame_bgr.shape[:2]
                H_resized, W_resized = img_resized.shape[:2]
                
                # Calculate scale factors for coordinate conversion
                scale_x = W_orig / W_resized
                scale_y = H_orig / H_resized
                
                # Use boxes_pix already converted for bbox selection
                if boxes is not None and len(boxes) > 0:
                    # boxes_pix already converted above for bbox selection
                    pass
                else:
                    boxes_pix = []
                
                # Helper: compute IoU and area
                def _area(x1, y1, x2, y2):
                    return max(0, x2 - x1) * max(0, y2 - y1)
                def _iou(a, b):
                    ax1, ay1, ax2, ay2 = a
                    bx1, by1, bx2, by2 = b
                    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                    inter = _area(ix1, iy1, ix2, iy2)
                    u = _area(*a) + _area(*b) - inter
                    return inter / u if u > 0 else 0.0

                # 1) Write exactly one class-0 using selected_bbox
                if selected_bbox is not None:
                    sbx1, sby1, sbx2, sby2 = map(int, selected_bbox)
                    # Convert resized coordinates to original image coordinates
                    sbx1_orig = sbx1 * scale_x
                    sby1_orig = sby1 * scale_y
                    sbx2_orig = sbx2 * scale_x
                    sby2_orig = sby2 * scale_y
                    
                    # Convert to YOLO format using ORIGINAL image dimensions
                    x_center = (sbx1_orig + sbx2_orig) / 2.0 / W_orig
                    y_center = (sby1_orig + sby2_orig) / 2.0 / H_orig
                    width = (sbx2_orig - sbx1_orig) / W_orig
                    height = (sby2_orig - sby1_orig) / H_orig
                    if validate_yolo_label(0, x_center, y_center, width, height):
                        yolo_detections.append({
                            "class_id": 0,
                            "class_name": "plastic box",
                            "bbox": [x_center, y_center, width, height],
                        })
                        _log_success("YOLO Convert", f"Added plastic box: class=0, bbox=({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
                
                # 2) Fruits: boxes inside selected box and much smaller
                total_needed = 0
                if isinstance(qr_items, dict):
                    try:
                        total_needed = sum(int(v) for v in qr_items.values())
                    except Exception:
                        total_needed = 0
                
                # 2) Fruits only: Skip plastic box, only process tangerines
                if boxes_pix is not None and len(boxes_pix) > 0:
                    _log_info("YOLO Detection", f"Processing {len(boxes_pix)} GroundingDINO detections for YOLO format")
                    for i, b in enumerate(boxes_pix):
                        # boxes_pix contains resized pixel coordinates [x1, y1, x2, y2]
                        x1, y1, x2, y2 = b.tolist()
                        
                        # Convert resized coordinates to original image coordinates
                        x1_orig = x1 * scale_x
                        y1_orig = y1 * scale_y
                        x2_orig = x2 * scale_x
                        y2_orig = y2 * scale_y
                        
                        # Convert to YOLO format using ORIGINAL image dimensions
                        x_center = (x1_orig + x2_orig) / 2.0 / W_orig
                        y_center = (y1_orig + y2_orig) / 2.0 / H_orig
                        width = (x2_orig - x1_orig) / W_orig
                        height = (y2_orig - y1_orig) / H_orig
                        
                        ph = phrases[i].lower() if i < len(phrases) else ""
                        
                        # Skip plastic box (already added above), only process fruits
                        if any(k in ph for k in ("box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate")):
                            _log_info("YOLO Convert", f"Skipping plastic box detection {i+1}: phrase='{ph}'")
                            continue
                        
                        # Dynamic class_id based on fruit type
                        class_id = self.ds._get_or_add_class_id(ph)
                        
                        _log_info("YOLO Convert", f"Detection {i+1}: phrase='{ph}', class_id={class_id}")
                        _log_info("YOLO Convert", f"  Resized coords: ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        _log_info("YOLO Convert", f"  Original coords: ({x1_orig:.1f},{y1_orig:.1f},{x2_orig:.1f},{y2_orig:.1f})")
                        _log_info("YOLO Convert", f"  YOLO norm: ({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
                        
                        # Validate and add to yolo_detections
                        if validate_yolo_label(class_id, x_center, y_center, width, height):
                            yolo_detections.append({
                                "class_id": class_id,
                                "class_name": ph,  # Use actual phrase as class_name
                                "bbox": [x_center, y_center, width, height],
                                "phrase": ph
                            })
                            _log_success("YOLO Convert", f"Added {ph} {i+1}: class={class_id}, bbox=({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
                        else:
                            _log_warning("YOLO Convert", f"Skipped invalid {ph} {i+1}: class_id={class_id}, bbox=({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})")
                
                _log_info("YOLO Detection", f"Final detections: {len(yolo_detections)} (box + fruits)")
            except Exception as e:
                _log_warning("Frame Processing", f"Failed to prepare YOLO detections: {e}")
                import traceback
                _log_warning("Frame Processing", f"Traceback: {traceback.format_exc()}")

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
            
            # Add to dataset with YOLO detections (EXACTLY like NCC_PIPELINE_NEW.py)
            # FIXED: Use frame_bgr (original image) for dataset, coordinates already converted
            success = self.ds.add_sample(frame_bgr, mask, metadata, box_id, yolo_detections=yolo_detections)
            if success:
                self.saved_samples += 1
                _log_success("Frame Processing", f"Saved sample {box_id}")
            else:
                self.rejected_samples += 1
                _log_warning("Frame Processing", f"Failed to save sample {box_id}")
        
        # Step 8: Create Visualization and Return Results
        if return_both_visualizations:
            gdino_vis = self._create_gdino_visualization(img_resized, boxes, logits, phrases)
            
            # Fallback nếu gdino_vis bị None
            if gdino_vis is None:
                _log_warning("Frame Processing", "gdino_vis is None, using original image as fallback")
                gdino_vis = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Create segmentation visualization (like NCC_PIPELINE_NEW.py)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            vis = img_rgb.copy()
            if mask is not None and mask.any():
                mask_colored = np.zeros_like(img_rgb)
                # Use mask > 0 to handle both [0,1] and [0,255] formats
                mask_colored[mask > 0] = [0, 255, 0]  # Green for mask
                vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
                
                # Draw contours
                if mask.max() <= 1:
                    mask_contour = (mask * 255).astype(np.uint8)
                else:
                    mask_contour = mask.astype(np.uint8)
                
                contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            
            # Create metadata
            meta = {
                'qr_detected': qr_data is not None,
                'gdino_detections': len(boxes) if boxes is not None else 0,
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
            
            # GPU memory cleanup before return
            if CFG.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return tuple format: (original_img, gdino_vis, seg_vis, meta, img_path, lab_path)
            return img_resized, gdino_vis, vis, meta, img_path, lab_path
        else:
            # Create metadata for single visualization
            meta = {
                'qr_detected': qr_data is not None,
                'gdino_detections': len(boxes) if boxes is not None else 0,
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
            
            # GPU memory cleanup before return
            if CFG.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return single visualization (original + segmentation)
            return img_resized, img_resized, vis, meta, img_path, lab_path
    
    except Exception as e:
        _log_error("Frame Processing", e, f"Failed to process frame {self.processed_frames}")
        # Return tuple instead of None to avoid unpacking errors
        if return_both_visualizations:
            return None, None, None, {}, None, None
        else:
            return None, None, None, {}, None, None

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
                # Sử dụng signature như NCC_PIPELINE_NEW.py
                boxes_pix = self._to_pixel_xyxy(boxes_original, W, H)
                
                for i, (box, logit, phrase) in enumerate(zip(boxes_pix, logits, phrases)):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    # Tất cả boxes đều cùng màu xanh lá (dễ nhìn hơn)
                    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_rgb, phrase, (x1, max(0, y1-8)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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

def cleanup_orphaned_files(dataset_root):
    """
    Clean up files that don't have corresponding metadata JSON
    Only keep files that have valid metadata
    """
    import os
    import json
    from pathlib import Path
    
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning
    except ImportError:
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    _log_info("Data Cleanup", f"Starting orphaned file cleanup for dataset: {dataset_root}")
    
    # Get all metadata files
    meta_dir = os.path.join(dataset_root, "meta")
    if not os.path.exists(meta_dir):
        _log_warning("Data Cleanup", f"Meta directory not found: {meta_dir}")
        return
    
    meta_files = [f for f in os.listdir(meta_dir) if f.endswith('.json')]
    valid_box_ids = [f.replace('.json', '') for f in meta_files]
    
    _log_info("Data Cleanup", f"Found {len(valid_box_ids)} valid metadata files")
    
    if len(valid_box_ids) == 0:
        _log_warning("Data Cleanup", "No metadata files found, skipping cleanup")
        return
    
    # Check each folder
    folders_to_check = ['images/train', 'images/val', 'labels/train', 'labels/val', 'masks/train', 'masks/val']
    total_removed = 0
    
    for folder in folders_to_check:
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.exists(folder_path):
            _log_info("Data Cleanup", f"Folder not found: {folder_path}")
            continue
            
        files_in_folder = os.listdir(folder_path)
        removed_count = 0
        
        for file in files_in_folder:
            # Extract box_id from filename
            if folder.startswith('images'):
                box_id = file.replace('.jpg', '')
            elif folder.startswith('labels'):
                box_id = file.replace('.txt', '')
            elif folder.startswith('masks'):
                box_id = file.replace('.png', '')
            else:
                continue
                
            # Check if this box_id has metadata
            if box_id not in valid_box_ids:
                file_path = os.path.join(folder_path, file)
                try:
                    os.remove(file_path)
                    removed_count += 1
                    total_removed += 1
                    _log_info("Data Cleanup", f"Removed orphaned file: {file}")
                except Exception as e:
                    _log_warning("Data Cleanup", f"Failed to remove {file_path}: {e}")
        
        if removed_count > 0:
            _log_info("Data Cleanup", f"Folder {folder}: Removed {removed_count} orphaned files")
    
    _log_success("Data Cleanup", f"Cleanup completed! Removed {total_removed} orphaned files")

def validate_dataset_consistency(dataset_root):
    """Validate that all files have corresponding metadata"""
    import os
    
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    _log_info("Data Validation", f"Validating dataset consistency: {dataset_root}")
    
    meta_dir = os.path.join(dataset_root, "meta")
    if not os.path.exists(meta_dir):
        _log_warning("Data Validation", f"Meta directory not found: {meta_dir}")
        return []
    
    meta_files = [f.replace('.json', '') for f in os.listdir(meta_dir) if f.endswith('.json')]
    _log_info("Data Validation", f"Found {len(meta_files)} metadata files")
    
    issues = []
    
    # Check each split
    for split in ['train', 'val']:
        for data_type in ['images', 'labels', 'masks']:
            folder_path = os.path.join(dataset_root, data_type, split)
            if not os.path.exists(folder_path):
                continue
                
            files = os.listdir(folder_path)
            for file in files:
                box_id = file.split('.')[0]
                if box_id not in meta_files:
                    issues.append(f"Orphaned {data_type}/{split}/{file}")
    
    if issues:
        _log_warning("Data Validation", f"Found {len(issues)} consistency issues")
        for issue in issues[:10]:  # Show first 10 issues
            _log_warning("Data Validation", f"  - {issue}")
        if len(issues) > 10:
            _log_warning("Data Validation", f"  ... and {len(issues) - 10} more issues")
    else:
        _log_info("Data Validation", "Dataset is consistent - no orphaned files found")
    
    return issues
