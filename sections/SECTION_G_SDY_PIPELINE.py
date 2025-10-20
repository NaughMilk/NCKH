# ========================= SECTION G: SDY PIPELINE ========================= #
# ========================= SECTION G: SDY PIPELINE ========================= #

import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path



import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
# ========================= SECTION G: SDYPIPELINE (TRAIN SDY + U²-NET) ========================= #

class SDYPipeline:
    """Main pipeline for dataset creation and training"""
    def __init__(self, cfg: Config, session_id: str = None, supplier_id: str = None):
        self.cfg = cfg
        self.session_id = session_id
        self.supplier_id = supplier_id
        self.qr = QR()
        self.gd = GDINO(cfg)
        
        # Initialize segmentation model based on config
        if cfg.use_white_ring_seg:
            _log_info("Pipeline Init", "Using Enhanced White-ring segmentation - no AI model needed")
            self.bg_removal = None  # White-ring doesn't need AI model
        else:
            _log_info("Pipeline Init", f"Using {cfg.bg_removal_model} for segmentation")
            self.bg_removal = BGRemovalWrap(cfg)
        
        self.ds = DSYDataset(cfg, session_id, supplier_id)
        # FIXED: Set generic fruit ID to match dataset
        self.generic_fruit_id = len(self.ds.class_names) - 1  # 21
        
        # Create rejected images directory
        self.rejected_dir = os.path.join(cfg.project_dir, cfg.rejected_images_dir)
        ensure_dir(self.rejected_dir)
        _log_info("Pipeline Init", f"Created rejected images directory: {self.rejected_dir}")
        
        # FIXED: Use dataset as single source of truth for class management
        # No need to copy references - always use self.ds directly
        
        # CRITICAL: Log session information for complete dataset tracking
        _log_success("Pipeline Init", f"Session: {self.ds.session_id}")
        _log_success("Pipeline Init", f"YOLO dataset: {self.ds.yolo_root}")
        _log_success("Pipeline Init", f"U²-Net dataset: {self.ds.u2net_root}")
        if supplier_id:
            _log_success("Pipeline Init", f"Supplier: {supplier_id}")
    
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
    
    def _save_rejected_image(self, frame_bgr, boxes, phrases, selected_bbox, reason):
        """Save rejected image with bbox visualization for debugging"""
        try:
            # Create visualization
            vis = frame_bgr.copy()
            
            # Draw all detected bboxes
            if boxes is not None and len(boxes) > 0:
                boxes_pixel = self._to_pixel_xyxy(boxes, frame_bgr.shape[1], frame_bgr.shape[0])
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
            filepath = os.path.join(self.rejected_dir, filename)
            cv2.imwrite(filepath, vis)
            
            _log_info("Rejected Image", f"Saved rejected image: {filepath}")
            
        except Exception as e:
            _log_error("Rejected Image", e, f"Failed to save rejected image: {reason}")
    
    def _to_pixel_xyxy(self, boxes_tensor, img_w, img_h):
        """Convert GroundingDINO normalized (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
        b = boxes_tensor.detach().cpu().float().clone()
        if b.numel() == 0:
            return b
        
        _log_info("Coordinate Conversion", f"Input boxes shape: {b.shape}, img_w={img_w}, img_h={img_h}")
        _log_info("Coordinate Conversion", f"Input boxes range: [{b.min():.3f}, {b.max():.3f}]")
        if len(b) > 0:
            _log_info("Coordinate Debug", f"Raw box 0 from GroundingDINO: {b[0].tolist()}")
            if len(b) > 1:
                _log_info("Coordinate Debug", f"Raw box 1 from GroundingDINO: {b[1].tolist()}")
        
        # GroundingDINO outputs normalized (cx, cy, w, h)
        cx = b[:, 0] * img_w
        cy = b[:, 1] * img_h
        w  = b[:, 2] * img_w
        h  = b[:, 3] * img_h

        x1 = (cx - w / 2).clamp(0, img_w - 1)
        y1 = (cy - h / 2).clamp(0, img_h - 1)
        x2 = (cx + w / 2).clamp(0, img_w - 1)
        y2 = (cy + h / 2).clamp(0, img_h - 1)

        out = torch.stack([x1, y1, x2, y2], dim=1)
        _log_info("Coordinate Conversion", f"Output boxes range: [{out.min():.1f}, {out.max():.1f}]")
        if len(out) > 0:
            _log_info("Coordinate Debug", f"Converted box 0: {out[0].tolist()}")
            if len(out) > 1:
                _log_info("Coordinate Debug", f"Converted box 1: {out[1].tolist()}")
        return out
    
    def _get_fruit_class_id(self, phrase: str, qr_items_dict: Dict[str, int]) -> int:
        """Map fruit phrase to fixed class ID 1 based on QR items; 0 for box"""
        phrase_lower = phrase.lower().strip()
        _log_info("Debug Classes", f"_get_fruit_class_id called with phrase: '{phrase}'")
        
        # Always map box/container to class 0
        box_keywords = ["box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate"]
        if any(keyword in phrase_lower for keyword in box_keywords):
            return 0

        # Determine fruit name from QR items (fallback to original phrase)
        target_name = None
        if qr_items_dict:
            for item in qr_items_dict.keys():
                if item.lower() in phrase_lower or phrase_lower in item.lower():
                    target_name = item
                    break
            if target_name is None:
                target_name = list(qr_items_dict.keys())[0]
        else:
            target_name = phrase

        normalized = self.ds._normalize_class_name(target_name)

        # Ensure class names have exactly two entries: ["plastic box", normalized]
        if len(self.ds.class_names) == 0:
            self.ds.class_names.append("plastic box")
        if len(self.ds.class_names) == 1:
            self.ds.class_names.append(normalized)
        else:
            self.ds.class_names[1] = normalized
        
        self.ds.detected_classes.add(normalized)
        return 1
    
    def _get_class_id_for_fruit(self, fruit_name: str) -> int:
        """Get class ID for specific fruit name using dynamic classes"""
        # Check if it's a box/container first
        fruit_name_lower = fruit_name.lower().strip()
        box_keywords = ["box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate"]
        if any(keyword in fruit_name_lower for keyword in box_keywords):
            _log_info("Dynamic Classes", f"Mapping '{fruit_name}' to class 0 (plastic box)")
            return 0  # box class
        
        # FIXED: Normalize class name to avoid duplicates
        normalized_fruit_name = self.ds._normalize_class_name(fruit_name)
        _log_info("Debug Classes", f"Normalized '{fruit_name}' to '{normalized_fruit_name}'")
        
        # Check if normalized class already exists
        if normalized_fruit_name not in self.ds.detected_classes:
            self.ds.detected_classes.add(normalized_fruit_name)
            self.ds.class_names.append(normalized_fruit_name)
            class_id = self.ds.class_id_counter
            self.ds.class_id_counter += 1
            _log_info("Dynamic Classes", f"Added new fruit class: '{normalized_fruit_name}' with ID {class_id}")
            _log_info("Dynamic Classes", f"Dataset now has {len(self.ds.class_names)} classes: {self.ds.class_names}")
        else:
            class_id = self.ds.class_names.index(normalized_fruit_name)
            _log_info("Dynamic Classes", f"Using existing fruit class: '{normalized_fruit_name}' with ID {class_id}")
        
        return class_id
    
    def update_class_names(self):
        """Update class names and regenerate YAML files - CHỈ TẠO YAML KHI CẦN THIẾT"""
        # FIXED: Use dataset as single source of truth
        # Ensure "plastic box" is always at index 0
        if "plastic box" not in self.ds.class_names:
            self.ds.class_names.insert(0, "plastic box")
            _log_info("Dynamic Classes", "Ensured 'plastic box' is at index 0")
        
        # FIXED: Chỉ tạo YAML khi có ít nhất 2 classes (plastic box + ít nhất 1 fruit)
        if len(self.ds.class_names) >= 2:  # Có ít nhất plastic box + 1 fruit
            _log_info("Dynamic Classes", f"Final class names: {self.ds.class_names}")
            _log_info("Dynamic Classes", f"Total classes: {len(self.ds.class_names)} (plastic box + {len(self.ds.class_names)-1} fruits)")
            
            # FIXED: Chỉ ghi YAML nếu chưa tồn tại hoặc cần cập nhật classes
            yaml_path = os.path.join(self.ds.yolo_root, "data.yaml")
            should_write_yaml = True
            
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Kiểm tra xem YAML đã có multi-class chưa
                        if 'nc: 2' in content or 'nc: 3' in content or 'nc: 4' in content:
                            # YAML đã có multi-class, chỉ cập nhật nếu classes thay đổi
                            existing_classes = []
                            names_match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
                            if names_match:
                                names_str = names_match.group(1)
                                existing_classes = [n.strip().strip("'\"") for n in names_str.split(',') if n.strip()]
                            
                            # Chỉ ghi lại nếu classes thực sự thay đổi
                            if set(existing_classes) == set(self.ds.class_names):
                                should_write_yaml = False
                                _log_info("Dynamic Classes", f"YAML already up-to-date with {len(existing_classes)} classes")
                except Exception as e:
                    _log_warning("Dynamic Classes", f"Could not check existing YAML: {e}")
            
            if should_write_yaml:
                self.ds.write_yaml()  # Regenerate YAML with updated classes
                _log_success("Dynamic Classes", f"Updated YAML with {len(self.ds.class_names)} classes")
            else:
                _log_info("Dynamic Classes", f"Skipped YAML update - no changes needed")
        else:
            _log_warning("Dynamic Classes", f"Not enough classes ({len(self.ds.class_names)}) to create YAML. Need at least 2 classes (plastic box + fruits)")
            _log_info("Dynamic Classes", f"Current classes: {self.ds.class_names}")
    
    def process_frame(self, frame_bgr: np.ndarray, preview_only: bool = False, save_dataset: bool = True, return_both_visualizations: bool = False):
        """Process single frame through full pipeline - CHÍNH XÁC 100% từ file gốc"""
        import time
        start_time = time.time()
        _log_info("Pipeline", "Starting frame processing...")
        
        # QR Decode
        qr_start = time.time()
        qr_text, qr_pts = self.qr.decode(frame_bgr)
        qr_time = time.time() - qr_start
        _log_info("Pipeline Timing", f"QR decode: {qr_time*1000:.1f}ms")
        meta = {"qr_raw": qr_text, "parsed": None}
        qr_polygon = None
        
        # VALIDATION: Chỉ tiếp tục nếu QR decode thành công
        if not qr_text:
            _log_warning("Pipeline", "No QR code detected - skipping frame")
            return None, None, {"error": "No QR code detected"}, None, None
        
        # Ensure qr_text is a string
        if not isinstance(qr_text, str):
            if isinstance(qr_text, (tuple, list)):
                qr_text = qr_text[0] if len(qr_text) > 0 else ""
            else:
                qr_text = str(qr_text) if qr_text is not None else ""
        
        _log_success("Pipeline", f"QR detected: {qr_text[:50]}...")
        meta["parsed"] = parse_qr_payload(qr_text)
        
        # Load per-id metadata JSON for dataset path
        def _load_qr_meta_by_id(cfg: Config, qr_id: str) -> Optional[dict]:
            if not qr_id:
                return None
            try:
                meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir, f"{qr_id}.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                _log_warning("QR Meta Load", f"Failed to load meta for id {qr_id}: {e}")
            return None

        qr_id = meta["parsed"].get("_qr") if meta.get("parsed") else None
        qr_meta = _load_qr_meta_by_id(self.cfg, qr_id)
        qr_items_dict = {}
        if qr_meta and isinstance(qr_meta.get("fruits"), dict):
            qr_items_dict = qr_meta["fruits"]
        _log_info("Pipeline", f"QR items from JSON: {list(qr_items_dict.keys()) if qr_items_dict else []}")
        
        if qr_pts is not None:
            qr_polygon = np.array(qr_pts, dtype=np.int32).reshape(-1,2)
        
        # GroundingDINO Detection
        gdino_start = time.time()
        qr_items = list(qr_items_dict.keys()) if qr_items_dict else []
        if qr_items:
            _log_info("Pipeline", f"QR items for GroundingDINO: {qr_items}")
        
        # 3-Stage GroundingDINO inference (như bạn yêu cầu)
        # Stage 1: Detect box container
        _log_info("GDINO Stage", "Stage 1: Detecting box container...")
        boxes_stage1, logits_stage1, phrases_stage1, img_resized = self.gd.infer(
            frame_bgr=frame_bgr,
            caption="box .",  # Chỉ detect box
            box_thr=0.35,
            text_thr=0.25
        )
        
        # Stage 2: Detect QR items
        boxes_stage2, logits_stage2, phrases_stage2 = [], [], []
        if qr_items:
            _log_info("GDINO Stage", f"Stage 2: Detecting QR items: {qr_items}")
            qr_prompt = f"{' '.join(qr_items)} ."  # Chỉ detect QR items
            boxes_stage2, logits_stage2, phrases_stage2, _ = self.gd.infer(
                frame_bgr=frame_bgr,
                caption=qr_prompt,
                box_thr=0.35,
                text_thr=0.25
            )
        
        # Combine Stage 1 + Stage 2 results
        if len(boxes_stage1) > 0 and len(boxes_stage2) > 0:
            boxes = torch.cat([boxes_stage1, boxes_stage2])
            logits = torch.cat([logits_stage1, logits_stage2])
            phrases = phrases_stage1 + phrases_stage2
        elif len(boxes_stage1) > 0:
            boxes, logits, phrases = boxes_stage1, logits_stage1, phrases_stage1
        elif len(boxes_stage2) > 0:
            boxes, logits, phrases = boxes_stage2, logits_stage2, phrases_stage2
        else:
            boxes, logits, phrases = torch.tensor([]), torch.tensor([]), []
        
        # FIXED: Check hand detection BEFORE filtering
        phrases_before_filter = phrases.copy() if isinstance(phrases, list) else []
        _log_info("Hand Detection", f"Checking phrases BEFORE filtering: {phrases_before_filter}")
        has_hand, hand_msg = check_hand_detection(phrases_before_filter)
        _log_info("Hand Detection", f"Result: has_hand={has_hand}, msg='{hand_msg}'")
        if has_hand and not preview_only:
            _log_warning("Pipeline", f"Hand detected, discarding: {hand_msg}")
            return None, None, {"error": f"Hand detected: {hand_msg}"}, None, None
        
        # Stage 3: Filter out hand detections for clean visualization
        _log_info("GDINO Stage", "Stage 3: Filtering hand detections for visualization...")
        valid_indices = []
        for i, phrase in enumerate(phrases):
            if not any(hand_word in phrase.lower() for hand_word in ['hand', 'finger', 'thumb', 'palm', 'wrist', 'nail']):
                valid_indices.append(i)
        
        if valid_indices and len(boxes) > 0:
            boxes = boxes[valid_indices]
            logits = logits[valid_indices]
            phrases = [phrases[i] for i in valid_indices]
            _log_info("GDINO Stage", f"After filtering: {len(phrases)} valid detections")
        else:
            boxes, logits, phrases = torch.tensor([]), torch.tensor([]), []
            _log_info("GDINO Stage", "No valid detections after filtering")
        gdino_time = time.time() - gdino_start
        _log_info("Pipeline Timing", f"GroundingDINO inference: {gdino_time*1000:.1f}ms")
        H, W = img_resized.shape[:2]
        num_detections = len(boxes) if boxes is not None else 0
        
        # QR Validation
        if qr_items_dict and not preview_only:
            # FIXED: Use phrases_before_filter for QR validation (before hand filtering)
            is_valid, validation_msg = validate_qr_detection(qr_items_dict, phrases_before_filter)
            if not is_valid:
                _log_warning("Pipeline", f"QR validation failed: {validation_msg}")
                return None, None, {"error": f"QR validation failed: {validation_msg}"}, None, None
        
        # Save original boxes for visualization
        if boxes is not None:
            if isinstance(boxes, torch.Tensor):
                boxes_original = boxes.clone()
            else:
                boxes_original = boxes.copy()
        else:
            boxes_original = None
        
        # Convert boxes to pixel coordinates and filter invalid ones
        if num_detections > 0:
            _log_info("GDINO Debug", f"Converting {num_detections} boxes to pixel coordinates")
            # Convert normalized coordinates to pixel coordinates
            boxes = self._to_pixel_xyxy(boxes, img_resized.shape[1], img_resized.shape[0])
            num_detections = len(boxes)
            _log_info("GDINO Debug", f"After coordinate conversion: {num_detections} valid boxes")
            
            if num_detections == 0:
                _log_warning("Pipeline", "No valid boxes after coordinate conversion")
                return None, None, {"error": "No valid boxes after coordinate conversion"}, None, None
        
        # Box selection
        bbox = self._pick_box_bbox(boxes, phrases, qr_polygon, img_resized.shape)
        if bbox is None:
            return None, None, {"error": "No bbox found"}, None, None
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        _log_info("Pipeline", f"Bbox area: {bbox_area} (min required: {CFG.min_bbox_area})")
        if bbox_area < CFG.min_bbox_area:
            _log_warning("Pipeline", f"Bbox too small: {bbox_area} < {CFG.min_bbox_area}")
            
            # Debug: Save rejected image with bbox visualization
            self._save_rejected_image(frame_bgr, boxes_original, phrases, bbox, f"bbox_too_small_{bbox_area}")
            
            return None, None, {"error": f"Bbox too small: {bbox_area}"}, None, None
        
        # Additional validation: ensure bbox is within image bounds
        H, W = img_resized.shape[:2]
        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
            _log_warning("Pipeline", f"Bbox out of bounds: ({x1}, {y1}, {x2}, {y2}) vs image ({W}, {H})")
            return None, None, {"error": f"Bbox out of bounds"}, None, None
        
        # Background Removal Segmentation
        seg_start = time.time()
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Use Enhanced White-ring segmentation for dataset creation if enabled
        if self.cfg.use_white_ring_seg:
            _log_info("Pipeline", "Using Enhanced White-ring segmentation for dataset creation")
            # Enhanced White-ring: Detect container mask using improved edge detection within ROI
            mask, rect_pts, process_time = process_white_ring_segmentation(img_resized, self.cfg)
            
            # Create overlay visualization with anti-aliasing
            overlay = img_resized.copy()
            if rect_pts is not None:
                # Vẽ polygon đã ép (square/rect) - viền trắng với anti-aliasing
                cv2.polylines(overlay, [rect_pts], True, (255, 255, 255), 6, lineType=cv2.LINE_AA)
                # Vẽ thêm khung xanh để phân biệt
                cv2.polylines(overlay, [rect_pts], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
            else:
                # Fallback: vẽ contour từ mask hoặc contour gốc
                c = largest_contour(mask)
                if c is not None:
                    c = cv2.approxPolyDP(c, 0.007 * cv2.arcLength(c, True), True)
                    cv2.polylines(overlay, [c], True, (255, 255, 255), 6, lineType=cv2.LINE_AA)
            
            # Tô mờ phần trong hộp (sử dụng mask đã được rectified)
            tint = np.zeros_like(overlay)
            tint[:] = (255, 255, 255)
            alpha = 0.25
            overlay = np.where(mask[..., None] > 0, (alpha*tint + (1-alpha)*overlay).astype(overlay.dtype), overlay)
            
            # Generate annotations based on segmentation mode
            annotations = []
            if self.cfg.seg_mode == "single":
                # 1 mask duy nhất = cả vùng trong hộp
                yx = np.column_stack(np.where(mask > 0))
                if yx.size > 0:
                    y_min, x_min = yx.min(axis=0)
                    y_max, x_max = yx.max(axis=0)
                    box = [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]
                    annotations.append({"bbox": box, "category_id": 1, "iscrowd": 0, "segmentation": []})
            else:  # "components"
                vis, n_components = components_inside(mask, img_resized, self.cfg.min_comp_area)
                # Generate annotations for each component
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) >= self.cfg.min_comp_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        annotations.append({"bbox": [int(x), int(y), int(w), int(h)], "category_id": 1, "iscrowd": 0, "segmentation": []})
            
            # VALIDATION: Kiểm tra mask area ratio - FIXED: Use count_nonzero instead of sum()
            mask_area = int(np.count_nonzero(mask))
            total_area = mask.shape[0] * mask.shape[1]
            mask_ratio = mask_area / total_area
            min_mask_ratio = 0.05  # 5% diện tích ảnh
            
            if mask_ratio < min_mask_ratio:
                _log_warning("Pipeline", f"Mask too small: {mask_ratio:.3f} < {min_mask_ratio:.3f} ({mask_area} pixels)")
                return None, None, {"error": f"Mask too small: {mask_ratio:.3f}"}, None, None
            
            _log_info("Pipeline", f"Enhanced White-ring segmentation: {len(annotations)} annotations, {mask_area} pixels, {process_time:.1f}ms, ratio: {mask_ratio:.3f}")
            
        # Use U²-Net for segmentation (fallback)
        else:
            _log_info("Pipeline", "Using U²-Net for segmentation")
            # U²-Net: Box-prompt cho hộp
            mask_box = self.bg_removal.segment_box_by_boxprompt(img_rgb, (x1, y1, x2, y2))
            
            # Point-prompt cho fruits và MERGE
            skip = ("box","container","plastic box","food container","hộp","thùng","tray","bin","crate","qr")
            if num_detections > 0:
                boxes_pix = self._to_pixel_xyxy(boxes, W, H)
                for i, b in enumerate(boxes_pix):
                    ph = (phrases[i] if i < len(phrases) else "").lower()
                    if any(k in ph for k in skip): 
                        continue
                    bx1,by1,bx2,by2 = map(int, b.tolist())
                    if bx1>=x1 and by1>=y1 and bx2<=x2 and by2<=y2:  # nằm trong hộp
                        cx, cy = (bx1+bx2)//2, (by1+by2)//2
                        m_obj = self.bg_removal.segment_object_by_point(img_rgb, (cx,cy), box_hint=(bx1,by1,bx2,by2))
                        mask_box = (mask_box | m_obj).astype(np.uint8)
            
            mask = mask_box
            annotations = []  # U²-Net doesn't generate annotations directly
        seg_time = time.time() - seg_start
        _log_info("Pipeline Timing", f"Segmentation: {seg_time*1000:.1f}ms")
        
        # FIXED: Use count_nonzero instead of sum() for mask area calculation
        if np.count_nonzero(mask) < 200:
            return None, None, {"error": "Mask too small"}, None, None
        
        # Debug: Create visualization of all detections
        if num_detections > 0:
            debug_vis = img_resized.copy()
            # boxes are already in pixel coordinates from _to_pixel_xyxy
            
            # Draw all detections with different colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, (box, phrase) in enumerate(zip(boxes, phrases)):
                x1, y1, x2, y2 = map(int, box.tolist())
                color = colors[i % len(colors)]
                cv2.rectangle(debug_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_vis, f"{i}:{phrase}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save debug visualization to organized directory
            debug_dir = os.path.join(CFG.project_dir, "debug_detections")
            ensure_dir(debug_dir)
            debug_path = os.path.join(debug_dir, f"debug_detections_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, debug_vis)
            _log_info("Debug", f"Saved debug visualization: {debug_path}")
        
        # Create YOLO detections from GroundingDINO - FIXED: Add polygon segmentation
        yolo_detections = []
        if num_detections > 0:
            _log_info("YOLO Detection", f"Processing {num_detections} GroundingDINO detections for YOLO format")
            # boxes are already in pixel coordinates from _to_pixel_xyxy
            for i, b in enumerate(boxes):
                ph = (phrases[i] if i < len(phrases) else "").lower()
                bx1, by1, bx2, by2 = map(int, b.tolist())
                
                # Determine class based on phrase
                if any(k in ph for k in ("box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate")):
                    class_id = 0  # box class
                else:
                    # Map fruit names to class IDs
                    class_id = self._get_fruit_class_id(ph, qr_items_dict)
                
                # Convert to normalized coordinates
                x_center = (bx1 + bx2) / 2.0 / W
                y_center = (by1 + by2) / 2.0 / H
                width = (bx2 - bx1) / W
                height = (by2 - by1) / H
                
                # FIXED: Validate bbox before adding to detections
                if validate_yolo_label(class_id, x_center, y_center, width, height):
                    yolo_detections.append({
                        "class_id": class_id,
                        "class_name": "box" if class_id == 0 else "fruit",  # FIXED: Use actual class names
                        "bbox": [x_center, y_center, width, height],
                        "confidence": 1.0,  # GroundingDINO confidence
                        "phrase": ph
                    })
                    _log_info("YOLO Detection", f"Added detection {i+1}: class={class_id} ({'box' if class_id == 0 else 'fruit'}), bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                else:
                    _log_warning("YOLO Detection", f"Skipped invalid detection {i+1}: class_id={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
        else:
            _log_warning("YOLO Detection", "No GroundingDINO detections found - will use fallback polygon conversion")
        
        # Enforce: exactly 1 plastic box and fruit counts equal to QR items before saving
        try:
            from collections import defaultdict
            box_bboxes = [d for d in yolo_detections if d.get("class_id", 0) == 0]
            if len(box_bboxes) != 1:
                _log_warning("Pipeline", f"Require exactly 1 box, got {len(box_bboxes)}. Skipping frame")
                return None, None, {"error": "require_exactly_one_box"}, None, None

            fruit_bboxes = [d for d in yolo_detections if d.get("class_id", 0) != 0]
            det_counts = defaultdict(int)
            for d in fruit_bboxes:
                ph = (d.get("phrase") or "").lower().strip()
                qr_key = _map_fruit_name_to_qr_item(ph, qr_items_dict)
                det_counts[qr_key] += 1

            mismatches = []
            for item_name, qty_expected in (qr_items_dict or {}).items():
                qty_detected = det_counts.get(item_name, 0)
                if qty_detected != qty_expected:
                    mismatches.append((item_name, qty_expected, qty_detected))

            if mismatches:
                _log_warning("Pipeline", f"Fruit counts do not match QR exactly: {mismatches}. Skipping frame")
                return None, None, {"error": "fruit_count_mismatch", "details": mismatches}, None, None
        except Exception as e:
            _log_error("Pipeline", e, "While enforcing exact-count rule")
            return None, None, {"error": "rule_enforce_failed"}, None, None
        
        # Save to dataset
        img_path, lab_path = None, None
        if save_dataset:
            dataset_start = time.time()
            box_id = None
            if meta.get("parsed") and meta["parsed"].get("Box"):
                box_id = meta["parsed"]["Box"].replace(" ", "")
            
            # FIXED: Ensure metadata consistency between preview and export
            # Include all information that appears in preview
            meta_out = {
                "qr": meta, 
                "gdino_phrases": phrases, 
                "bbox": [x1,y1,x2,y2], 
                "yolo_detections": yolo_detections,
                "white_ring_annotations": annotations if self.cfg.use_white_ring_seg else [],
                # FIXED: Add additional metadata for consistency - Use count_nonzero instead of sum()
                "box_id": box_id,
                "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
                "mask_ratio": float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])) if mask is not None else 0.0,
                "num_detections": num_detections,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Save white-ring outputs if enabled
            if self.cfg.use_white_ring_seg:
                # Save overlay and mask to organized directories
                seg_dir = os.path.join(CFG.project_dir, "seg_overlays")
                ensure_dir(seg_dir)
                overlay_path = os.path.join(seg_dir, f"seg_overlay_{int(time.time())}.png")
                mask_path = os.path.join(seg_dir, f"mask_container_{int(time.time())}.png")
                cv2.imwrite(overlay_path, overlay)
                cv2.imwrite(mask_path, mask)
                _log_info("White Ring", f"Saved overlay: {overlay_path}")
                _log_info("White Ring", f"Saved mask: {mask_path}")
                
                # FIXED: Use the same mask for both preview and dataset to ensure consistency
                # Don't read back from file as it may cause inconsistencies
                dataset_mask = mask.copy()
                _log_info("Pipeline Debug", f"Using consistent mask for dataset: {np.count_nonzero(dataset_mask)} pixels")
            else:
                dataset_mask = mask
            
            img_path, lab_path = self.ds.add_sample(img_resized, dataset_mask, meta_out, box_id, yolo_detections)
            dataset_time = time.time() - dataset_start
            _log_info("Pipeline Timing", f"Dataset creation: {dataset_time*1000:.1f}ms")
            
            # FIXED: Update class names and regenerate YAML after adding sample
            _log_info("Debug Classes", f"Before update_class_names - Dataset classes: {self.ds.class_names}")
            self.update_class_names()
            _log_info("Debug Classes", f"After update_class_names - Dataset classes: {self.ds.class_names}")
        
        # Visualization - FIXED: Handle both [0,1] and [0,255] mask formats
        vis = img_rgb.copy()
        if mask is not None and mask.any():
            mask_colored = np.zeros_like(img_rgb)
            # FIXED: Use mask > 0 instead of mask == 1 to handle both formats
            mask_colored[mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
            
            # Ensure mask is in correct format for contour detection
            if mask.max() <= 1:
                mask_contour = (mask * 255).astype(np.uint8)
            else:
                mask_contour = mask.astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Total processing time
        total_time = time.time() - start_time
        _log_success("Pipeline Timing", f"Total processing time: {total_time*1000:.1f}ms")
        
        # FIXED: Ensure preview metadata matches export metadata - Use count_nonzero instead of sum()
        preview_meta = {
            "qr": meta, 
            "bbox": [x1,y1,x2,y2],
            "box_id": box_id,
            "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
            "mask_ratio": float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])) if mask is not None else 0.0,
            "num_detections": num_detections,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # Update class names after processing
        self.update_class_names()
        
        # FIXED: Đảm bảo YAML được tạo ngay khi có đủ classes
        if len(self.ds.class_names) >= 2:
            _log_success("Pipeline", f"Dataset ready with {len(self.ds.class_names)} classes: {self.ds.class_names}")
        else:
            _log_warning("Pipeline", f"Dataset not ready yet. Need more detections to create multi-class YAML. Current: {self.ds.class_names}")
        
        if return_both_visualizations:
            # Create bbox visualization using original GroundingDINO detections
            vis_bbox = self._create_gdino_visualization(img_resized, boxes_original, logits, phrases)
            return vis_bbox, vis, preview_meta, img_path, lab_path
        
        return vis, vis, preview_meta, img_path, lab_path
    
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
        """Convert pixel coordinates back to normalized coordinates for _annotate"""
        if boxes is None or len(boxes) == 0:
            return boxes
        
        # Ensure boxes is numpy array
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        elif not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        
        # Convert from pixel to normalized
        normalized_boxes = boxes.copy().astype(np.float32)
        normalized_boxes[:, [0, 2]] = normalized_boxes[:, [0, 2]] / img_w  # x coordinates
        normalized_boxes[:, [1, 3]] = normalized_boxes[:, [1, 3]] / img_h  # y coordinates
        
        _log_info("Coordinate Conversion", f"Converted {len(normalized_boxes)} boxes from pixel to normalized")
        _log_info("Coordinate Conversion", f"Normalized boxes range: [{normalized_boxes.min():.3f}, {normalized_boxes.max():.3f}]")
        
        return normalized_boxes
    
    def validate_dataset_before_training(self, data_yaml: str) -> bool:
        """Validate dataset before training to catch common issues"""
        _log_info("Dataset Validation", "Starting dataset validation...")
        
        try:
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check YAML structure
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data:
                    _log_error("Dataset Validation", f"Missing required key in YAML: {key}")
                    return False
            
            # Check paths exist
            base_path = data['path']
            train_path = os.path.join(base_path, data['train'])
            val_path = os.path.join(base_path, data['val'])
            
            if not os.path.exists(train_path):
                _log_error("Dataset Validation", f"Train path does not exist: {train_path}")
                return False
                
            if not os.path.exists(val_path):
                _log_error("Dataset Validation", f"Val path does not exist: {val_path}")
                return False
            
            # Count images and labels
            train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            _log_info("Dataset Validation", f"Found {len(train_images)} train images, {len(val_images)} val images")
            
            if len(train_images) == 0:
                _log_error("Dataset Validation", "No training images found!")
                return False
                
            if len(val_images) == 0:
                _log_error("Dataset Validation", "No validation images found!")
                return False
            
            # Check labels directory
            train_labels_path = train_path.replace('images', 'labels')
            val_labels_path = val_path.replace('images', 'labels')
            
            if not os.path.exists(train_labels_path):
                _log_error("Dataset Validation", f"Train labels path does not exist: {train_labels_path}")
                return False
                
            if not os.path.exists(val_labels_path):
                _log_error("Dataset Validation", f"Val labels path does not exist: {val_labels_path}")
                return False
            
            # Validate label files
            valid_train_labels = 0
            valid_val_labels = 0
            
            for img_file in train_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(train_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:  # Non-empty label file
                            valid_train_labels += 1
                            # Validate each line
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    _log_warning("Dataset Validation", f"Invalid label format in {label_file}: {line.strip()}")
                                    continue
                                try:
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:5]]
                                    if not validate_yolo_label(class_id, coords[0], coords[1], coords[2], coords[3]):
                                        _log_warning("Dataset Validation", f"Invalid label values in {label_file}: {line.strip()}")
                                except ValueError:
                                    _log_warning("Dataset Validation", f"Invalid label values in {label_file}: {line.strip()}")
                else:
                    _log_warning("Dataset Validation", f"Missing label file: {label_file}")
            
            for img_file in val_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(val_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:  # Non-empty label file
                            valid_val_labels += 1
            
            _log_info("Dataset Validation", f"Valid labels: {valid_train_labels} train, {valid_val_labels} val")
            
            if valid_train_labels == 0:
                _log_error("Dataset Validation", "No valid training labels found!")
                return False
                
            if valid_val_labels == 0:
                _log_error("Dataset Validation", "No valid validation labels found!")
                return False
            
            # Check class distribution
            class_counts = {0: 0, 1: 0}
            for img_file in train_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(train_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    class_id = int(parts[0])
                                    if class_id in class_counts:
                                        class_counts[class_id] += 1
                                except ValueError:
                                    pass
            
            _log_info("Dataset Validation", f"Class distribution: {class_counts}")
            
            if class_counts[0] == 0:
                _log_warning("Dataset Validation", "No 'box' class (0) found in training data!")
                
            if class_counts[1] == 0:
                _log_warning("Dataset Validation", "No 'fruit' class (1) found in training data!")
            
            _log_success("Dataset Validation", "Dataset validation completed successfully!")
            return True
            
        except Exception as e:
            _log_error("Dataset Validation", f"Validation failed: {str(e)}")
            return False

    def _check_training_environment(self):
        """Check for common training environment issues that cause loss calculation problems"""
        _log_info("Training Environment", "Checking training environment...")
        
        try:
            # Check CUDA availability
            import torch
            if torch.cuda.is_available():
                if not _suppress_all_cuda_logs:
                    _log_info("Training Environment", f"CUDA available: {torch.cuda.get_device_name(0)}")
                    _log_info("Training Environment", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                _log_warning("Training Environment", "CUDA not available - using CPU")
            
            # Check ultralytics installation
            try:
                from ultralytics import YOLO
                _log_success("Training Environment", "Ultralytics YOLO imported successfully")
            except ImportError as e:
                _log_error("Training Environment", f"Failed to import ultralytics: {e}")
                return False
            
            # Check for common path issues
            import os
            if not os.path.exists(self.cfg.project_dir):
                _log_warning("Training Environment", f"Project directory does not exist: {self.cfg.project_dir}")
                os.makedirs(self.cfg.project_dir, exist_ok=True)
                _log_info("Training Environment", f"Created project directory: {self.cfg.project_dir}")
            
            # Check for sufficient disk space
            import shutil
            free_space = shutil.disk_usage(self.cfg.project_dir).free
            if free_space < 5 * 1024**3:  # 5 GB
                _log_warning("Training Environment", f"Low disk space: {free_space / 1024**3:.1f} GB available")
            
            _log_success("Training Environment", "Environment check completed successfully")
            return True
            
        except Exception as e:
            _log_error("Training Environment", f"Environment check failed: {str(e)}")
            return False

    def train_sdy(self, data_yaml: str, continue_if_exists: bool = True, resume_from: str = None):
        """Train YOLOv8 model with enhanced hyperparameters and fine-tuning support"""
        import time
        start_time = time.time()
        from ultralytics import YOLO
        
        # FIXED: Validate dataset before training
        _log_info("YOLO Training", "Starting YOLO training with dataset validation...")
        if not self.validate_dataset_before_training(data_yaml):
            _log_error("YOLO Training", "Dataset validation failed - aborting training")
            return None, None
        
        # FIXED: Additional training configuration to prevent loss issues
        _log_info("YOLO Training", "Configuring training parameters to prevent loss calculation issues...")
        
        # FIXED: Check for common training issues
        if not self._check_training_environment():
            _log_error("YOLO Training", "Training environment check failed - aborting training")
            return None, None
        
        _log_info("YOLO Training", "Starting YOLOv8 training...")
        # Suppress all CUDA logging during training
        global _suppress_all_cuda_logs
        _suppress_all_cuda_logs = True
        save_dir = os.path.join(CFG.project_dir, "runs_sdy")
        ensure_dir(save_dir)
        
        # FIXED: Support fine-tuning from existing weights
        weights_dir = os.path.join(save_dir, "sdy_train", "weights")
        best_path_default = os.path.join(weights_dir, "best.pt")
        start_weights = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)
        
        # FIXED: Additional debugging for training issues
        _log_info("YOLO Training", f"Using weights: {start_weights if start_weights else 'yolov8n.pt (default)'}")
        _log_info("YOLO Training", f"Data YAML: {data_yaml}")
        _log_info("YOLO Training", f"Save directory: {save_dir}")
        
        # FIXED: Additional validation before training
        if not os.path.exists(data_yaml):
            _log_error("YOLO Training", f"Data YAML file not found: {data_yaml}")
            return None, None
        
        try:
            if start_weights and os.path.isfile(start_weights):
                _log_info("YOLO Training", f"Fine-tuning from: {start_weights}")
                model = YOLO(start_weights)  # Load existing weights
            else:
                _log_info("YOLO Training", f"Training from scratch using: {CFG.yolo_base}")
                model = YOLO(CFG.yolo_base)  # FIXED: Only load base model if no existing weights
            
            device = 0 if CFG.device.startswith("cuda") else "cpu"
            
            # Enhanced training parameters with GPU optimization
            train_args = {
                "data": data_yaml,
                "epochs": CFG.yolo_epochs,
                "imgsz": CFG.yolo_imgsz,
                "batch": CFG.yolo_batch,
                "device": device,
                "project": save_dir,
                "name": "sdy_train",
                "exist_ok": True,
                "amp": True if CFG.device.startswith("cuda") else False,
                "lr0": CFG.yolo_lr0,
                "lrf": CFG.yolo_lrf,
                "weight_decay": CFG.yolo_weight_decay,
                "workers": CFG.yolo_workers,
                "mosaic": CFG.yolo_mosaic,
                "fliplr": 0.5 if CFG.yolo_flip else 0.0,  # Horizontal flip probability
                "flipud": 0.0,  # Vertical flip probability (usually 0 for object detection)
                "hsv_h": 0.015 if CFG.yolo_hsv else 0.0,
                "hsv_s": 0.7 if CFG.yolo_hsv else 0.0,
                "hsv_v": 0.4 if CFG.yolo_hsv else 0.0,
                "save_period": 10,  # Save checkpoint every 10 epochs
                "plots": True,  # Generate training plots
                "val": True,  # Validate during training
                # GPU optimization parameters
                "cache": False,  # Disable caching to prevent memory issues
                "single_cls": False,  # Multi-class detection
                "rect": False,  # Disable rectangular training for stability
                "cos_lr": True,  # Cosine learning rate scheduler
                "close_mosaic": 10,  # Close mosaic augmentation in last 10 epochs
            }
        
            _log_info("YOLO Train", f"Starting training with args: {train_args}")
            
            # Pre-training GPU memory optimization
            if CFG.device.startswith("cuda"):
                smart_gpu_memory_management()
                _log_info("YOLO Train", "GPU memory optimized before training")
            
            results = model.train(**train_args)
        
            # Generate training curves and metrics
            self._generate_yolo_metrics(save_dir, results)
            
            total_time = time.time() - start_time
            _log_success("YOLO Training", f"Training completed in {total_time/60:.1f} minutes")
            
            weights_dir = os.path.join(save_dir, "sdy_train", "weights")
            best = os.path.join(weights_dir, "best.pt")
            return best if os.path.isfile(best) else "", weights_dir
            
        except Exception as e:
            _log_error("YOLO Training", f"Training failed: {str(e)}")
            return None, None
    
    def _generate_yolo_metrics(self, save_dir: str, results):
        """Generate training metrics and curves for YOLO"""
        try:
            import matplotlib.pyplot as plt
            
            # Create metrics directory
            metrics_dir = os.path.join(save_dir, "sdy_train", "metrics")
            ensure_dir(metrics_dir)
            
            # Save training results as JSON
            metrics_file = os.path.join(metrics_dir, "training_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "results": results.results_dict if hasattr(results, 'results_dict') else {},
                    "best_fitness": results.fitness if hasattr(results, 'fitness') else 0,
                    "epochs": CFG.yolo_epochs,
                    "batch_size": CFG.yolo_batch,
                    "image_size": CFG.yolo_imgsz,
                    "learning_rate": CFG.yolo_lr0,
                }, f, ensure_ascii=False, indent=2)
            
            _log_success("YOLO Metrics", f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            _log_warning("YOLO Metrics", f"Could not generate metrics: {e}")
    
    def train_u2net(self, continue_if_exists: bool = True, resume_from: str = None):
        """Train U²-Net model with comprehensive metrics tracking and fine-tuning support"""
        import time
        start_time = time.time()
        ds_root = self.ds.root
        
        _log_info("U²-Net Training", "Starting U²-Net training...")
        # Suppress all CUDA logging during training
        global _suppress_all_cuda_logs
        _suppress_all_cuda_logs = True
        setup_gpu_memory(CFG)
        
        device = torch.device(CFG.device)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        _log_info("U2Net Train", f"🚀 Training on device: {device} ({gpu_name})")
        imgsz = CFG.u2_imgsz
        
        # Datasets
        train_set = U2PairDataset(ds_root, "train", imgsz)
        val_set = U2PairDataset(ds_root, "val", imgsz)
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=CFG.u2_batch, shuffle=True,
            num_workers=CFG.u2_workers, pin_memory=True, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=CFG.u2_batch, shuffle=False,
            num_workers=CFG.u2_workers, pin_memory=True, drop_last=False
        )
        
        # Model - support all 3 variants
        variant = CFG.u2_variant.lower()
        if variant == "u2netp":
            net = U2NETP(3, 1)
        elif variant == "u2net":
            net = U2NET(3, 1)
        elif variant == "u2net_lite":
            net = U2NET_LITE(3, 1)
        else:
            raise ValueError(f"Unknown U2Net variant: {variant}")
        
        net = net.to(device)
        
        # FIXED: Support fine-tuning from existing weights
        run_dir = os.path.join(CFG.project_dir, CFG.u2_runs_dir)
        best_path_default = os.path.join(run_dir, CFG.u2_best_name)
        start_path = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)
        
        if start_path and os.path.isfile(start_path):
            _log_info("U2Net Train", f"Fine-tuning from: {start_path}")
            try:
                checkpoint = torch.load(start_path, map_location=device)
                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    net.load_state_dict(checkpoint["state_dict"], strict=True)
                else:
                    net.load_state_dict(checkpoint, strict=True)
                _log_success("U2Net Train", f"Loaded weights from: {start_path}")
            except Exception as e:
                _log_warning("U2Net Train", f"Failed to load weights from {start_path}: {e}")
                _log_info("U2Net Train", "Continuing with training from scratch")
        else:
            _log_info("U2Net Train", "Training from scratch - no existing weights found")
        
        _log_success("U2Net Train", f"Model {variant} created and moved to {device}")
        
        # Optimizer & Loss with enhanced options
        if CFG.u2_optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay)
        elif CFG.u2_optimizer.lower() == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay, momentum=0.9)
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay)
        
        # Loss function selection
        if CFG.u2_loss.lower() == "bce":
            loss_fn = nn.BCEWithLogitsLoss()
        elif CFG.u2_loss.lower() == "dice":
            loss_fn = BCEDiceLoss()  # Use BCEDiceLoss as it includes Dice
        else:  # BCEDice
            loss_fn = BCEDiceLoss()
        
        # Edge loss for better boundary quality
        edge_loss_fn = EdgeLoss() if CFG.u2_use_edge_loss else None
        
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CFG.u2_amp))
        
        # FIXED: Setup AMP context for consistent usage
        from contextlib import nullcontext
        amp_enabled = (device.type == "cuda" and CFG.u2_amp)
        amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        
        # Training loop with metrics tracking
        run_dir = os.path.join(CFG.project_dir, CFG.u2_runs_dir)
        ensure_dir(run_dir)
        best_path = os.path.join(run_dir, CFG.u2_best_name)
        last_path = os.path.join(run_dir, CFG.u2_last_name)
        
        # Metrics tracking
        train_losses = []
        val_losses = []
        train_ious = []
        val_ious = []
        train_dices = []
        val_dices = []
        epochs = []
        
        best_val = 1e9
        for ep in range(1, CFG.u2_epochs + 1):
            # Train
            net.train()
            ep_loss = 0.0
            ep_iou = 0.0
            ep_dice = 0.0
            train_samples = 0
            
            for img_t, mask_t, _ in train_loader:
                img_t, mask_t = img_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                
                # FIXED: Use consistent AMP context with proper fallback
                with amp_ctx(enabled=amp_enabled):
                    logits = net(img_t)
                    main_loss = loss_fn(logits, mask_t)
                    
                    # Add edge loss if enabled
                    if edge_loss_fn is not None:
                        edge_loss = edge_loss_fn(logits, mask_t)
                        loss = main_loss + CFG.u2_edge_loss_weight * edge_loss
                    else:
                        loss = main_loss
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                # Calculate metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                    
                    # IoU calculation
                    intersection = (pred_mask * mask_t).sum(dim=(2, 3))
                    union = pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) - intersection
                    iou = (intersection / (union + 1e-8)).mean()
                    
                    # Dice calculation
                    dice = (2 * intersection / (pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) + 1e-8)).mean()
                    
                    ep_loss += loss.item() * img_t.size(0)
                    ep_iou += iou.item() * img_t.size(0)
                    ep_dice += dice.item() * img_t.size(0)
                    train_samples += img_t.size(0)
            
            # Validation
            net.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_dice = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for img_t, mask_t, _ in val_loader:
                    img_t, mask_t = img_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
                    # FIXED: Use consistent AMP context with proper fallback
                    with amp_ctx(enabled=amp_enabled):
                        logits = net(img_t)
                        loss = loss_fn(logits, mask_t)
                    
                    # Calculate metrics
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                    
                    # IoU calculation
                    intersection = (pred_mask * mask_t).sum(dim=(2, 3))
                    union = pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) - intersection
                    iou = (intersection / (union + 1e-8)).mean()
                    
                    # Dice calculation
                    dice = (2 * intersection / (pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) + 1e-8)).mean()
                    
                    val_loss += loss.item() * img_t.size(0)
                    val_iou += iou.item() * img_t.size(0)
                    val_dice += dice.item() * img_t.size(0)
                    val_samples += img_t.size(0)
            
            # Average metrics
            ep_loss /= max(1, train_samples)
            ep_iou /= max(1, train_samples)
            ep_dice /= max(1, train_samples)
            val_loss /= max(1, val_samples)
            val_iou /= max(1, val_samples)
            val_dice /= max(1, val_samples)
            
            # Store metrics
            epochs.append(ep)
            train_losses.append(ep_loss)
            val_losses.append(val_loss)
            train_ious.append(ep_iou)
            val_ious.append(val_iou)
            train_dices.append(ep_dice)
            val_dices.append(val_dice)
            
            _log_info("U2Net Train", f"Epoch {ep}/{CFG.u2_epochs} | train_loss={ep_loss:.4f} | val_loss={val_loss:.4f} | train_iou={ep_iou:.4f} | val_iou={val_iou:.4f} | train_dice={ep_dice:.4f} | val_dice={val_dice:.4f}")
            
            # Save
            torch.save({"epoch": ep, "state_dict": net.state_dict()}, last_path)
            if val_loss < best_val:
                best_val = val_loss
                shutil.copyfile(last_path, best_path)
                _log_success("U2Net Train", f"New best @ epoch {ep}: val_loss={val_loss:.4f}")
            
            # Smart memory cleanup
            if CFG.enable_memory_optimization:
                smart_gpu_memory_management()
        
        _log_success("U2Net Train", f"Training completed! Best model: {best_path}")
        
        # Export to ONNX
        onnx_path = self._export_u2net_onnx(net, best_path, run_dir)
        
        # Generate comprehensive training metrics and plots
        training_metrics = {
            "epochs": epochs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_ious": train_ious,
            "val_ious": val_ious,
            "train_dices": train_dices,
            "val_dices": val_dices,
            "best_val_loss": best_val
        }
        self._generate_u2net_metrics(run_dir, training_metrics, net, val_loader, device)
        
        total_time = time.time() - start_time
        _log_success("U²-Net Training", f"Training completed in {total_time/60:.1f} minutes")
        
        return best_path, run_dir, onnx_path
    
    def _export_u2net_onnx(self, net, best_path: str, run_dir: str) -> str:
        """Export U²-Net model to ONNX format"""
        try:
            import torch.onnx
            
            # Load best model
            checkpoint = torch.load(best_path, map_location=CFG.device)
            net.load_state_dict(checkpoint["state_dict"])
            net.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, CFG.u2_imgsz, CFG.u2_imgsz).to(CFG.device)
            
            # Export to ONNX
            onnx_path = os.path.join(run_dir, "u2net_best.onnx")
            torch.onnx.export(
                net,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            
            _log_success("U2Net ONNX", f"Exported to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            _log_warning("U2Net ONNX", f"Could not export ONNX: {e}")
            return ""
    
    def _generate_u2net_metrics(self, run_dir: str, training_metrics: dict, model, val_loader, device):
        """Generate comprehensive training metrics and plots for U²-Net"""
        try:
            import matplotlib.pyplot as plt
            
            # Set style
            plt.style.use('default')
            if HAVE_SEABORN:
                sns.set_palette("husl")
            
            # Create plots directory
            plots_dir = os.path.join(run_dir, "plots")
            ensure_dir(plots_dir)
            
            # 1. Training Curves
            self._plot_training_curves(plots_dir, training_metrics)
            
            # 2. Confusion Matrix (only if sklearn available)
            if HAVE_SKLEARN:
                self._plot_confusion_matrix(plots_dir, model, val_loader, device)
            else:
                _log_warning("U2Net Metrics", "sklearn not available, skipping confusion matrix")
            
            # 3. Batch Visualizations
            self._plot_batch_samples(plots_dir, model, val_loader, device)
            
            # 4. Metrics Summary
            self._save_metrics_summary(run_dir, training_metrics)
            
            _log_success("U2Net Metrics", f"Generated comprehensive metrics and plots in {run_dir}")
            
        except Exception as e:
            _log_warning("U2Net Metrics", f"Could not generate metrics: {e}")
    
    def _plot_training_curves(self, plots_dir: str, training_metrics: dict):
        """Plot training curves for loss, IoU, and Dice"""
        try:
            epochs = training_metrics["epochs"]
            train_losses = training_metrics["train_losses"]
            val_losses = training_metrics["val_losses"]
            train_ious = training_metrics["train_ious"]
            val_ious = training_metrics["val_ious"]
            train_dices = training_metrics["train_dices"]
            val_dices = training_metrics["val_dices"]
            
            # Loss curves
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # IoU curves
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_ious, 'b-', label='Train IoU', linewidth=2)
            plt.plot(epochs, val_ious, 'r-', label='Val IoU', linewidth=2)
            plt.title('Training & Validation IoU', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Dice curves
            plt.subplot(1, 3, 3)
            plt.plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
            plt.plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
            plt.title('Training & Validation Dice', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual curves
            # Loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            plt.title('U²-Net Training Loss', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # IoU curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_ious, 'b-', label='Train IoU', linewidth=2)
            plt.plot(epochs, val_ious, 'r-', label='Val IoU', linewidth=2)
            plt.title('U²-Net IoU Score', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('IoU', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'iou_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dice curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
            plt.plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
            plt.title('U²-Net Dice Score', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Dice Score', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'dice_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            _log_success("U2Net Plots", "Training curves generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate training curves: {e}")
    
    def _plot_confusion_matrix(self, plots_dir: str, model, val_loader, device):
        """Generate confusion matrix for segmentation results"""
        try:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for img_t, mask_t, _ in val_loader:
                    img_t, mask_t = img_t.to(device), mask_t.to(device)
                    
                    logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                    
                    # Flatten for confusion matrix
                    pred_flat = pred_mask.view(-1).cpu().numpy()
                    target_flat = mask_t.view(-1).cpu().numpy()
                    
                    all_preds.extend(pred_flat)
                    all_targets.extend(target_flat)
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Generate confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            if HAVE_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Background', 'Foreground'],
                           yticklabels=['Background', 'Foreground'])
            else:
                # Fallback to matplotlib
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Background', 'Foreground'])
                plt.yticks(tick_marks, ['Background', 'Foreground'])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.title('U²-Net Segmentation Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(8, 6))
            if HAVE_SEABORN:
                sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                           xticklabels=['Background', 'Foreground'],
                           yticklabels=['Background', 'Foreground'])
            else:
                # Fallback to matplotlib
                plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Background', 'Foreground'])
                plt.yticks(tick_marks, ['Background', 'Foreground'])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, f'{cm_norm[i, j]:.3f}', ha="center", va="center")
            plt.title('U²-Net Normalized Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            _log_success("U2Net Plots", "Confusion matrices generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate confusion matrix: {e}")
    
    def _plot_batch_samples(self, plots_dir: str, model, val_loader, device):
        """Generate batch visualization samples"""
        try:
            model.eval()
            
            # Get a batch of validation data
            for img_t, mask_t, names in val_loader:
                img_t, mask_t = img_t.to(device), mask_t.to(device)
                
                with torch.no_grad():
                    logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                
                # Convert to numpy
                images = img_t.cpu().numpy()
                gt_masks = mask_t.cpu().numpy()
                pred_masks = pred_mask.cpu().numpy()
                
                # Create visualization
                batch_size = min(4, len(images))  # Show max 4 samples
                fig, axes = plt.subplots(3, batch_size, figsize=(4*batch_size, 12))
                if batch_size == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(batch_size):
                    # Original image
                    img = np.transpose(images[i], (1, 2, 0))
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    # Ground truth mask
                    gt_mask = gt_masks[i, 0]
                    axes[1, i].imshow(gt_mask, cmap='gray')
                    axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
                    axes[1, i].axis('off')
                    
                    # Predicted mask
                    pred_mask_vis = pred_masks[i, 0]
                    axes[2, i].imshow(pred_mask_vis, cmap='gray')
                    axes[2, i].set_title(f'Predicted {i+1}', fontweight='bold')
                    axes[2, i].axis('off')
                
                plt.suptitle('U²-Net Validation Batch Samples', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'val_batch_samples.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create overlay visualization
                fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
                if batch_size == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(batch_size):
                    # Ground truth overlay
                    img = np.transpose(images[i], (1, 2, 0))
                    gt_mask = gt_masks[i, 0]
                    overlay_gt = img.copy()
                    overlay_gt[gt_mask > 0.5] = [0, 1, 0]  # Green for ground truth
                    axes[0, i].imshow(overlay_gt)
                    axes[0, i].set_title(f'GT Overlay {i+1}', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    # Prediction overlay
                    pred_mask_vis = pred_masks[i, 0]
                    overlay_pred = img.copy()
                    overlay_pred[pred_mask_vis > 0.5] = [1, 0, 0]  # Red for prediction
                    axes[1, i].imshow(overlay_pred)
                    axes[1, i].set_title(f'Pred Overlay {i+1}', fontweight='bold')
                    axes[1, i].axis('off')
                
                plt.suptitle('U²-Net Overlay Visualizations', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'val_batch_overlay.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                break  # Only process first batch
            
            _log_success("U2Net Plots", "Batch samples generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate batch samples: {e}")
    
    def _save_metrics_summary(self, run_dir: str, training_metrics: dict):
        """Save comprehensive metrics summary"""
        try:
            # Save detailed metrics as JSON
            metrics_file = os.path.join(run_dir, "training_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "training_metrics": training_metrics,
                    "config": {
                        "epochs": CFG.u2_epochs,
                        "batch_size": CFG.u2_batch,
                        "image_size": CFG.u2_imgsz,
                        "learning_rate": CFG.u2_lr,
                        "optimizer": CFG.u2_optimizer,
                        "loss_function": CFG.u2_loss,
                        "mixed_precision": CFG.u2_amp,
                        "workers": CFG.u2_workers,
                        "variant": CFG.u2_variant
                    },
                    "best_metrics": {
                        "best_val_loss": min(training_metrics["val_losses"]),
                        "best_val_iou": max(training_metrics["val_ious"]),
                        "best_val_dice": max(training_metrics["val_dices"]),
                        "final_train_loss": training_metrics["train_losses"][-1],
                        "final_val_loss": training_metrics["val_losses"][-1],
                        "final_train_iou": training_metrics["train_ious"][-1],
                        "final_val_iou": training_metrics["val_ious"][-1],
                        "final_train_dice": training_metrics["train_dices"][-1],
                        "final_val_dice": training_metrics["val_dices"][-1]
                    }
                }, f, ensure_ascii=False, indent=2)
            
            # Save CSV for easy analysis
            csv_file = os.path.join(run_dir, "training_results.csv")
            df = pd.DataFrame({
                'epoch': training_metrics["epochs"],
                'train_loss': training_metrics["train_losses"],
                'val_loss': training_metrics["val_losses"],
                'train_iou': training_metrics["train_ious"],
                'val_iou': training_metrics["val_ious"],
                'train_dice': training_metrics["train_dices"],
                'val_dice': training_metrics["val_dices"]
            })
            df.to_csv(csv_file, index=False)
            
            _log_success("U2Net Metrics", f"Metrics summary saved to {metrics_file} and {csv_file}")
            
        except Exception as e:
            _log_warning("U2Net Metrics", f"Could not save metrics summary: {e}")
    
    def write_yaml(self) -> str:
        """Write dataset YAML for YOLO"""
        return self.ds.write_yaml()

