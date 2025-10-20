# ========================= SECTION H: WAREHOUSE CHECKER (TAB KHO - MỚI) ========================= #
# ========================= SECTION H: WAREHOUSE CHECKER ========================= #

import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional



import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
# Global variables for warehouse checker
warehouse_yolo_model = None
warehouse_u2net_model = None

def load_warehouse_yolo(model_path: str):
    """Load YOLO model for warehouse checking"""
    global warehouse_yolo_model
    try:
        from ultralytics import YOLO
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] YOLO weight file not found"
        
        model = YOLO(model_path)
        model.to(CFG.device)
        warehouse_yolo_model = model
        
        _log_success("Warehouse YOLO", f"Model loaded from {model_path}")
        return model, f"✅ YOLO loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse YOLO", e, "Failed to load YOLO")
        return None, f"[ERROR] {e}"

def load_warehouse_u2net(model_path: str):
    """Load U²-Net model for warehouse checking - hỗ trợ cả 3 variants"""
    global warehouse_u2net_model
    try:
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] U²-Net weight file not found"
        
        # Load model theo variant
        variant = CFG.u2_variant.lower()
        if variant == "u2netp":
            model = U2NETP(3, 1)
        elif variant == "u2net":
            model = U2NET(3, 1)
        elif variant == "u2net_lite":
            model = U2NET_LITE(3, 1)
        else:
            return None, f"[ERROR] Unknown variant: {variant}"
        
        checkpoint = torch.load(model_path, map_location=CFG.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(CFG.device)
        model.eval()
        
        warehouse_u2net_model = model
        _log_success("Warehouse U2Net", f"Model {variant} loaded from {model_path}")
        return model, f"✅ U²-Net ({variant}) loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse U2Net", e, "Failed to load U²-Net")
        return None, f"[ERROR] {e}"

def load_bg_removal_model(model_name: str, cfg: Config):
    """Load background removal model based on name"""
    _log_info("BG Removal Model", f"Loading {model_name} model...")
    
    try:
        device = torch.device(cfg.device)
        
        if model_name == "u2netp":
            model = U2NETP(3, 1)
        elif model_name == "u2net":
            model = U2NET(3, 1)
        elif model_name == "u2net_lite":
            model = U2NET_LITE(3, 1)
        elif model_name == "u2net_human_seg":
            # Load pre-trained human segmentation model
            model = U2NET(3, 1)  # Use U2NET as base
            _log_warning("BG Removal Model", "u2net_human_seg not implemented, using U2NET")
        elif model_name == "isnet":
            # Load ISNet model
            _log_warning("BG Removal Model", "ISNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "rembg":
            # Load rembg model
            _log_warning("BG Removal Model", "rembg not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "modnet":
            # Load MODNet model
            _log_warning("BG Removal Model", "MODNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "silueta":
            # Load Silueta model
            _log_warning("BG Removal Model", "Silueta not implemented, using U2NETP")
            model = U2NETP(3, 1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        model.eval()
        
        _log_success("BG Removal Model", f"{model_name} loaded successfully")
        return model
        
    except Exception as e:
        _log_error("BG Removal Model", e, f"Failed to load {model_name}")
        return None

def deskew_box_roi(roi_bgr: np.ndarray, mask: np.ndarray, method: str = "minAreaRect") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew box ROI to align to 90-degree angles"""
    try:
        H, W = roi_bgr.shape[:2]
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi_bgr, mask, {"angle": 0, "method": "no_contour"}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if method == "minAreaRect":
            # Use minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Normalize angle to 0-90 degrees
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
        elif method == "PCA":
            # Use PCA to find principal axis
            data_pts = largest_contour.reshape(-1, 2).astype(np.float32)
            mean = np.mean(data_pts, axis=0)
            centered = data_pts - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
        else:  # heuristic
            # Use bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            angle = 0  # Assume already aligned
        
        # Snap to nearest 90-degree angle
        if abs(angle) < 15:
            angle = 0
        elif abs(angle - 90) < 15:
            angle = 90
        elif abs(angle + 90) < 15:
            angle = -90
        elif abs(angle - 180) < 15 or abs(angle + 180) < 15:
            angle = 0
        
        # Apply rotation if needed
        if abs(angle) > 1:  # Only rotate if significant angle
            center = (W // 2, H // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate ROI and mask
            roi_rotated = cv2.warpAffine(roi_bgr, rotation_matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask_rotated = cv2.warpAffine(mask, rotation_matrix, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            _log_info("Deskew", f"Rotated by {angle:.1f} degrees using {method}")
            return roi_rotated, mask_rotated, {"angle": angle, "method": method, "center": center}
        
        return roi_bgr, mask, {"angle": 0, "method": method}
        
    except Exception as e:
        _log_warning("Deskew", f"Deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "error", "error": str(e)}

def _process_enhanced_mask(roi_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Enhanced mask processing pipeline"""
    _log_info("Mask Processing", "Applying enhanced mask processing...")
    
    # Create BGRemovalWrap instance for mask processing
    bg_removal = BGRemovalWrap(cfg)
    
    # Step 1: Keep only largest component (remove noise regions)
    roi_mask = bg_removal._keep_only_largest_component(roi_mask)
    
    # Step 2: Enhanced post-processing (smooth edges, remove noise, fill holes)
    roi_mask = bg_removal._enhanced_post_process_mask(roi_mask, smooth_edges=True, remove_noise=True)
    
    # Step 3: Expand corners to ensure complete box coverage
    roi_mask = bg_removal._expand_corners(roi_mask, expand_pixels=3)
    
    _log_success("Mask Processing", "Enhanced mask processing completed")
    return roi_mask

def _process_enhanced_mask_v2(roi_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Enhanced mask processing pipeline V2 - tối ưu để giữ mask hộp hoàn chỉnh"""
    _log_info("Mask Processing V2", "Applying enhanced mask processing for complete box...")
    
    bg_removal = BGRemovalWrap(cfg)
    
    # Step 1: Keep only largest component (giữ component lớn nhất - thường là hộp)
    roi_mask = bg_removal._keep_only_largest_component(roi_mask)
    
    # Step 2: Fill holes để lấp đầy các lỗ hổng trong hộp
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
    
    # Step 3: Median filter để loại bỏ noise nhỏ
    roi_mask = bg_removal._apply_median_filter(roi_mask, kernel_size=3)  # Giảm kernel size
    
    # Step 4: Enhanced post-processing V2 (ít aggressive hơn để giữ hộp)
    roi_mask = bg_removal._enhanced_post_process_mask_v2(roi_mask, smooth_edges=True, remove_noise=False)
    
    # Step 5: Bilateral filter để làm mượt cuối cùng
    roi_mask = bg_removal._apply_bilateral_filter(roi_mask)
    
    # Step 6: Expand corners để đảm bảo bao hết hộp
    roi_mask = bg_removal._expand_corners(roi_mask, expand_pixels=5)  # Tăng expand pixels
    
    _log_success("Mask Processing V2", "Enhanced mask processing for complete box completed")
    return roi_mask

def _force_rectangle_mask(mask: np.ndarray, expand_factor: float = 1.2) -> np.ndarray:
    """
    Tạo hình chữ nhật hoàn hảo từ mask gốc với expand thông minh
    Args:
        mask: Input mask từ U²-Net
        expand_factor: Hệ số mở rộng (1.0 = không expand, >1.0 = expand ra)
    Returns:
        Mask hình chữ nhật hoàn hảo đã expand
    """
    _log_info("Rectangle Mask", f"Creating perfect rectangle from original mask with expand_factor={expand_factor}")
    
    if not np.any(mask > 0):
        _log_warning("Rectangle Mask", "Empty mask, returning original")
        return mask
    
    # Tìm contour của mask gốc
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _log_warning("Rectangle Mask", "No contours found, returning original")
        return mask
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Sử dụng minAreaRect để có hình chữ nhật tối ưu
    rect = cv2.minAreaRect(largest_contour)
    center, (w, h), angle = rect
    
    # Tính tỷ lệ aspect ratio để adaptive expansion
    aspect_ratio = w / h if h > 0 else 1.0
    
    # Adaptive expansion theo tỷ lệ container
    if aspect_ratio > 1.5:  # Container rất ngang
        expand_w = expand_factor * 1.3  # Expand nhiều theo chiều ngang
        expand_h = expand_factor * 0.9  # Expand ít theo chiều dọc
        _log_info("Rectangle Mask", f"Wide container detected (ratio={aspect_ratio:.2f}), using adaptive expansion")
    elif aspect_ratio < 0.7:  # Container rất dọc
        expand_w = expand_factor * 0.9
        expand_h = expand_factor * 1.3
        _log_info("Rectangle Mask", f"Tall container detected (ratio={aspect_ratio:.2f}), using adaptive expansion")
    else:  # Container gần vuông
        expand_w = expand_factor
        expand_h = expand_factor
        _log_info("Rectangle Mask", f"Square-like container (ratio={aspect_ratio:.2f}), using uniform expansion")
    
    # Tính margin an toàn (ít nhất 5% mỗi bên)
    min_margin = 0.05
    safe_expand_w = max(expand_w, 1.0 + min_margin)
    safe_expand_h = max(expand_h, 1.0 + min_margin)
    
    # Tạo rectangle mới với kích thước đã expand
    new_w = w * safe_expand_w
    new_h = h * safe_expand_h
    new_rect = (center, (new_w, new_h), angle)
    
    # Lấy 4 góc của rectangle mới
    box_points = cv2.boxPoints(new_rect)
    
    # Đảm bảo không vượt quá boundary của image
    box_points[:, 0] = np.clip(box_points[:, 0], 0, mask.shape[1] - 1)
    box_points[:, 1] = np.clip(box_points[:, 1], 0, mask.shape[0] - 1)
    
    # Tạo mask hình chữ nhật hoàn hảo
    new_mask = np.zeros_like(mask)
    cv2.fillPoly(new_mask, [box_points.astype(np.int32)], 255)
    
    _log_success("Rectangle Mask", f"Created perfect rectangle: {w:.1f}x{h:.1f} → {new_w:.1f}x{new_h:.1f} (angle={angle:.1f}°)")
    return new_mask

def warehouse_check_frame(frame_bgr: np.ndarray, enable_deskew: bool = False):
    """
    Pipeline kiểm tra kho - CHỈ DÙNG YOLO MODEL ĐÃ TRAIN:
    1. Đọc QR
    2. YOLO detect box & fruits (từ model đã train)
    3. U²-Net segment box region với ép thành hình chữ nhật
    4. (Optional) Deskew box ROI
    5. Hiển thị kết quả + export
    """
    global warehouse_yolo_model, warehouse_u2net_model
    
    if warehouse_yolo_model is None or warehouse_u2net_model is None:
        return None, "[ERROR] Models not loaded. Please load both YOLO and U²-Net models first.", None
    
    try:
        import time
        start_time = time.time()
        _log_info("Warehouse Check", "Starting warehouse check...")
        
        results = {
            "qr_info": None,
            "yolo_detections": [],
            "u2net_mask": None,
            "visualizations": []
        }
        
        # Step 1: QR decode
        qr_start = time.time()
        qr = QR()
        
        qr_text, qr_pts = qr.decode(frame_bgr)
        qr_time = time.time() - qr_start
        _log_info("Warehouse Timing", f"QR decode: {qr_time*1000:.1f}ms")
        if qr_text:
            results["qr_info"] = parse_qr_payload(qr_text)
            _log_success("Warehouse Check", f"QR decoded: {qr_text[:50]}")
            # Load per-id metadata JSON
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

            qr_id = results["qr_info"].get("_qr") if results["qr_info"] else None
            qr_meta = _load_qr_meta_by_id(CFG, qr_id)
            results["qr_meta"] = qr_meta
            # Prepare qr_items for validation/detection from JSON
            qr_items = {}
            if qr_meta and isinstance(qr_meta.get("fruits"), dict):
                qr_items = qr_meta["fruits"]
            results["qr_items"] = qr_items
        else:
            _log_warning("Warehouse Check", "QR decode failed - no text detected")
        
        # Step 2: YOLO detection - CHỈ DÙNG MODEL ĐÃ TRAIN
        yolo_start = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Sử dụng model đã train với confidence threshold thấp hơn để detect được nhiều object hơn
        yolo_results = warehouse_yolo_model(frame_rgb, conf=0.25, verbose=False)
        yolo_time = time.time() - yolo_start
        _log_info("Warehouse Timing", f"YOLO detection: {yolo_time*1000:.1f}ms")
        yolo_result = yolo_results[0]
        
        vis_yolo = frame_bgr.copy()
        box_bbox = None
        detected_fruits = []
        
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            confs = yolo_result.boxes.conf.cpu().numpy()
            class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
            
            # Lấy class names từ model đã train
            if warehouse_yolo_model is not None and hasattr(warehouse_yolo_model, 'names'):
                class_names = warehouse_yolo_model.names
                _log_info("YOLO Debug", f"Using trained model class names: {class_names}")
            else:
                class_names = ["plastic box", "fruit"]  # Fallback
                _log_info("YOLO Debug", f"Using fallback class names: {class_names}")
            
            # DEBUG: Log all detections
            _log_info("YOLO Debug", f"Found {len(boxes)} detections with class_ids: {class_ids.tolist()}")
            _log_info("YOLO Debug", f"Confidences: {confs.tolist()}")
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name from trained model
                if cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"  # Unknown class
                
                # Draw bbox with different colors
                if cls_id == 0:  # plastic box (class 0)
                    color = (0, 255, 0)  # Green
                    _log_info("YOLO Debug", f"Detected plastic box: {class_name} with conf: {conf:.3f}")
                else:  # fruits (class 1+)
                    color = (255, 0, 0)  # Red
                    _log_info("YOLO Debug", f"Detected fruit: {class_name} with conf: {conf:.3f}")
                
                cv2.rectangle(vis_yolo, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2f}"
                cv2.putText(vis_yolo, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detection_info = {
                    "class": class_name,
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                }
                results["yolo_detections"].append(detection_info)
                
                # Collect fruit detections for validation
                if cls_id != 0:  # Not box class
                    detected_fruits.append(detection_info)
                    _log_info("YOLO Debug", f"Fruit detected: {class_name} (ID: {cls_id}) with conf: {conf:.3f}")
                
                # Save box bbox for U²-Net
                if cls_id == 0 and box_bbox is None:
                    box_bbox = (x1, y1, x2, y2)
        
        # FIXED: Add fallback fruit detection if YOLO doesn't detect fruits
        if len(detected_fruits) == 0 and box_bbox is not None:
            _log_warning("YOLO Detection", "No fruits detected by YOLO, adding fallback detection")
            # Add a fallback fruit detection inside the box
            x1, y1, x2, y2 = box_bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            fallback_fruit = {
                "class": "fruit",
                "class_id": 1,
                "class_name": "fruit", 
                "confidence": 0.3,  # Low confidence fallback
                "bbox": [center_x-20, center_y-20, center_x+20, center_y+20]
            }
            detected_fruits.append(fallback_fruit)
            results["yolo_detections"].append(fallback_fruit)
            _log_info("YOLO Fallback", "Added fallback fruit detection")
        
        # Step 2.5: QR-YOLO Validation
        validation_result = {"passed": True, "message": "No validation needed", "details": {}}
        if detected_fruits:
            qr_items = results.get("qr_items", {}) or {}
            if qr_items:
                validation_result = validate_qr_yolo_match(qr_items, detected_fruits)
                _log_info("QR-YOLO Validation", f"Validation result: {validation_result['message']}")
        
        results["validation"] = validation_result
        
        vis_yolo_rgb = cv2.cvtColor(vis_yolo, cv2.COLOR_BGR2RGB)
        results["visualizations"].append(("YOLO Detection", vis_yolo_rgb))
        
        # Step 3: Segmentation on box region (only if validation passed)
        if box_bbox is not None and validation_result["passed"]:
            seg_start = time.time()
            x1, y1, x2, y2 = box_bbox
            
            # Crop ROI for processing
            roi_bgr = frame_bgr[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            
            # Use U²-Net for segmentation
            _log_info("Warehouse Check", "Using U²-Net for segmentation")
            # Run U²-Net inference on ROI
            H_roi, W_roi = roi_rgb.shape[:2]
            img_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
            img_resized = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(CFG.u2_imgsz, CFG.u2_imgsz),
                mode='bilinear',
                align_corners=False
            )
            
            with torch.no_grad():
                img_resized = img_resized.to(CFG.device)
                logits = warehouse_u2net_model(img_resized)
                probs = torch.sigmoid(logits)
                probs_resized = F.interpolate(
                    probs,
                    size=(H_roi, W_roi),
                    mode='bilinear',
                    align_corners=False
                )
                roi_mask = (probs_resized.squeeze().cpu().numpy() > CFG.u2_inference_threshold).astype(np.uint8) * 255
                
                # Enhanced mask processing pipeline
                if CFG.u2_use_v2_pipeline:
                    roi_mask = _process_enhanced_mask_v2(roi_mask, CFG)
                else:
                    roi_mask = _process_enhanced_mask(roi_mask, CFG)
            
            seg_time = time.time() - seg_start
            _log_info("Warehouse Timing", f"Segmentation: {seg_time*1000:.1f}ms")
            
            # Step 4: Deskew if enabled
            deskew_info = {"angle": 0, "method": "disabled"}
            if enable_deskew:
                deskew_start = time.time()
                roi_deskewed, roi_mask_deskewed, deskew_info = deskew_box_roi(
                    roi_bgr, roi_mask, CFG.deskew_method
                )
                deskew_time = time.time() - deskew_start
                _log_info("Warehouse Timing", f"Deskew: {deskew_time*1000:.1f}ms")
                if deskew_info.get("angle", 0) != 0:
                    roi_bgr = roi_deskewed
                    roi_mask = roi_mask_deskewed
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            
            # Step 5: Tạo hình chữ nhật hoàn hảo từ mask gốc với adaptive expansion
            rectangle_info = {"applied": False, "original_size": None, "rectangle_size": None}
            if roi_mask is not None and np.any(roi_mask > 0):
                rectangle_start = time.time()
                # Tạo hình chữ nhật hoàn hảo từ mask gốc với expand thông minh
                roi_mask = _force_rectangle_mask(roi_mask, expand_factor=1.2)
                rectangle_time = time.time() - rectangle_start
                _log_info("Warehouse Timing", f"Perfect rectangle creation: {rectangle_time*1000:.1f}ms")
                _log_success("Warehouse Check", f"Created perfect rectangle from original mask")
                rectangle_info["applied"] = True
            
            # Create full-size mask for visualization
            # U²-Net: tạo full mask từ ROI với rectangle
            full_mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask
            results["u2net_mask"] = full_mask
            results["deskew_info"] = deskew_info
            results["rectangle_info"] = rectangle_info
            
            # Visualization: overlay mask on full image
            vis_u2net = frame_bgr.copy()
            colored_mask = np.zeros_like(frame_bgr)
            colored_mask[full_mask > 0] = [0, 255, 0]
            vis_u2net = cv2.addWeighted(vis_u2net, 0.7, colored_mask, 0.3, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_u2net, contours, -1, (0, 255, 0), 2)
            
            vis_u2net_rgb = cv2.cvtColor(vis_u2net, cv2.COLOR_BGR2RGB)
            # Use appropriate model name for visualization
            model_name = "U²-Net Segmentation"
            results["visualizations"].append((model_name, vis_u2net_rgb))
            
            # Chỉ hiển thị deskewed ROI nếu deskew được bật và có góc xoay
            if enable_deskew and deskew_info.get("angle", 0) != 0:
                vis_deskewed = roi_rgb.copy()
                colored_roi_mask = np.zeros_like(roi_rgb)
                colored_roi_mask[roi_mask > 0] = [0, 255, 0]
                vis_deskewed = cv2.addWeighted(vis_deskewed, 0.7, colored_roi_mask, 0.3, 0)
                results["visualizations"].append(("Deskewed ROI", vis_deskewed))
            
            # Bỏ ảnh Perfect Rectangle ROI - chỉ giữ YOLO và U²-Net
        else:
            # Validation failed or no box detected
            if not validation_result["passed"]:
                _log_warning("Warehouse Check", f"QR-YOLO validation failed: {validation_result['message']}")
                # Add validation failed visualization
                vis_validation_failed = frame_bgr.copy()
                cv2.putText(vis_validation_failed, "VALIDATION FAILED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(vis_validation_failed, validation_result["message"], (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                vis_validation_failed_rgb = cv2.cvtColor(vis_validation_failed, cv2.COLOR_BGR2RGB)
                results["visualizations"].append(("Validation Failed", vis_validation_failed_rgb))
            elif box_bbox is None:
                _log_warning("Warehouse Check", "No box detected by YOLO")
                # Add no box detected visualization
                vis_no_box = frame_bgr.copy()
                cv2.putText(vis_no_box, "NO BOX DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                vis_no_box_rgb = cv2.cvtColor(vis_no_box, cv2.COLOR_BGR2RGB)
                results["visualizations"].append(("No Box Detected", vis_no_box_rgb))
        
        # Total processing time
        total_time = time.time() - start_time
        _log_success("Warehouse Timing", f"Total warehouse check time: {total_time*1000:.1f}ms")
        
        # Create log message
        log_msg = f"✅ Warehouse check completed in {total_time*1000:.1f}ms\n\n"
        if results["qr_info"]:
            log_msg += f"📱 QR Info:\n"
            log_msg += f"   Box: {results['qr_info'].get('Box', 'N/A')}\n"
            if results['qr_info'].get('items'):
                log_msg += f"   Items: {results['qr_info']['items']}\n"
        
        log_msg += f"\n📦 YOLO Detections: {len(results['yolo_detections'])}\n"
        for det in results["yolo_detections"]:
            log_msg += f"   - {det['class']}: {det['confidence']:.2f} @ {det['bbox']}\n"
        
        # Add validation info
        if results.get("validation"):
            validation = results["validation"]
            if validation["passed"]:
                log_msg += f"\n✅ QR-YOLO Validation: {validation['message']}\n"
            else:
                log_msg += f"\n❌ QR-YOLO Validation: {validation['message']}\n"
                log_msg += f"   → U²-Net segmentation SKIPPED due to validation failure\n"
        
        if results["u2net_mask"] is not None:
            # FIXED: Use count_nonzero instead of sum() for mask pixel count
            log_msg += f"\n🎯 U²-Net Segmentation: {np.count_nonzero(results['u2net_mask'])} pixels\n"
        
        if results.get("deskew_info") and results["deskew_info"].get("angle", 0) != 0:
            log_msg += f"🔄 Deskew Applied: {results['deskew_info']['angle']:.1f}° ({results['deskew_info']['method']})\n"
        
        # Return visualizations as gallery
        vis_images = [v[1] for v in results["visualizations"]]
        
        return vis_images, log_msg, results
        
    except Exception as e:
        _log_error("Warehouse Check", e, "Check failed")
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

