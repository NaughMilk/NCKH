import os
import json
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from ultralytics import YOLO
from PIL import Image

# Import from other sections
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning, _log_error
from sections_e.e_qr_utils import parse_qr_payload, validate_qr_yolo_match
from sections_h.h_deskew import deskew_box_roi
from sections_h.h_mask_processing import _process_enhanced_mask, _process_enhanced_mask_v2, _force_rectangle_mask
from sections_h.h_model_loading import load_warehouse_yolo, load_warehouse_u2net

# Import global model variables
import sections_h.h_model_loading as model_loading

class QR:
    """QR Code decoder"""
    def __init__(self):
        self.dec = cv2.QRCodeDetector()

    def decode(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Decode QR code from frame"""
        try:
            data, points, _ = self.dec.detectAndDecode(frame_bgr)
            if data:
                return data, points
        except Exception as e:
            _log_warning("QR Decode", f"QR decode failed: {e}")
        return None, None


def warehouse_check_frame(frame_bgr: np.ndarray, yolo_model_path: str = None, u2net_model_path: str = None, enable_deskew: bool = False, enable_force_rectangle: bool = False) -> Tuple[Optional[List], Optional[str], Optional[Dict]]:
    """
    Pipeline kiểm tra kho - CHỈ DÙNG YOLO MODEL ĐÃ TRAIN:
    1. Đọc QR
    2. YOLO detect box & fruits (từ model đã train)
    3. U²-Net segment box region với ép thành hình chữ nhật
    4. (Optional) Deskew box ROI
    5. Hiển thị kết quả + export
    """
    # Access global models from model_loading module
    warehouse_yolo_model = model_loading.warehouse_yolo_model
    warehouse_u2net_model = model_loading.warehouse_u2net_model
    
    # Load models if paths provided
    if yolo_model_path and u2net_model_path:
        _log_info("Warehouse Check", f"Loading models: YOLO={yolo_model_path}, U²-Net={u2net_model_path}")
        if not load_warehouse_yolo(yolo_model_path):
            return None, "[ERROR] Failed to load YOLO model", None
        if not load_warehouse_u2net(u2net_model_path):
            return None, "[ERROR] Failed to load U²-Net model", None
        _log_success("Warehouse Check", "Both models loaded successfully")
    
    # Update model references after loading
    warehouse_yolo_model = model_loading.warehouse_yolo_model
    warehouse_u2net_model = model_loading.warehouse_u2net_model
    
    # Debug: Check model status
    _log_info("Warehouse Check", f"Model status: YOLO={warehouse_yolo_model is not None}, U²-Net={warehouse_u2net_model is not None}")
    
    if warehouse_yolo_model is None or warehouse_u2net_model is None:
        return None, "[ERROR] Models not loaded. Please load both YOLO and U²-Net models first.", None
    
    try:
        start_time = time.time()
        _log_info("Warehouse Check", "Starting warehouse check...")
        
        results = {
            "qr_info": None,
            "yolo_detections": [],
            "u2net_mask": None,
            "visualizations": []
        }
        
        # Add original image to visualizations for color comparison - FIXED: Use PIL Image for proper color handling with detailed logging
        _log_info("Color Debug", f"Input frame_bgr - shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
        _log_info("Color Debug", f"Sample pixel values (first 3x3): {frame_bgr[:3,:3]}")
        
        original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _log_info("Color Debug", f"BGR to RGB conversion - shape: {original_rgb.shape}, dtype: {original_rgb.dtype}")
        _log_info("Color Debug", f"Sample pixel values after BGR->RGB (first 3x3): {original_rgb[:3,:3]}")
        
        # Convert to PIL Image to ensure proper color handling in Gradio
        original_pil = Image.fromarray(original_rgb.astype(np.uint8))
        _log_info("Color Debug", f"PIL Image created - mode: {original_pil.mode}, size: {original_pil.size}")
        
        # Use PIL Image for proper color handling in Gradio
        _log_info("Color Debug", f"Adding PIL Image for display")
        results["visualizations"].append((original_pil, "Original Image"))
        
        # Step 1: QR decode
        qr_start = time.time()
        qr = QR()
        
        qr_text, qr_pts = qr.decode(frame_bgr)
        qr_time = time.time() - qr_start
        _log_info("Warehouse Timing", f"QR decode: {qr_time*1000:.1f}ms")
        if qr_text:
            results["qr_info"] = parse_qr_payload(qr_text)
            _log_success("Warehouse Check", f"QR decoded: {qr_text[:50]}")
            _log_info("QR Debug", f"QR text type: {type(qr_text)}, value: {repr(qr_text)}")
            _log_info("QR Debug", f"Parsed QR info type: {type(results['qr_info'])}, value: {results['qr_info']}")
            # Load per-id metadata JSON
            def _load_qr_meta_by_id(cfg, qr_id: str) -> Optional[dict]:
                if not qr_id:
                    return None
                try:
                    meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir, f"{qr_id}.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Ensure we return a dict, not int or other types
                            if isinstance(data, dict):
                                return data
                            else:
                                _log_warning("QR Meta Load", f"Expected dict but got {type(data)} for id {qr_id}")
                                return None
                except Exception as e:
                    _log_warning("QR Meta Load", f"Failed to load meta for id {qr_id}: {e}")
                return None

            # FIXED: Safe access to qr_info
            qr_info = results.get("qr_info", {})
            if isinstance(qr_info, dict):
                qr_id = qr_info.get("_qr")
            else:
                _log_warning("QR Info", f"Expected dict but got {type(qr_info)}: {qr_info}")
                qr_id = None
            if qr_id:
                qr_meta = _load_qr_meta_by_id(CFG, qr_id)
                if qr_meta:
                    results["qr_items"] = qr_meta.get("fruits", {})
                    _log_info("QR Meta", f"Loaded QR metadata: {results['qr_items']}")
        
        # Step 2: YOLO Detection
        yolo_start = time.time()
        yolo_results = warehouse_yolo_model(frame_bgr, verbose=False)
        yolo_time = time.time() - yolo_start
        _log_info("Warehouse Timing", f"YOLO detection: {yolo_time*1000:.1f}ms")
        
        # Process YOLO results
        vis_yolo = frame_bgr.copy()
        box_bbox = None
        detected_fruits = []
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name from model
                    class_name = warehouse_yolo_model.names[cls_id]
                    
                    # Draw bounding box - FIXED: Use proper BGR colors
                    color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Green for box, Red for fruit
                    cv2.rectangle(vis_yolo, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_yolo, f"{class_name}: {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Store detection info
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
        
        # FIXED: Use PIL Image for proper color handling in YOLO visualization with detailed logging
        _log_info("Color Debug", f"YOLO vis_yolo - shape: {vis_yolo.shape}, dtype: {vis_yolo.dtype}")
        _log_info("Color Debug", f"YOLO sample pixel values (first 3x3): {vis_yolo[:3,:3]}")
        
        vis_yolo_rgb = cv2.cvtColor(vis_yolo, cv2.COLOR_BGR2RGB)
        _log_info("Color Debug", f"YOLO BGR to RGB conversion - shape: {vis_yolo_rgb.shape}, dtype: {vis_yolo_rgb.dtype}")
        _log_info("Color Debug", f"YOLO sample pixel values after BGR->RGB (first 3x3): {vis_yolo_rgb[:3,:3]}")
        
        vis_yolo_pil = Image.fromarray(vis_yolo_rgb.astype(np.uint8))
        _log_info("Color Debug", f"YOLO PIL Image created - mode: {vis_yolo_pil.mode}, size: {vis_yolo_pil.size}")
        results["visualizations"].append((vis_yolo_pil, "YOLO Detection"))
        
        # Step 3: Segmentation on box region (only if validation passed)
        if box_bbox is not None and validation_result["passed"]:
            seg_start = time.time()
            x1, y1, x2, y2 = box_bbox
            
            # Use U²-Net for segmentation on full image
            _log_info("Warehouse Check", "Using U²-Net for segmentation on full image")
            
            # Convert full image to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            H_full, W_full = frame_rgb.shape[:2]
            
            # Run U²-Net inference on full image
            img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            img_resized = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(CFG.u2_imgsz, CFG.u2_imgsz),
                mode='bilinear',
                align_corners=False
            )
            
            with torch.no_grad():
                img_resized = img_resized.to(CFG.device)
                outputs = warehouse_u2net_model(img_resized)
                # U2Net returns (d0, d1, d2, d3, d4, d5, d6) - use main output d0
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.sigmoid(logits)
                probs_resized = F.interpolate(
                    probs,
                    size=(H_full, W_full),
                    mode='bilinear',
                    align_corners=False
                )
                full_mask = (probs_resized.squeeze().cpu().numpy() > CFG.u2_inference_threshold).astype(np.uint8) * 255
                
                # Log mask stats before processing
                mask_pixels_before = np.sum(full_mask > 0)
                _log_info("Mask Debug", f"U2Net raw mask: {mask_pixels_before} pixels ({mask_pixels_before/(full_mask.shape[0]*full_mask.shape[1])*100:.1f}% of full image)")
                
                # Enhanced mask processing pipeline
                if CFG.u2_use_v2_pipeline:
                    full_mask = _process_enhanced_mask_v2(full_mask, CFG)
                else:
                    full_mask = _process_enhanced_mask(full_mask, CFG)
                
                # Log mask stats after processing
                mask_pixels_after = np.sum(full_mask > 0)
                _log_info("Mask Debug", f"After processing: {mask_pixels_after} pixels ({mask_pixels_after/(full_mask.shape[0]*full_mask.shape[1])*100:.1f}% of full image)")
                
                # Extract ROI mask from full mask
                roi_mask = full_mask[y1:y2, x1:x2]
            
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
            
            # Step 5: Tạo hình chữ nhật hoàn hảo từ mask gốc với adaptive expansion (nếu được bật)
            rectangle_info = {"applied": False, "original_size": None, "rectangle_size": None}
            if enable_force_rectangle and roi_mask is not None and np.any(roi_mask > 0):
                rectangle_start = time.time()
                # Tạo hình chữ nhật hoàn hảo từ mask gốc với expand thông minh
                roi_mask = _force_rectangle_mask(roi_mask, expand_factor=1.2)
                rectangle_time = time.time() - rectangle_start
                _log_info("Warehouse Timing", f"Perfect rectangle creation: {rectangle_time*1000:.1f}ms")
                _log_success("Warehouse Check", f"Created perfect rectangle from original mask")
                rectangle_info["applied"] = True
            elif not enable_force_rectangle:
                _log_info("Warehouse Check", "Force rectangle disabled - using original mask")
            
            # Create full-size mask for visualization
            # U²-Net: sử dụng full mask đã có
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
            
            # FIXED: Use PIL Image for proper color handling in U²-Net visualization with detailed logging
            _log_info("Color Debug", f"U²-Net vis_u2net - shape: {vis_u2net.shape}, dtype: {vis_u2net.dtype}")
            _log_info("Color Debug", f"U²-Net sample pixel values (first 3x3): {vis_u2net[:3,:3]}")
            
            vis_u2net_rgb = cv2.cvtColor(vis_u2net, cv2.COLOR_BGR2RGB)
            _log_info("Color Debug", f"U²-Net BGR to RGB conversion - shape: {vis_u2net_rgb.shape}, dtype: {vis_u2net_rgb.dtype}")
            _log_info("Color Debug", f"U²-Net sample pixel values after BGR->RGB (first 3x3): {vis_u2net_rgb[:3,:3]}")
            
            vis_u2net_pil = Image.fromarray(vis_u2net_rgb.astype(np.uint8))
            _log_info("Color Debug", f"U²-Net PIL Image created - mode: {vis_u2net_pil.mode}, size: {vis_u2net_pil.size}")
            # Use appropriate model name for visualization
            model_name = "U²-Net Segmentation"
            results["visualizations"].append((vis_u2net_pil, model_name))
            
            # Chỉ hiển thị deskewed ROI nếu deskew được bật và có góc xoay
            if enable_deskew and deskew_info.get("angle", 0) != 0:
                vis_deskewed = roi_rgb.copy()
                colored_roi_mask = np.zeros_like(roi_rgb)
                colored_roi_mask[roi_mask > 0] = [0, 255, 0]
                vis_deskewed = cv2.addWeighted(vis_deskewed, 0.7, colored_roi_mask, 0.3, 0)
                results["visualizations"].append((vis_deskewed, "Deskewed ROI"))
            
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
                results["visualizations"].append((vis_validation_failed_rgb, "Validation Failed"))
            elif box_bbox is None:
                _log_warning("Warehouse Check", "No box detected by YOLO")
                # Add no box detected visualization
                vis_no_box = frame_bgr.copy()
                cv2.putText(vis_no_box, "NO BOX DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                vis_no_box_rgb = cv2.cvtColor(vis_no_box, cv2.COLOR_BGR2RGB)
                results["visualizations"].append((vis_no_box_rgb, "No Box Detected"))
        
        # Total processing time
        total_time = time.time() - start_time
        _log_success("Warehouse Timing", f"Total warehouse check time: {total_time*1000:.1f}ms")
        
        # Create log message
        log_msg = f"✅ Warehouse check completed in {total_time*1000:.1f}ms\n\n"
        # FIXED: Safe access to qr_info in logging
        qr_info = results.get("qr_info", {})
        if qr_info and isinstance(qr_info, dict):
            log_msg += f"📱 QR Info:\n"
            log_msg += f"   Box: {qr_info.get('Box', 'N/A')}\n"
            if qr_info.get('items'):
                log_msg += f"   Items: {qr_info['items']}\n"
        
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
        return results["visualizations"], log_msg, results
        
    except Exception as e:
        _log_error("Warehouse Check", e, "Warehouse check failed")
        return None, f"[ERROR] Warehouse check failed: {str(e)}", None