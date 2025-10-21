# ========================= SECTION H: WAREHOUSE CORE ========================= #

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import traceback
import json
from typing import Dict, Any, List, Tuple, Optional

# Import dependencies
from sections_a.a_config import Config, CFG, _log_info, _log_success, _log_warning, _log_error
from sections_e.e_qr_detection import QR
from sections_e.e_qr_utils import parse_qr_payload, validate_qr_yolo_match
from sections_h.h_model_loading import warehouse_yolo_model, warehouse_u2net_model
from sections_h.h_deskew import deskew_box_roi
from sections_h.h_mask_processing import _process_enhanced_mask, _process_enhanced_mask_v2, _force_rectangle_mask

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
