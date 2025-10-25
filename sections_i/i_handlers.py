import os
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Import from other sections
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning, _log_error
from sections_h.h_warehouse_core import load_warehouse_yolo, load_warehouse_u2net, warehouse_check_frame

def handle_capture(cam_image, img_upload, video_path, supplier_id=None):
    """Handle camera capture, image upload, or video upload"""
    try:
        if cam_image is not None:
            # Process camera image
            try:
                from sections_g.g_sdy_core import SDYPipeline
                from sections_a.a_config import CFG
                pipeline = SDYPipeline(CFG)
                result = pipeline.process_frame(cam_image, preview_only=False, save_dataset=True, return_both_visualizations=True)
                if len(result) == 6:
                    original_img, gdino_vis, seg_vis, meta, img_path, lab_path = result
                else:
                    # Handle case when image is rejected (returns 5 values)
                    original_img, gdino_vis, seg_vis, meta, img_path = result
                    lab_path = None
                if original_img is not None and gdino_vis is not None and seg_vis is not None:
                    # Return GroundingDINO detection and segmentation
                    print(f"[DEBUG] Camera: gdino_vis shape: {gdino_vis.shape if gdino_vis is not None else 'None'}")
                    print(f"[DEBUG] Camera: seg_vis shape: {seg_vis.shape if seg_vis is not None else 'None'}")
                    print(f"[DEBUG] Camera: gdino_vis type: {type(gdino_vis)}")
                    print(f"[DEBUG] Camera: seg_vis type: {type(seg_vis)}")
                    return [gdino_vis, seg_vis], f"[SUCCESS] Camera image processed: {meta.get('message', 'Processed successfully')}", None
                else:
                    print(f"[DEBUG] Camera: original_img={original_img is not None}, gdino_vis={gdino_vis is not None}, seg_vis={seg_vis is not None}")
                    return [cam_image], "[SUCCESS] Camera image captured (no processing)", None
            except Exception as e:
                _log_error("Camera Processing", e, "Camera processing failed, returning original")
                return [cam_image], "[SUCCESS] Camera image captured", None
        elif img_upload is not None:
            # Process uploaded image
            try:
                from sections_g.g_sdy_core import SDYPipeline
                from sections_a.a_config import CFG
                pipeline = SDYPipeline(CFG)
                result = pipeline.process_frame(img_upload, preview_only=False, save_dataset=True, return_both_visualizations=True)
                if len(result) == 6:
                    original_img, gdino_vis, seg_vis, meta, img_path, lab_path = result
                else:
                    # Handle case when image is rejected (returns 5 values)
                    original_img, gdino_vis, seg_vis, meta, img_path = result
                    lab_path = None
                if original_img is not None and gdino_vis is not None and seg_vis is not None:
                    # Return GroundingDINO detection and segmentation
                    print(f"[DEBUG] Image: gdino_vis shape: {gdino_vis.shape if gdino_vis is not None else 'None'}")
                    print(f"[DEBUG] Image: seg_vis shape: {seg_vis.shape if seg_vis is not None else 'None'}")
                    print(f"[DEBUG] Image: gdino_vis type: {type(gdino_vis)}")
                    print(f"[DEBUG] Image: seg_vis type: {type(seg_vis)}")
                    return [gdino_vis, seg_vis], f"[SUCCESS] Image processed: {meta.get('message', 'Processed successfully')}", None
                else:
                    print(f"[DEBUG] Image: original_img={original_img is not None}, gdino_vis={gdino_vis is not None}, seg_vis={seg_vis is not None}")
                    return [img_upload], "[SUCCESS] Image uploaded (no processing)", None
            except Exception as e:
                _log_error("Image Processing", e, "Image processing failed, returning original")
                return [img_upload], "[SUCCESS] Image uploaded", None
        elif video_path is not None:
            # Process video to extract frames
            try:
                from sections_a.a_video import extract_gallery_from_video
                from sections_a.a_config import CFG
                from sections_a.a_edges import EDGE
                from sections_f.f_dataset_utils import cleanup_dataset_files
                
                # Extract frames from video with default parameters
                result = extract_gallery_from_video(
                    video_path=video_path,
                    cfg=CFG,
                    backend=EDGE,
                    canny_lo=CFG.canny_lo,
                    canny_hi=CFG.canny_hi,
                    dexi_thr=CFG.video_dexi_thr,
                    dilate_iters=CFG.video_dilate_iters,
                    close_kernel=CFG.video_close_kernel,
                    min_area_ratio=CFG.video_min_area_ratio,
                    rect_score_min=CFG.video_rect_score_min,
                    ar_min=CFG.video_ar_min,
                    ar_max=CFG.video_ar_max,
                    erode_inner=CFG.video_erode_inner,
                    smooth_close=CFG.video_smooth_close,
                    smooth_open=CFG.video_smooth_open,
                    use_hull=CFG.video_use_hull,
                    rectify_mode=CFG.video_rectify_mode,
                    rect_pad=CFG.video_rect_pad,
                    expand_factor=CFG.video_expand_factor,
                    mode=CFG.video_mode,
                    min_comp_area=CFG.video_min_comp_area,
                    show_green_frame=CFG.video_show_green_frame,
                    frame_step=CFG.video_frame_step,
                    max_frames=CFG.video_max_frames,
                    keep_only_detected=True,
                    use_pair_filter=True,
                    pair_min_gap=CFG.video_pair_min_gap,
                    pair_max_gap=CFG.video_pair_max_gap,
                    lock_enable=CFG.video_lock_enable,
                    lock_n_warmup=CFG.video_lock_n_warmup,
                    lock_trim=CFG.video_lock_trim,
                    lock_pad=CFG.video_lock_pad
                )
                
                if result is None:
                    print("[ERROR] extract_gallery_from_video returned None")
                    return [], "[ERROR] Video processing failed", None
                
                images, message = result
                
                # Cleanup dataset sau khi xử lý video (backup cleanup)
                print("\n🧹 Đang cleanup dataset (backup)...")
                cleanup_result = cleanup_dataset_files()
                if cleanup_result["status"] == "clean":
                    print("✅ Dataset đã sạch, không cần cleanup")
                elif cleanup_result["status"] == "cleaned":
                    print(f"✅ Đã cleanup: {cleanup_result['deleted']} files thừa")
                
                if images:
                    return images, f"[SUCCESS] Video processed: {len(images)} frames extracted", None
                else:
                    return [], f"[WARNING] Video processed but no frames extracted: {message}", None
                    
            except Exception as video_error:
                import traceback
                error_details = traceback.format_exc()
                print(f"[ERROR] Video processing failed: {str(video_error)}")
                print(f"[ERROR] Full traceback: {error_details}")
                _log_error("Video Processing", video_error, "Video processing failed")
                return [], f"[ERROR] Video processing failed: {video_error}", None
        else:
            return [], "[ERROR] No image or video provided", None
    except Exception as e:
        _log_error("Capture", e, "Camera capture failed")
        return [], f"[ERROR] Capture failed: {e}", None

def handle_multiple_uploads(images, videos, supplier_id=None):
    """Handle multiple file uploads (legacy compatibility)"""
    try:
        results = []
        if images:
            for img in images:
                results.append((img, "[SUCCESS] Image processed"))
        if videos:
            for vid in videos:
                results.append((vid, "[SUCCESS] Video processed"))
        return results, "[SUCCESS] Multiple files processed", None
    except Exception as e:
        _log_error("Multiple Uploads", e, "Multiple uploads failed")
        return [], f"[ERROR] Multiple uploads failed: {e}", None

def handle_qr_generation(box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count, 
                        fruit_type="", quantity=0, note=""):
    """Handle QR code generation (legacy compatibility)"""
    try:
        # Simple QR generation logic
        qr_data = {
            "box_id": box_id,
            "fruits": {
                fruit1_name: fruit1_count,
                fruit2_name: fruit2_count
            },
            "fruit_type": fruit_type,
            "quantity": quantity,
            "note": note
        }
        
        # Create simple QR visualization
        qr_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.putText(qr_image, f"Box: {box_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(qr_image, f"Fruits: {fruit1_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(qr_image, f"Count: {fruit1_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return qr_image, str(qr_data), None, None
    except Exception as e:
        _log_error("QR Generation", e, "QR generation failed")
        return None, f"[ERROR] QR generation failed: {e}", None, None

def handle_warehouse_upload(uploaded_image, yolo_model_path: str, u2net_model_path: str, 
                          enable_deskew: bool = False, deskew_method: str = "minAreaRect", enable_force_rectangle: bool = False, rect_padding: int = 10) -> Tuple[Optional[List], Optional[str], Optional[Dict]]:
    """
    Handle warehouse check with uploaded image and model paths
    
    Args:
        uploaded_image: Uploaded image file
        yolo_model_path: Path to YOLO model file
        u2net_model_path: Path to U²-Net model file
        enable_deskew: Whether to enable deskew
        deskew_method: Deskew method to use
        enable_force_rectangle: Whether to force rectangular mask
        
    Returns:
        Tuple of (visualizations, log_message, results)
    """
    try:
        # Validate inputs
        if uploaded_image is None:
            return None, "[ERROR] No image uploaded", None
        
        if not yolo_model_path or not os.path.exists(yolo_model_path):
            return None, "[ERROR] YOLO model path is invalid or file not found", None
            
        if not u2net_model_path or not os.path.exists(u2net_model_path):
            return None, "[ERROR] U²-Net model path is invalid or file not found", None
        
        # Models will be loaded in warehouse_check_frame
        _log_info("Warehouse Handler", "Models will be loaded during processing...")
        
        # Update deskew config if needed
        if enable_deskew:
            CFG.enable_deskew = True
            CFG.deskew_method = deskew_method
            _log_info("Warehouse Handler", f"Deskew enabled with method: {deskew_method}")
        
        # Process image
        _log_info("Warehouse Handler", "Processing uploaded image...")
        
        # Convert uploaded image to numpy array - FIXED: Handle color channels properly with detailed logging
        if hasattr(uploaded_image, 'name'):
            # File path
            image_path = uploaded_image.name
            frame_bgr = cv2.imread(image_path)
            _log_info("Color Debug", f"Loaded from file: {image_path}")
            _log_info("Color Debug", f"cv2.imread result - shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}, channels: {frame_bgr.shape[2] if len(frame_bgr.shape) == 3 else 'N/A'}")
            _log_info("Color Debug", f"cv2.imread assumes BGR format")
        else:
            # Already numpy array or PIL Image
            if isinstance(uploaded_image, np.ndarray):
                _log_info("Color Debug", f"Uploaded as numpy array - shape: {uploaded_image.shape}, dtype: {uploaded_image.dtype}")
                _log_info("Color Debug", f"Sample pixel values from uploaded numpy: {uploaded_image[:3,:3]}")
                
                # Check if it's grayscale (2D) or color (3D)
                if len(uploaded_image.shape) == 2:
                    _log_warning("Color Debug", "Uploaded image is grayscale (2D), converting to BGR")
                    frame_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_GRAY2BGR)
                elif len(uploaded_image.shape) == 3:
                    if uploaded_image.shape[2] == 3:
                        # Assume it's RGB from Gradio upload
                        _log_info("Color Debug", "3-channel image detected, assuming RGB format from Gradio")
                        frame_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)
                    elif uploaded_image.shape[2] == 4:
                        # RGBA image
                        _log_info("Color Debug", "4-channel RGBA image detected, converting to BGR")
                        frame_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGBA2BGR)
                    else:
                        _log_warning("Color Debug", f"Unknown channel count: {uploaded_image.shape[2]}, using as-is")
                        frame_bgr = uploaded_image
                else:
                    _log_error("Color Debug", f"Invalid image shape: {uploaded_image.shape}")
                    frame_bgr = uploaded_image
                
                _log_info("Color Debug", f"Final frame_bgr - shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
                _log_info("Color Debug", f"Final sample pixel values: {frame_bgr[:3,:3]}")
            else:
                # PIL Image - FIXED: Keep as RGB, don't convert to BGR
                _log_info("Color Debug", f"Uploaded as PIL Image - type: {type(uploaded_image)}")
                frame_rgb = np.array(uploaded_image)
                _log_info("Color Debug", f"PIL to numpy - shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
                _log_info("Color Debug", f"PIL Image is RGB format")
                # Convert RGB to BGR for OpenCV processing
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                _log_info("Color Debug", f"RGB to BGR conversion - shape: {frame_bgr.shape}, dtype: {frame_bgr.dtype}")
                _log_info("Color Debug", f"Sample pixel values (first 3x3): RGB={frame_rgb[:3,:3]}, BGR={frame_bgr[:3,:3]}")
        
        if frame_bgr is None:
            return None, "[ERROR] Failed to load image", None
        
        _log_info("Warehouse Handler", f"Image loaded: {frame_bgr.shape}")
        
        # Run warehouse check
        start_time = time.time()
        visualizations, log_message, results = warehouse_check_frame(frame_bgr, yolo_model_path, u2net_model_path, enable_deskew, enable_force_rectangle, rect_padding)
        processing_time = time.time() - start_time
        
        _log_success("Warehouse Handler", f"Processing completed in {processing_time*1000:.1f}ms")
        
        return visualizations, log_message, results
        
    except Exception as e:
        _log_error("Warehouse Handler", e, "Warehouse upload processing failed")
        return None, f"[ERROR] Warehouse processing failed: {str(e)}", None

def handle_warehouse_model_upload(yolo_model_file, u2net_model_file) -> Tuple[bool, str]:
    """
    Handle model file uploads and save them to appropriate locations
    
    Args:
        yolo_model_file: YOLO model file
        u2net_model_file: U²-Net model file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if yolo_model_file is None or u2net_model_file is None:
            return False, "[ERROR] Please upload both YOLO and U²-Net model files"
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(CFG.project_dir, "warehouse_models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save YOLO model
        yolo_path = os.path.join(models_dir, "yolo_model.pt")
        if hasattr(yolo_model_file, 'name'):
            # Copy file
            import shutil
            shutil.copy2(yolo_model_file.name, yolo_path)
        else:
            # Save from file object
            with open(yolo_path, 'wb') as f:
                f.write(yolo_model_file.read())
        
        # Save U²-Net model
        u2net_path = os.path.join(models_dir, "u2net_model.pth")
        if hasattr(u2net_model_file, 'name'):
            # Copy file
            import shutil
            shutil.copy2(u2net_model_file.name, u2net_path)
        else:
            # Save from file object
            with open(u2net_path, 'wb') as f:
                f.write(u2net_model_file.read())
        
        _log_success("Warehouse Models", f"Models saved to {models_dir}")
        return True, f"✅ Models uploaded successfully!\n\nYOLO: {yolo_path}\nU²-Net: {u2net_path}"
        
    except Exception as e:
        _log_error("Warehouse Models", e, "Failed to upload models")
        return False, f"[ERROR] Failed to upload models: {str(e)}"
