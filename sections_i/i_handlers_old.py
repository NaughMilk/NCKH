import os
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Import from other sections
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning, _log_error
from sections_h.h_warehouse_core import load_warehouse_yolo, load_warehouse_u2net, warehouse_check_frame

def handle_capture(cam_image, img_upload, video_path, supplier_id=None):
    """Handle camera capture (legacy compatibility)"""
    try:
        if cam_image is not None:
            return cam_image, "[SUCCESS] Camera image captured"
        elif img_upload is not None:
            return img_upload, "[SUCCESS] Image uploaded"
                else:
            return None, "[ERROR] No image provided"
    except Exception as e:
        _log_error("Capture", e, "Camera capture failed")
        return None, f"[ERROR] Capture failed: {e}"

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
        return results
    except Exception as e:
        _log_error("Multiple Uploads", e, "Multiple uploads failed")
        return []

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
                          enable_deskew: bool = False, deskew_method: str = "minAreaRect", enable_force_rectangle: bool = False) -> Tuple[Optional[List], Optional[str], Optional[Dict]]:
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
        
        # Convert uploaded image to numpy array
        if hasattr(uploaded_image, 'name'):
            # File path
            image_path = uploaded_image.name
            frame_bgr = cv2.imread(image_path)
        else:
            # Already numpy array or PIL Image
            if isinstance(uploaded_image, np.ndarray):
                frame_bgr = uploaded_image
            else:
                # PIL Image
                frame_bgr = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        
        if frame_bgr is None:
            return None, "[ERROR] Failed to load image", None
        
        _log_info("Warehouse Handler", f"Image loaded: {frame_bgr.shape}")
        
        # Run warehouse check
        start_time = time.time()
        visualizations, log_message, results = warehouse_check_frame(frame_bgr, yolo_model_path, u2net_model_path, enable_deskew, enable_force_rectangle)
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
