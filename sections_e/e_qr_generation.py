import cv2
import numpy as np
import qrcode
from typing import Dict, Any

def generate_qr_code(box_id: str, fruits: Dict[str, int], metadata: Dict[str, Any] = None) -> np.ndarray:
    """Generate QR code with box and fruit information"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Create payload
    payload = {
        "box_id": box_id,
        "fruits": fruits,
        "metadata": metadata or {}
    }
    
    # Convert to string
    import json
    qr_text = json.dumps(payload, ensure_ascii=False)
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    # Create image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img)
    
    _log_info("QR Generation", f"Generated QR for box {box_id} with {len(fruits)} fruit types")
    return qr_array

def generate_qr_with_metadata(cfg, box_id: str, fruits: Dict[str, int], 
                             metadata: Dict[str, Any] = None) -> np.ndarray:
    """Generate QR code with enhanced metadata and rendering"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Create enhanced payload
    enhanced_metadata = {
        "timestamp": cfg.timestamp if hasattr(cfg, 'timestamp') else None,
        "session_id": cfg.session_id if hasattr(cfg, 'session_id') else None,
        "version": "1.0",
        **(metadata or {})
    }
    
    payload = {
        "box_id": box_id,
        "fruits": fruits,
        "metadata": enhanced_metadata
    }
    
    # Convert to string
    import json
    qr_text = json.dumps(payload, ensure_ascii=False)
    
    # Generate QR code with higher error correction
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    # Create image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img)
    
    # Convert to RGB for better rendering
    qr_rgb = cv2.cvtColor(qr_array, cv2.COLOR_GRAY2RGB)
    
    # Add caption with box name and fruit info
    box_name = f"Box {box_id}"
    fruit_type_str = ", ".join([f"{k}: {v}" for k, v in fruits.items()])
    total_qty = sum(fruits.values())
    
    qr_with_caption = _render_qr_with_caption(qr_rgb, box_name, fruit_type_str, total_qty)
    
    _log_info("QR Generation", f"Generated enhanced QR for box {box_id} with {len(fruits)} fruit types")
    return qr_with_caption

def _render_qr_with_caption(qr_rgb: np.ndarray, box_name: str, fruit_type_str: str, total_qty: int) -> np.ndarray:
    """Render QR code with caption text"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # Calculate dimensions
    qr_h, qr_w = qr_rgb.shape[:2]
    caption_h = 80
    total_h = qr_h + caption_h
    
    # Create canvas
    canvas = np.ones((total_h, qr_w, 3), dtype=np.uint8) * 255
    
    # Place QR code
    canvas[:qr_h, :qr_w] = qr_rgb
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 0, 0)
    thickness = 1
    
    # Box name
    text_y = qr_h + 25
    cv2.putText(canvas, box_name, (10, text_y), font, font_scale, color, thickness)
    
    # Fruit info
    text_y += 25
    fruit_text = f"Fruits: {fruit_type_str} (Total: {total_qty})"
    cv2.putText(canvas, fruit_text, (10, text_y), font, font_scale, color, thickness)
    
    _log_info("QR Rendering", f"Rendered QR with caption: {box_name}")
    return canvas
