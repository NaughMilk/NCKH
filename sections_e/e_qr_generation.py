import cv2
import numpy as np
import qrcode
from typing import Dict, Any, Tuple

def generate_qr_code(qr_id: str) -> np.ndarray:
    """Generate QR code that contains ONLY QR ID"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    # QR only contains the ID
    qr_text = qr_id
    
    # Generate QR code - Smaller dots but normal size
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,  # Normal size but smaller dots
        border=3,    # Normal border
    )
    qr.add_data(qr_text)
    qr.make(fit=True)
    
    # Create image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img)
    
    # Ensure correct dtype and format for Gradio
    if qr_array.dtype == bool:
        qr_array = qr_array.astype(np.uint8) * 255
    
    if len(qr_array.shape) == 2:
        qr_array = cv2.cvtColor(qr_array, cv2.COLOR_GRAY2RGB)
    
    # Ensure uint8 dtype for Gradio compatibility
    qr_array = qr_array.astype(np.uint8)
    
    _log_info("QR Generation", f"Generated QR with ID: {qr_id}")
    return qr_array

def generate_qr_with_metadata(cfg, box_id: str, fruits: Dict[str, int], 
                             fruit_type: str = "", quantity: int = 0, note: str = "") -> Tuple[np.ndarray, str, str]:
    """Generate QR with unique id_qr; QR encodes ONLY id_qr; save editable JSON per id"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
        from sections_a.a_utils import generate_unique_box_name, generate_unique_qr_id, ensure_dir, atomic_write_text
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def generate_unique_box_name(cfg): return f"BOX-{int(time.time())}"
        def generate_unique_qr_id(cfg): return f"QR{int(time.time())}"
        def ensure_dir(path): os.makedirs(path, exist_ok=True)
        def atomic_write_text(path, text): 
            with open(path, 'w', encoding='utf-8') as f: f.write(text)
    
    # Generate unique box name if not provided
    if not box_id or box_id.strip() == "":
        box_id = generate_unique_box_name(cfg)
    elif not box_id.startswith(cfg.box_name_prefix):
        box_id = f"{cfg.box_name_prefix}{box_id}"
    
    # Generate unique 6-digit QR id
    qr_id = generate_unique_qr_id(cfg)

    # Create metadata with detailed format
    import datetime
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get primary fruit name (first fruit with count > 0)
    fruit_name = ""
    for name, count in fruits.items():
        if count > 0:
            fruit_name = name
            break
    
    metadata = {
        "id_qr": qr_id,
        "box_name": box_id,
        "fruit_name": fruit_name,
        "quantity": quantity,
        "fruits": fruits,
        "note": note,
        "created_at": created_at
    }
    
    # Generate QR code (only contains QR ID)
    qr_image = generate_qr_code(qr_id)
    
    # Save metadata to JSON file in sdy_project/qr_meta/{qr_id}/ folder
    import os
    import json
    import time
    qr_folder = os.path.join(cfg.project_dir, "qr_meta", qr_id)
    ensure_dir(qr_folder)
    meta_file = os.path.join(qr_folder, f"{qr_id}.json")
    atomic_write_text(meta_file, json.dumps(metadata, ensure_ascii=False, indent=2))
    
    # Create QR content string for display
    qr_content = json.dumps(metadata, ensure_ascii=False, indent=2)
    
    _log_success("QR Generation", f"Generated QR {qr_id} for box {box_id}")
    return qr_image, qr_content, meta_file

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
