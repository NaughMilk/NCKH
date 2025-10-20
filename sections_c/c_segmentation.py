import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def segment_box_by_boxprompt(bg_removal_wrap, img_rgb, box_xyxy):
    """Segment box - KHÔI PHỤC HOÀN TOÀN: Remove background toàn ảnh → Mask toàn bộ"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
        from .c_postprocess import _improve_mask_with_rectangle_fitting
    except ImportError:
        # Fallback if functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _improve_mask_with_rectangle_fitting(mask): return mask
    
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    
    # Clamp coordinates
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W-1, x2); y2 = min(H-1, y2)
    
    # BƯỚC 1: Remove background của toàn ảnh (giữ nguyên như ban đầu)
    full_pil = Image.fromarray(img_rgb)
    full_rgba = bg_removal_wrap.remove_background(full_pil, bg_color=None)
    
    # BƯỚC 2: Mask toàn bộ những gì không bị remove
    full_alpha = np.array(full_rgba)[:, :, 3]
    full_mask = (full_alpha > 128).astype(np.uint8)
    
    # BƯỚC 3: Cải thiện mask với rectangle fitting (mới)
    improved_mask = _improve_mask_with_rectangle_fitting(full_mask)
    
    _log_info("BG Removal", f"Box segmentation: {np.count_nonzero(full_mask)} -> {np.count_nonzero(improved_mask)} pixels")
    return improved_mask

def segment_object_by_point(bg_removal_wrap, img_rgb, point_xy, box_hint=None):
    """Segment object - KHÔI PHỤC HOÀN TOÀN: Remove background toàn ảnh → Mask toàn bộ"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    full_pil = Image.fromarray(img_rgb)
    full_rgba = bg_removal_wrap.remove_background(full_pil, bg_color=None)
    full_alpha = np.array(full_rgba)[:, :, 3]
    full_mask = (full_alpha > 128).astype(np.uint8)
    _log_info("BG Removal", f"Object segmentation: {np.count_nonzero(full_mask)} pixels total")
    return full_mask
