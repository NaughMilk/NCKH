import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple

def _apply_feather_alpha(pil_rgba: Image.Image, feather_px: int = 0) -> Image.Image:
    """Feather (blur) the alpha channel to soften edges."""
    if feather_px <= 0:
        return pil_rgba
    arr = np.array(pil_rgba)
    if arr.shape[2] != 4:
        return pil_rgba
    alpha = arr[:, :, 3]
    k = max(1, int(2 * round(feather_px) + 1))
    alpha_blur = cv2.GaussianBlur(alpha, (k, k), 0)
    arr[:, :, 3] = alpha_blur
    return Image.fromarray(arr, mode="RGBA")

def _composite_background(pil_rgba: Image.Image, bg_color: Optional[Tuple[int, int, int]]) -> Image.Image:
    """Composite RGBA onto a solid background color; if bg_color is None, return RGBA (transparent)."""
    if bg_color is None:
        return pil_rgba
    bg = Image.new("RGB", pil_rgba.size, bg_color)
    return Image.alpha_composite(bg.convert("RGBA"), pil_rgba).convert("RGB")

def _post_process_mask_light(mask):
    """Post-process mask nhẹ để tránh làm mất thông tin"""
    if np.count_nonzero(mask) == 0:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_clean

def _improve_mask_with_rectangle_fitting(mask):
    """Cải thiện mask bằng cách detect 4 góc và fit rectangle"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    if np.count_nonzero(mask) == 0:
        return mask
    
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            improved_mask = np.zeros_like(mask)
            cv2.fillPoly(improved_mask, [box], 1)
            
            original_area = np.count_nonzero(mask)
            improved_area = np.count_nonzero(improved_mask)
            
            if 0.8 * original_area <= improved_area <= 1.5 * original_area:
                _log_info("BG Removal", f"Rectangle fitting applied: {original_area} -> {improved_area} pixels")
                return improved_mask
        
        return _post_process_mask_light(mask)
        
    except Exception as e:
        _log_warning("BG Removal", f"Rectangle fitting failed: {e}")
        return _post_process_mask_light(mask)

def _keep_only_largest_component(mask):
    """Chỉ giữ vùng segment lớn nhất, bỏ hết vùng nhỏ (noise)"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    if np.count_nonzero(mask) == 0:
        return mask
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    # Tìm vùng lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask mới chỉ với vùng này
    new_mask = np.zeros_like(mask)
    cv2.fillPoly(new_mask, [largest_contour], 255)
    
    _log_info("BG Removal", f"Kept largest component: {cv2.contourArea(largest_contour)} pixels")
    return new_mask

def _enhanced_post_process_mask(mask, smooth_edges=True, remove_noise=True):
    """Enhanced post-processing for better mask quality"""
    if np.count_nonzero(mask) == 0:
        return mask
    
    # 1. Remove noise with morphological operations
    if remove_noise:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)  # Loại bỏ noise nhỏ
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)  # Lấp đầy lỗ hổng nhỏ
    
    # 2. Smooth edges with Gaussian blur + threshold
    if smooth_edges:
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (5, 5), 1.0)  # Blur nhẹ
        mask = (mask_blurred > 0.5).astype(np.uint8) * 255        # Threshold lại
    
    # 3. Fill holes
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
    
    return mask

def _expand_corners(mask, expand_pixels=3):
    """Mở rộng 4 góc của mask để ăn hết cái hộp"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    if np.count_nonzero(mask) == 0:
        return mask
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tạo bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Mở rộng 4 góc
    expanded_mask = mask.copy()
    
    # Mở rộng từng góc
    corners = [
        (x, y),  # Top-left
        (x + w - expand_pixels, y),  # Top-right
        (x, y + h - expand_pixels),  # Bottom-left
        (x + w - expand_pixels, y + h - expand_pixels)  # Bottom-right
    ]
    
    for corner_x, corner_y in corners:
        # Tạo vùng mở rộng cho mỗi góc
        cv2.rectangle(expanded_mask, 
                     (corner_x, corner_y), 
                     (corner_x + expand_pixels, corner_y + expand_pixels), 
                     255, -1)
    
    _log_info("BG Removal", f"Expanded corners by {expand_pixels} pixels")
    return expanded_mask

def _enhanced_post_process_mask_v2(mask, smooth_edges=True, remove_noise=True):
    """Enhanced post-processing V2 - mạnh hơn"""
    if np.count_nonzero(mask) == 0:
        return mask
    
    # 1. Remove noise với kernel lớn hơn
    if remove_noise:
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Tăng từ 3x3 lên 5x5
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium, iterations=3)  # Tăng iterations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    
    # 2. Smooth edges mạnh hơn
    if smooth_edges:
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (7, 7), 2.0)  # Tăng kernel và sigma
        mask = (mask_blurred > 0.6).astype(np.uint8) * 255        # Tăng threshold từ 0.5 lên 0.6
    
    # 3. Fill holes với kernel lớn hơn
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Tăng từ 5x5 lên 7x7
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)  # Tăng iterations
    
    return mask

def _apply_median_filter(mask, kernel_size=5):
    """Áp dụng median filter để làm mượt mask"""
    if np.count_nonzero(mask) == 0:
        return mask
    
    # Median filter để loại bỏ noise
    mask_filtered = cv2.medianBlur(mask, kernel_size)
    return mask_filtered

def _apply_bilateral_filter(mask, d=9, sigma_color=75, sigma_space=75):
    """Áp dụng bilateral filter để làm mượt nhưng giữ edges"""
    if np.count_nonzero(mask) == 0:
        return mask
    
    # Bilateral filter
    mask_float = mask.astype(np.float32) / 255.0
    mask_filtered = cv2.bilateralFilter(mask_float, d, sigma_color, sigma_space)
    mask_filtered = (mask_filtered > 0.5).astype(np.uint8) * 255
    
    return mask_filtered
