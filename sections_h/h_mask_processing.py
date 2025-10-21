# ========================= SECTION H: MASK PROCESSING ========================= #

import cv2
import numpy as np
from typing import Dict

# Import dependencies
from sections_a.a_config import Config, _log_info, _log_success, _log_warning
from sections_c.c_bg_removal import BGRemovalWrap

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
