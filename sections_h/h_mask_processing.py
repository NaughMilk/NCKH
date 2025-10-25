import cv2
import numpy as np
from typing import Tuple
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning

def _process_enhanced_mask(roi_mask: np.ndarray, cfg) -> np.ndarray:
    """Enhanced mask processing pipeline"""
    try:
        # Convert to binary if needed
        if roi_mask.max() > 1:
            roi_mask = (roi_mask > 127).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        
        return roi_mask
        
    except Exception as e:
        _log_warning("Mask Processing", f"Enhanced mask processing failed: {e}")
        return roi_mask

def _process_enhanced_mask_v2(roi_mask: np.ndarray, cfg) -> np.ndarray:
    """Enhanced mask processing pipeline V2"""
    try:
        # Convert to binary if needed
        if roi_mask.max() > 1:
            roi_mask = (roi_mask > 127).astype(np.uint8)
        
        # More aggressive morphological operations
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Remove small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask)
        if num_labels > 1:
            # Keep only largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            roi_mask = (labels == largest_label).astype(np.uint8) * 255
        
        # Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        return roi_mask
        
    except Exception as e:
        _log_warning("Mask Processing", f"Enhanced mask processing V2 failed: {e}")
        return roi_mask

def _force_rectangle_mask(mask: np.ndarray, expand_factor: float = 1.2) -> np.ndarray:
    """
    Force mask to be a perfect rectangle with intelligent expansion
    
    Args:
        mask: Input binary mask
        expand_factor: Factor to expand the rectangle (default: 1.2)
        
    Returns:
        Perfect rectangle mask
    """
    try:
        if not np.any(mask > 0):
            return mask
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate expansion
        center_x = x + w // 2
        center_y = y + h // 2
        new_w = int(w * expand_factor)
        new_h = int(h * expand_factor)
        
        # Ensure expansion doesn't exceed image bounds
        img_h, img_w = mask.shape[:2]
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        
        # Create perfect rectangle
        rectangle_mask = np.zeros_like(mask)
        rectangle_mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
        
        _log_success("Rectangle Mask", f"Created perfect rectangle: {new_w}x{new_h} at ({new_x},{new_y})")
        return rectangle_mask
        
    except Exception as e:
        _log_warning("Rectangle Mask", f"Failed to create perfect rectangle: {e}")
        return mask