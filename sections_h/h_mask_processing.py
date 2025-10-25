import cv2
import numpy as np
from typing import Tuple
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning

def _ensure_mask_format(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is in correct format for cv2.connectedComponentsWithStats()"""
    try:
        # Handle empty or invalid masks
        if mask is None or mask.size == 0:
            _log_warning("Mask Format", "Empty mask provided, returning default")
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # Ensure 2D shape
        if len(mask.shape) == 3:
            if mask.shape[2] == 1:
                mask = mask.squeeze(2)
            elif mask.shape[0] == 1:
                mask = mask.squeeze(0)
            elif mask.shape[1] == 1:
                mask = mask.squeeze(1)
            else:
                # Convert to grayscale by taking mean across channels
                mask = np.mean(mask, axis=2).astype(np.uint8)
        elif len(mask.shape) > 3:
            # Flatten to 2D if needed
            mask = mask.reshape(-1, mask.shape[-1])
            if mask.shape[1] > 1:
                mask = np.mean(mask, axis=1).astype(np.uint8)
            mask = mask.reshape(int(np.sqrt(mask.shape[0])), int(np.sqrt(mask.shape[0])))
        
        # Ensure minimum size
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            _log_warning("Mask Format", f"Mask too small: {mask.shape}, resizing")
            mask = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary mask
        mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    except Exception as e:
        _log_warning("Mask Format", f"Failed to format mask with shape {mask.shape}, dtype {mask.dtype}: {e}")
        # Return a safe default
        return np.zeros((100, 100), dtype=np.uint8)

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
        try:
            roi_mask = _ensure_mask_format(roi_mask)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask)
            if num_labels > 1:
                # Keep only largest component
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                roi_mask = (labels == largest_label).astype(np.uint8) * 255
        except Exception as e:
            _log_warning("Connected Components", f"Failed to process components: {e}")
            # Keep original mask if processing fails
        
        # Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        return roi_mask
        
    except Exception as e:
        _log_warning("Mask Processing", f"Enhanced mask processing V2 failed: {e}")
        return roi_mask

def _remove_background_noise(mask: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
    """
    Remove small background noise components
    
    Args:
        mask: Input binary mask
        min_area_ratio: Minimum area ratio for connected components
        
    Returns:
        Cleaned mask with noise removed
    """
    try:
        if not np.any(mask > 0):
            return mask
        
        # Calculate image area
        img_area = mask.shape[0] * mask.shape[1]
        min_area = img_area * min_area_ratio
        
        # Find connected components
        try:
            mask = _ensure_mask_format(mask)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Keep only large components
            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area:  # Only keep large components
                    cleaned_mask[labels == i] = 255
        except Exception as e:
            _log_warning("Background Noise Removal", f"Failed to process components: {e}")
            # Return original mask if processing fails
            cleaned_mask = mask
        
        _log_info("Background Noise Removal", f"Removed {num_labels-1} small components, kept {np.sum(cleaned_mask > 0)} pixels")
        return cleaned_mask
        
    except Exception as e:
        _log_warning("Background Noise Removal", f"Failed to remove background noise: {e}")
        return mask

def _apply_morphological_cleaning(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological operations to clean the mask
    
    Args:
        mask: Input binary mask
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned mask
    """
    try:
        if not np.any(mask > 0):
            return mask
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological operations
        # Close to fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Open to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Close again to ensure connectivity
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
        
    except Exception as e:
        _log_warning("Morphological Cleaning", f"Failed to apply morphological cleaning: {e}")
        return mask

def _process_enhanced_mask_v3(roi_mask: np.ndarray, cfg) -> np.ndarray:
    """Enhanced mask processing pipeline V3 with background noise removal"""
    try:
        # Convert to binary if needed
        if roi_mask.max() > 1:
            roi_mask = (roi_mask > 127).astype(np.uint8)
        
        # Step 1: Remove background noise
        roi_mask = _remove_background_noise(roi_mask, cfg.u2_min_area_ratio)
        
        # Step 2: Apply morphological cleaning
        roi_mask = _apply_morphological_cleaning(roi_mask, cfg.u2_morphology_kernel_size)
        
        # Step 3: Keep only largest component
        try:
            roi_mask = _ensure_mask_format(roi_mask)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
            if num_labels > 1:
                # Keep only largest component
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                roi_mask = (labels == largest_label).astype(np.uint8) * 255
        except Exception as e:
            _log_warning("Enhanced Mask Processing", f"Failed to process components: {e}")
            # Keep original mask if processing fails
        
        # Step 4: Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        _log_success("Mask Processing V3", f"Processed mask: {np.sum(roi_mask > 0)} pixels")
        return roi_mask
        
    except Exception as e:
        _log_warning("Mask Processing V3", f"Enhanced mask processing V3 failed: {e}")
        return roi_mask

def _force_rectangle_mask(mask: np.ndarray, expand_factor: float = 1.2, padding_px: int = 10) -> np.ndarray:
    """
    Force mask to be a perfect rectangle with intelligent expansion using minAreaRect
    Similar to white ring segment mechanism - căng đều 4 góc ra
    
    Args:
        mask: Input binary mask
        expand_factor: Factor to expand the rectangle (default: 1.2)
        padding_px: Additional padding in pixels
        
    Returns:
        Perfect rectangle mask with intelligent corner expansion
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
        
        # Use minAreaRect for intelligent rectangle fitting (like white ring segment)
        (cx, cy), (w, h), angle = cv2.minAreaRect(largest_contour)
        
        # Calculate expansion with padding
        new_w = w * expand_factor + padding_px
        new_h = h * expand_factor + padding_px
        
        # Create rectangle using boxPoints (like white ring segment)
        rect = ((cx, cy), (new_w, new_h), angle)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)
        
        # Create new mask with the rectangle
        h_mask, w_mask = mask.shape
        new_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
        
        # Fill the rectangle
        cv2.fillPoly(new_mask, [box_points], 255)
        
        _log_success("Rectangle Mask", f"Created intelligent rectangle with minAreaRect: {new_w:.1f}x{new_h:.1f} at angle {angle:.1f}°")
        return new_mask
        
    except Exception as e:
        _log_warning("Rectangle Mask", f"Failed to create intelligent rectangle: {e}")
        return mask