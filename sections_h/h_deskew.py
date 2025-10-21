# ========================= SECTION H: DESKEW ========================= #

import cv2
import numpy as np
from typing import Tuple, Dict

# Import dependencies
from sections_a.a_config import _log_info, _log_warning

def deskew_box_roi(roi_bgr: np.ndarray, mask: np.ndarray, method: str = "minAreaRect") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew box ROI to align to 90-degree angles"""
    try:
        H, W = roi_bgr.shape[:2]
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi_bgr, mask, {"angle": 0, "method": "no_contour"}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if method == "minAreaRect":
            # Use minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Normalize angle to 0-90 degrees
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
        elif method == "PCA":
            # Use PCA to find principal axis
            data_pts = largest_contour.reshape(-1, 2).astype(np.float32)
            mean = np.mean(data_pts, axis=0)
            centered = data_pts - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
        else:  # heuristic
            # Use bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            angle = 0  # Assume already aligned
        
        # Snap to nearest 90-degree angle
        if abs(angle) < 15:
            angle = 0
        elif abs(angle - 90) < 15:
            angle = 90
        elif abs(angle + 90) < 15:
            angle = -90
        elif abs(angle - 180) < 15 or abs(angle + 180) < 15:
            angle = 0
        
        # Apply rotation if needed
        if abs(angle) > 1:  # Only rotate if significant angle
            center = (W // 2, H // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate ROI and mask
            roi_rotated = cv2.warpAffine(roi_bgr, rotation_matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask_rotated = cv2.warpAffine(mask, rotation_matrix, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            _log_info("Deskew", f"Rotated by {angle:.1f} degrees using {method}")
            return roi_rotated, mask_rotated, {"angle": angle, "method": method, "center": center}
        
        return roi_bgr, mask, {"angle": 0, "method": method}
        
    except Exception as e:
        _log_warning("Deskew", f"Deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "error", "error": str(e)}
