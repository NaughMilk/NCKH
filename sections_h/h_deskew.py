import cv2
import numpy as np
from typing import Tuple, Dict, Any
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning

def deskew_box_roi(roi_bgr: np.ndarray, mask: np.ndarray, method: str = "minAreaRect") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Deskew box ROI using various methods
    
    Args:
        roi_bgr: Input ROI image
        mask: Binary mask
        method: Deskew method ("minAreaRect", "PCA", "heuristic")
        
    Returns:
        Tuple of (deskewed_roi, deskewed_mask, deskew_info)
    """
    try:
        if method == "minAreaRect":
            return _deskew_min_area_rect(roi_bgr, mask)
        elif method == "PCA":
            return _deskew_pca(roi_bgr, mask)
        elif method == "heuristic":
            return _deskew_heuristic(roi_bgr, mask)
        else:
            _log_warning("Deskew", f"Unknown method: {method}, using minAreaRect")
            return _deskew_min_area_rect(roi_bgr, mask)
    except Exception as e:
        _log_warning("Deskew", f"Deskew failed: {e}, returning original")
        return roi_bgr, mask, {"angle": 0, "method": "failed"}

def _deskew_min_area_rect(roi_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew using minimum area rectangle"""
    try:
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi_bgr, mask, {"angle": 0, "method": "no_contours"}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle
        if angle < -45:
            angle += 90
        
        if abs(angle) < 1:  # Skip if angle is too small
            return roi_bgr, mask, {"angle": 0, "method": "min_angle"}
        
        # Rotate image
        h, w = roi_bgr.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed_roi = cv2.warpAffine(roi_bgr, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        deskewed_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        _log_success("Deskew", f"Applied rotation: {angle:.1f}°")
        return deskewed_roi, deskewed_mask, {"angle": angle, "method": "minAreaRect"}
        
    except Exception as e:
        _log_warning("Deskew", f"MinAreaRect deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "failed"}

def _deskew_pca(roi_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew using PCA analysis"""
    try:
        # Get mask points
        points = np.column_stack(np.where(mask > 0))
        if len(points) < 10:
            return roi_bgr, mask, {"angle": 0, "method": "insufficient_points"}
        
        # PCA analysis
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get principal axis
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi
        
        if abs(angle) < 1:
            return roi_bgr, mask, {"angle": 0, "method": "min_angle"}
        
        # Rotate image
        h, w = roi_bgr.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed_roi = cv2.warpAffine(roi_bgr, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        deskewed_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        _log_success("Deskew", f"Applied PCA rotation: {angle:.1f}°")
        return deskewed_roi, deskewed_mask, {"angle": angle, "method": "PCA"}
        
    except Exception as e:
        _log_warning("Deskew", f"PCA deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "failed"}

def _deskew_heuristic(roi_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew using heuristic edge analysis"""
    try:
        # Find edges
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return roi_bgr, mask, {"angle": 0, "method": "no_lines"}
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Get dominant angle
        angle_hist, angle_bins = np.histogram(angles, bins=36, range=(-90, 90))
        dominant_angle = angle_bins[np.argmax(angle_hist)]
        
        if abs(dominant_angle) < 1:
            return roi_bgr, mask, {"angle": 0, "method": "min_angle"}
        
        # Rotate image
        h, w = roi_bgr.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
        
        deskewed_roi = cv2.warpAffine(roi_bgr, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
        deskewed_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        _log_success("Deskew", f"Applied heuristic rotation: {dominant_angle:.1f}°")
        return deskewed_roi, deskewed_mask, {"angle": dominant_angle, "method": "heuristic"}
        
    except Exception as e:
        _log_warning("Deskew", f"Heuristic deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "failed"}