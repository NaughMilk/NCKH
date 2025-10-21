import cv2
import numpy as np
import time
from .a_geometry import robust_box_from_contour

def process_white_ring_segmentation(bgr, cfg):
    """Main white-ring segmentation function - Updated to use new config attributes"""
    start_time = time.time()
    
    # Import functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_preprocess import preprocess
        from sections_a.a_edges import EDGE
        from sections_a.a_geometry import (
            erode_inner, smooth_edges, smooth_mask, largest_contour, robust_box_from_contour, ring_mask_from_edges
        )
        from sections_a.a_config import _log_info, _log_warning
        
        # Preprocess
        gray = preprocess(bgr)
        
        # Edge detection using new config attributes
        edges = EDGE.detect(bgr, backend=cfg.edge_backend, canny_lo=cfg.canny_low, canny_hi=cfg.canny_high, dexi_thr=cfg.dexined_threshold)
        _log_info("White Ring", f"Edge detection result: {edges is not None}, shape: {edges.shape if edges is not None else 'None'}")
        
        if edges is None or np.count_nonzero(edges) == 0:
            _log_warning("White Ring", "No edges detected, returning empty mask")
            return np.zeros_like(gray), None, time.time() - start_time
        
        # Apply pair-edge filter if enabled
        if cfg.ring_pair_edge_filter:
            from sections_a.a_geometry import keep_paired_edges
            edges = keep_paired_edges(edges, cfg.pair_min_gap, cfg.pair_max_gap)
            _log_info("White Ring", f"Applied pair-edge filter: min_gap={cfg.pair_min_gap}, max_gap={cfg.pair_max_gap}")
        
        # Ring mask detection using new config attributes
        mask, contour = ring_mask_from_edges(
            edges, cfg.dilate_iters, cfg.close_kernel, cfg.ban_border_px,
            cfg.min_area_ratio, cfg.rect_score_min, cfg.aspect_ratio_min, cfg.aspect_ratio_max
        )
        
        # Erode inner using new config attribute
        mask = erode_inner(mask, cfg.erode_inner)
        
        # Apply smoothing using new config attributes
        mask = smooth_mask(mask, close=cfg.smooth_close, open_=cfg.smooth_open, use_hull=cfg.convex_hull)
        
        # FIXED: Re-enabled contour filtering to keep only the largest contour (the container)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Giữ lại contour có diện tích lớn nhất (là cái hộp)
            largest = max(contours, key=cv2.contourArea)
            # Tạo mask mới từ contour lớn nhất
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            _log_info("White Ring", f"Filtered to largest contour: {cv2.contourArea(largest):.0f} pixels")
        
        # Apply rectification based on mode (compatible with NCC_PIPELINE_NEW.py)
        rect_pts = None
        if cfg.force_rectify != "Off":
            # minAreaRect
            (cx, cy), (w, h), ang = cv2.minAreaRect(largest)
            if cfg.force_rectify == "Square":
                s = max(w, h) + 2*cfg.rect_pad
                rect = ((cx, cy), (s, s), ang)
            elif cfg.force_rectify == "Rectangle":   # rotated rectangle
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            elif cfg.force_rectify == "Robust (erode-fit-pad)":
                # Use robust box from contour
                poly_core = robust_box_from_contour(largest, trim=0.03)
                rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
                # Add padding
                (cx, cy), (w, h), ang = rect
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            else:
                # Default to Rectangle mode for any other value
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)

            rect_pts = cv2.boxPoints(rect).astype(np.int32)

            # thay mask bằng polygon ép (để phần fill bên trong cũng phẳng)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [rect_pts], 255)
            _log_info("White Ring", f"Applied {cfg.force_rectify} rectification with padding: {cfg.rect_pad}px")
        
        # Apply rectangle expansion factor
        if cfg.rectangle_expansion_factor != 1.0:
            kernel_size = int(cfg.rectangle_expansion_factor * 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if cfg.rectangle_expansion_factor > 1.0:
                mask = cv2.dilate(mask, kernel)
            else:
                mask = cv2.erode(mask, kernel)
            _log_info("White Ring", f"Applied rectangle expansion factor: {cfg.rectangle_expansion_factor}")
        
        # Filter components by minimum area
        if cfg.min_component_area > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_filtered = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) >= cfg.min_component_area:
                    cv2.fillPoly(mask_filtered, [contour], 255)
            mask = mask_filtered
            _log_info("White Ring", f"Filtered components by minimum area: {cfg.min_component_area}")
        
        # Show green frame if enabled
        if cfg.show_green_frame:
            # Add green border to mask for visualization
            mask_with_border = mask.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_with_border, contours, -1, (0, 255, 0), 3)
            mask = mask_with_border
        
        process_time = time.time() - start_time
        _log_info("White Ring", f"Segmentation completed in {process_time:.3f}s")
        
        return mask, contour, process_time
        
    except Exception as e:
        _log_warning("White Ring", f"Segmentation failed: {e}")
        return np.zeros_like(gray), None, time.time() - start_time

def ring_mask_from_edges(edges, dil_iters, close_k, ban_border_px,
                         min_area_ratio, rect_score_min, ar_min, ar_max):
    """Extract ring mask from edges using morphology and filtering"""
    h, w = edges.shape
    
    # Dilate edges
    if dil_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dil_iters)
    
    # Close operation to connect edges
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), None
    
    # Filter contours
    best = None
    best_score = -1
    
    for cnt in contours:
        # Check if contour touches border
        if ban_border_px > 0:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            if (x < ban_border_px or y < ban_border_px or 
                x + w_cnt > w - ban_border_px or y + h_cnt > h - ban_border_px):
                continue
        
        # Check area ratio
        area = cv2.contourArea(cnt)
        area_ratio = (area / (h * w)) * 100
        if area_ratio < min_area_ratio:
            continue
        
        # Check aspect ratio
        rect = cv2.minAreaRect(cnt)
        (_, _), (w_rect, h_rect), _ = rect
        if w_rect > 0 and h_rect > 0:
            aspect_ratio = min(w_rect, h_rect) / max(w_rect, h_rect)
            if aspect_ratio < ar_min or aspect_ratio > ar_max:
                continue
        
        # Check rectangle score
        rect_area = w_rect * h_rect
        if rect_area > 0:
            rect_score = area / rect_area
            if rect_score < rect_score_min:
                continue
        
        # Calculate final score
        score = area_ratio * rect_score
        if score > best_score:
            best_score = score
            best = cnt
    
    if best is None: 
        return np.zeros((h, w), dtype=np.uint8), None
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [best], 255)
    return mask, best

def overlay_white_ring(bgr, poly, inner_mask):
    """Overlay white ring visualization on image"""
    if poly is None or inner_mask is None:
        return bgr
    
    # Create overlay
    overlay = bgr.copy()
    
    # Draw polygon outline
    cv2.polylines(overlay, [poly], True, (0, 255, 0), 2)
    
    # Draw inner mask
    overlay[inner_mask > 0] = [0, 0, 255]  # Red for inner area
    
    # Blend with original
    result = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
    
    return result