import cv2
import numpy as np
import time

def process_white_ring_segmentation(bgr, cfg):
    """Main white-ring segmentation function - 100% copy from SEGMENT_GRADIO.py"""
    start_time = time.time()
    
    # Import functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_preprocess import preprocess
        from sections_a.a_edges import EDGE
        from sections.SECTION_A_CONFIG_UTILS import (
            erode_inner, smooth_edges, smooth_mask, largest_contour, robust_box_from_contour, ring_mask_from_edges
        )
        from sections_a.a_config import _log_info, _log_warning
        
        # Preprocess
        gray = preprocess(bgr)
        
        # Edge detection - DexiNed only
        edges = EDGE.detect(bgr, backend="DexiNed", canny_lo=cfg.canny_lo, canny_hi=cfg.canny_hi, dexi_thr=cfg.video_dexi_thr)
        
        # Ring mask detection
        mask, contour = ring_mask_from_edges(
            edges, cfg.dilate_px, cfg.close_px, cfg.ban_border_px,
            cfg.min_area_ratio, cfg.rect_score_min, cfg.ar_min, cfg.ar_max
        )
        
        # Erode inner
        mask = erode_inner(mask, cfg.erode_inner_px)
        
        # FIXED: Re-enabled contour filtering to keep only the largest contour (the container)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Giữ lại contour có diện tích lớn nhất (là cái hộp)
            largest = max(contours, key=cv2.contourArea)
            # Tạo mask mới từ contour lớn nhất
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            _log_info("White Ring", f"Filtered to largest contour: {cv2.contourArea(largest):.0f} pixels")
        
        # Apply edge smoothing based on mode
        if cfg.smooth_mode == "Off":
            pass  # No smoothing
        elif cfg.smooth_mode == "Light":
            mask = smooth_edges(mask, 1, 3)
        elif cfg.smooth_mode == "Medium":
            mask = smooth_edges(mask, cfg.smooth_iterations, cfg.gaussian_kernel)
        elif cfg.smooth_mode == "Strong":
            mask = smooth_edges(mask, cfg.smooth_iterations + 1, cfg.gaussian_kernel + 2)
        
        # Apply advanced post-processing - làm mịn trước khi ép
        if cfg.use_convex_hull:
            mask = smooth_mask(mask, close=15, open_=3, use_hull=True)
        
        # ----- Force rectify (always try) -----
        rect_pts = None
        base_cnt = None
        if cfg.force_rectify != "Off":
            base_cnt = largest_contour(mask)
            if base_cnt is None and contour is not None:
                base_cnt = contour  # dùng contour tốt nhất từ bước ring

            if base_cnt is not None:
                # minAreaRect
                (cx, cy), (w, h), ang = cv2.minAreaRect(base_cnt)
                if cfg.force_rectify == "Square":
                    s = max(w, h) + 2*cfg.rect_pad
                    rect = ((cx, cy), (s, s), ang)
                elif cfg.force_rectify == "Rectangle":   # rotated rectangle
                    rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
                elif cfg.force_rectify == "Robust (erode-fit-pad)":
                    # Use robust box from contour
                    poly_core = robust_box_from_contour(base_cnt, trim=0.03)
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
        
        # FIXED: Re-enabled mask size validation to prevent overly large masks
        mask_area = int(np.count_nonzero(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_ratio = mask_area / total_pixels if total_pixels > 0 else 0
        
        # If mask covers more than 80% of image, it's likely wrong
        if mask_ratio > 0.8:
            _log_warning("White Ring", f"Mask too large: {mask_area} pixels ({mask_ratio:.1%} of image)")
            # Create a smaller mask in the center
            h, w = mask.shape
            center_mask = np.zeros_like(mask)
            margin = min(h, w) // 4
            center_mask[margin:h-margin, margin:w-margin] = 255
            mask = center_mask
            _log_info("White Ring", f"Replaced with center mask: {np.count_nonzero(mask)} pixels")
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        return mask, rect_pts, process_time
        
    except ImportError:
        # Fallback if other functions not available yet
        return np.zeros_like(bgr[:,:,0], dtype=np.uint8), None, 0

def ring_mask_from_edges(edges, dil_iters, close_k, ban_border_px,
                         min_area_ratio, rect_score_min, ar_min, ar_max):
    """Find container mask from edges using white-ring detection"""
    h, w = edges.shape
    # nở + đóng để nối viền
    if dil_iters>0:
        edges = cv2.dilate(edges, None, iterations=int(dil_iters))
    if close_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_k), int(close_k)))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    # fill
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = -1
    img_area = h*w
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * (min_area_ratio/100.0): 
            continue
        x,y,wc,hc = cv2.boundingRect(c)
        # cấm bám sát biên
        if x<=ban_border_px or y<=ban_border_px or \
           x+wc>=w-ban_border_px or y+hc>=h-ban_border_px:
            pass  # cho phép chạm nhẹ biên, không loại sớm
        # chấm điểm chữ nhật + độ đặc
        rect = cv2.minAreaRect(c)
        (cx,cy),(rw,rh),_ = rect
        rect_area = max(rw*rh, 1)
        rect_score = float(area)/rect_area            # ~1 nếu gần hình chữ nhật
        hull = cv2.convexHull(c)
        solidity = float(area)/max(cv2.contourArea(hull),1)
        ar = max(rw, rh)/max(1.0, min(rw, rh))
        if rect_score < rect_score_min or ar<ar_min or ar>ar_max:
            continue
        score = 0.6*rect_score + 0.4*solidity + 0.000001*area
        if score > best_score:
            best = c; best_score = score

    if best is None:
        return np.zeros_like(edges, np.uint8), None

    mask = np.zeros_like(edges, np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=-1)
    return mask, best

def overlay_white_ring(bgr, poly, inner_mask):
    """
    Create overlay visualization with white ring and filled inner area
    Args:
        bgr: BGR image
        poly: Polygon contour
        inner_mask: Binary mask of inner area
    Returns:
        vis: Visualization image
    """
    vis = bgr.copy()
    if poly is not None:
        cv2.polylines(vis, [poly], isClosed=True, color=(255, 255, 255), thickness=3)
    # Tô mờ phần trong hộp
    tint = np.zeros_like(vis)
    tint[:] = (255, 255, 255)
    alpha = 0.25
    vis = np.where(inner_mask[..., None] > 0, (alpha*tint + (1-alpha)*vis).astype(vis.dtype), vis)
    return vis
