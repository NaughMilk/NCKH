import cv2
import numpy as np

def robust_avg_box(samples, trim_ratio=0.1):
    """
    samples: list of (long_side, short_side)
    Dùng median hoặc trimmed mean để chống outlier.
    Trả về (long_avg, short_avg, n_used).
    """
    if not samples:
        return None, None, 0
    
    longs = sorted([s[0] for s in samples])
    shorts = sorted([s[1] for s in samples])
    k = int(len(longs) * trim_ratio)
    
    def tmean(a):
        if len(a) >= 2*k+1: 
            a = a[k:-k]
        return float(np.median(a)) if len(a) > 0 else None
    
    return tmean(longs), tmean(shorts), len(longs)

def apply_locked_box(cx, cy, w_obs, h_obs, ang_deg, long_locked, short_locked, pad_px=0):
    """
    Dựa trên hướng quan sát (w_obs >= h_obs hay ngược lại) để gán đúng chiều
    cho (long_locked, short_locked). Trả về poly 4 đỉnh.
    """
    if w_obs >= h_obs:
        sz = (long_locked + 2*pad_px, short_locked + 2*pad_px)
    else:
        sz = (short_locked + 2*pad_px, long_locked + 2*pad_px)
    
    rect = ((cx, cy), sz, ang_deg)
    poly = cv2.boxPoints(rect).astype(np.int32)
    return poly

def robust_box_from_contour(cnt, trim=0.03):
    """Fit minAreaRect để lấy góc, rồi loại bỏ outlier theo trục quay"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    pts = cnt.reshape(-1, 2).astype(np.float32)

    th = np.deg2rad(ang)
    R = np.array([[np.cos(th),  np.sin(th)],
                  [-np.sin(th), np.cos(th)]], np.float32)
    pts_r = (pts - [cx, cy]) @ R.T

    k = max(1, int(len(pts_r) * trim))
    xs = np.sort(pts_r[:, 0]); ys = np.sort(pts_r[:, 1])
    x1, x2 = xs[k], xs[-k-1]
    y1, y2 = ys[k], ys[-k-1]

    box_r = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    box = (box_r @ R) + [cx, cy]
    return box.astype(np.int32)

def minarearect_on_eroded(mask, erode_px=3, pad=12, trim=0.03):
    """Erode để cắt bóng, fit hộp chắc, rồi nới ra pad"""
    # Import largest_contour from other modules (will be available after all sections are loaded)
    try:
        from sections.SECTION_A_CONFIG_UTILS import largest_contour
    except ImportError:
        # Fallback implementation
        def largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max(contours, key=cv2.contourArea) if contours else None
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1,)*2)
    m_small = cv2.erode(mask, k)
    cnt = largest_contour(m_small)
    if cnt is None: 
        return None, mask
    poly_core = robust_box_from_contour(cnt, trim=trim)

    rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
    (cx,cy),(w,h),ang = rect
    rect = ((cx,cy),(w+2*pad, h+2*pad), ang)
    poly = cv2.boxPoints(rect).astype(np.int32)
    out = np.zeros_like(mask); cv2.fillPoly(out,[poly],255)
    return poly, out

def keep_paired_edges(edge, min_gap=4, max_gap=18):
    """Giữ những biên có đối biên cách trong [min_gap, max_gap]"""
    e = (edge>0).astype(np.uint8)*255
    inv = 255 - e
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    paired = ((dist>=min_gap) & (dist<=max_gap)).astype(np.uint8)*255
    paired = cv2.dilate(paired, None, iterations=1)  # khôi phục nét
    
    # Nếu quá ít edges được giữ lại, fallback về edges gốc
    result = cv2.bitwise_and(e, paired)
    if np.count_nonzero(result) < np.count_nonzero(e) * 0.1:  # Nếu mất >90% edges
        print(f"[WARNING] Pair-edge filter quá strict, fallback về edges gốc")
        return e
    
    return result

def fit_rect_core(rgb, backend, canny_lo, canny_hi, dexi_thr,
                  dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                  ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                  use_hull, use_pair_filter, pair_min_gap, pair_max_gap):
    """
    Trả về: (cx, cy, w, h, angle_deg, mask, poly_fit)
    - poly_fit là polygon theo 'Robust (erode-fit-pad)' trước khi nới pad.
    - w,h là kích thước của minAreaRect trên mask đã erode-fit (không pad).
    - angle theo chuẩn OpenCV minAreaRect.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Import functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_edges import EDGE
        from sections.SECTION_A_CONFIG_UTILS import (
            ring_mask_from_edges, smooth_mask, largest_contour
        )
        
        # Use EDGE.detect() for advanced edge detection (DexiNed or Canny)
        edges = EDGE.detect(bgr, backend, canny_lo, canny_hi, dexi_thr)
        
        # Apply pair-edge filter if enabled
        if use_pair_filter:
            edges = keep_paired_edges(edges, pair_min_gap, pair_max_gap)
        
        # Get mask from edges using existing function
        mask, best = ring_mask_from_edges(edges, dilate_iters, close_kernel, 8,  # ban_border_px
                                          min_area_ratio, rect_score_min, ar_min, ar_max)
        
        # Erode if needed
        if erode_inner > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_inner*2+1), int(erode_inner*2+1)))
            mask = cv2.erode(mask, k)
        
        # Smooth mask using existing function
        mask = smooth_mask(mask, close=smooth_close, open_=smooth_open, use_hull=use_hull)
        
        # Get largest contour
        c = largest_contour(mask)
        if c is None:
            return None, None, None, None, None, mask, None
        
        # Apply robust box fitting (erode-fit without pad)
        poly_core = robust_box_from_contour(c, trim=0.03)
        rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
        (cx, cy), (w, h), ang = rect
        
        return cx, cy, w, h, ang, mask, poly_core
        
    except ImportError:
        # Fallback if other functions not available yet
        return None, None, None, None, None, None, None
