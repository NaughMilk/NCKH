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

def erode_inner(mask, px):
    """Erode mask inward"""
    if px<=0: return mask
    px = int(px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.erode(mask, k)

def largest_contour(mask):
    """Tìm contour lớn nhất trong mask"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def smooth_mask(mask, close=15, open_=5, use_hull=False):
    """Làm mịn mask với morphology + convex hull"""
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_, open_))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)

    if use_hull:
        cnt = largest_contour(m)
        if cnt is not None:
            hull = cv2.convexHull(cnt)
            m = np.zeros_like(mask)
            cv2.fillPoly(m, [hull], 255)
    return m

def smooth_edges(mask, smooth_iterations, gaussian_kernel):
    """Làm mịn viền mask với thuật toán cải tiến"""
    if smooth_iterations <= 0 or gaussian_kernel <= 0:
        return mask
    
    # Convert to integers
    smooth_iterations = int(smooth_iterations)
    gaussian_kernel = int(gaussian_kernel)
    
    # Đảm bảo kernel size là số lẻ
    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1
    
    # Bước 1: Morphological opening để loại bỏ noise nhỏ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Bước 2: Morphological closing để lấp đầy lỗ hổng nhỏ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Bước 3: Gaussian blur để làm mịn viền
    mask_smooth = cv2.GaussianBlur(mask_clean, (gaussian_kernel, gaussian_kernel), 0)
    
    # Bước 4: Threshold lại để có mask binary
    _, mask_final = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
    
    return mask_final

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
        hull_area = cv2.contourArea(hull)
        hull_score = float(area)/max(hull_area, 1)    # ~1 nếu gần convex
        ar = max(rw,rh)/max(min(rw,rh),1)
        if ar<ar_min or ar>ar_max: continue
        score = rect_score * hull_score
        if score > rect_score_min and score > best_score:
            best_score = score; best = c
    
    if best is None: 
        return np.zeros((h,w), dtype=np.uint8), None
    
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillPoly(mask, [best], 255)
    return mask, best

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
    
    # Import EDGE from a_edges
    try:
        from sections_a.a_edges import EDGE
    except ImportError:
        # Fallback
        class EDGE:
            def detect(self, bgr, backend, canny_lo, canny_hi, dexi_thr): return None
    
    edges = EDGE.detect(bgr, backend, canny_lo, canny_hi, dexi_thr)
    
    # Apply pair-edge filter if enabled
    if use_pair_filter:
        edges = keep_paired_edges(edges, pair_min_gap, pair_max_gap)
    
    # Get mask from edges
    mask, best = ring_mask_from_edges(edges, dilate_iters, close_kernel,
                                      min_area_ratio, rect_score_min, ar_min, ar_max)
    
    # Erode if needed
    if erode_inner > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_inner*2+1), int(erode_inner*2+1)))
        mask = cv2.erode(mask, k)
    
    # Smooth mask
    mask = smooth_mask(mask, close=smooth_close, open_=smooth_open, hull=use_hull)
    
    # Get largest contour
    c = largest_contour(mask)
    if c is None:
        return None, None, None, None, None, mask, None
    
    # Apply robust box fitting (erode-fit without pad)
    poly_core = robust_box_from_contour(c, trim=0.03)
    rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
    (cx, cy), (w, h), ang = rect
    
    return cx, cy, w, h, ang, mask, poly_core
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
