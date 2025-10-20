import cv2
import numpy as np

def mask_to_polygon_norm(mask: np.ndarray, img_w: int, img_h: int, max_points: int = 200):
    """Convert mask to normalized polygon"""
    mask_u8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], (0,0,0,0)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, num=max_points, dtype=int)
        pts = pts[idx]
    x1, y1, w, h = cv2.boundingRect(pts.astype(np.int32))
    x_c = (x1 + w/2.0) / img_w
    y_c = (y1 + h/2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    poly = []
    for (x, y) in pts:
        poly.extend([float(x)/img_w, float(y)/img_h])
    return poly, (x_c, y_c, w_n, h_n)
