import cv2
import numpy as np
import os
import glob

def cleanup_dataset_files(dataset_path="sdy_project/dataset_sdy_box"):
    """Xóa images/masks không có metadata tương ứng"""
    
    meta_dir = os.path.join(dataset_path, "meta")
    img_dir = os.path.join(dataset_path, "images", "train")
    mask_dir = os.path.join(dataset_path, "masks", "train")
    
    if not os.path.exists(img_dir):
        print(f"❌ Không tìm thấy {img_dir}")
        return {"status": "error", "message": f"Không tìm thấy {img_dir}"}
    
    # Lấy danh sách metadata files
    meta_files = set()
    if os.path.exists(meta_dir):
        meta_files = set(os.path.splitext(f)[0] for f in os.listdir(meta_dir) if f.endswith('.json'))
    
    # Lấy danh sách image files
    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    deleted_count = 0
    kept_count = 0
    
    # Xóa images không có metadata
    for img_file in image_files:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        if base_name not in meta_files:
            os.remove(img_file)
            deleted_count += 1
            
            mask_file = os.path.join(mask_dir, f"{base_name}.png")
            if os.path.exists(mask_file):
                os.remove(mask_file)
            
            print(f"🗑️  Xóa: {base_name}")
        else:
            kept_count += 1
    
    if deleted_count == 0:
        print(f"✅ Dataset đã sạch: {kept_count} files khớp hoàn toàn")
        return {"status": "clean", "kept": kept_count, "deleted": 0}
    else:
        print(f"✅ Cleanup hoàn thành: Giữ {kept_count}, Xóa {deleted_count}")
        return {"status": "cleaned", "kept": kept_count, "deleted": deleted_count}

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
