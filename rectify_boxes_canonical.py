#!/usr/bin/env python3
"""
rectify_boxes_canonical.py
--------------------------
- Crop/warp theo segment_corners (không resize ép kích thước).
- Đặt cạnh dài nằm ngang bằng minAreaRect (không scale chuẩn hoá).
- Chuẩn hoá hướng bằng *xoay 0/90/180/270* sao cho tâm QR **gần góc phải-dưới** nhất.
  (Không lật ngang/lật dọc, không resize.)

Cài đặt:
pip install opencv-python numpy tqdm

Chạy:
python rectify_boxes_canonical.py --root /path/to/dataset --out /path/to/output --save-meta
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_json_pairs(root: Path):
    images: Dict[str, Path] = {}
    metas: Dict[str, Path] = {}
    
    # Chỉ tìm ảnh trong folder images/, không phải masks/
    for p in root.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMG_EXTS and "images" in str(p):
                images[p.stem] = p
            elif ext == ".json" and "meta" in str(p):
                metas[p.stem] = p
    
    pairs = []
    for stem, ip in images.items():
        jp = metas.get(stem)
        if jp is not None:
            pairs.append((ip, jp))
    return sorted(pairs, key=lambda x: x[0].as_posix())


def _flatten_points(points_raw: List) -> np.ndarray:
    pts = []
    for item in points_raw:
        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple, np.ndarray)):
            x, y = item[0]
        else:
            x, y = item
        pts.append([float(x), float(y)])
    return np.asarray(pts, dtype=np.float32)


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    assert pts.shape == (4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[:, 0]
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect


def corners_of_min_area_rect(poly_pts: np.ndarray) -> np.ndarray:
    contour = poly_pts.reshape(-1, 1, 2).astype(np.float32)
    rect = cv2.minAreaRect(contour)   # ((cx,cy), (w,h), angle)
    box = cv2.boxPoints(rect)         # 4x2
    box = order_points_clockwise(box)

    # đặt cạnh dài nằm ngang bằng đổi thứ tự 4 điểm (không đổi scale)
    w = np.linalg.norm(box[1] - box[0])
    h = np.linalg.norm(box[3] - box[0])
    if h > w:
        box = np.array([box[3], box[0], box[1], box[2]], dtype=np.float32)
    return box.astype(np.float32)


def warp_rect_patch(image: np.ndarray, rect4: np.ndarray):
    tl, tr, br, bl = rect4
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    W = max(1, int(round(max(widthA, widthB))))
    H = max(1, int(round(max(heightA, heightB))))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect4.astype(np.float32), dst)
    patch = cv2.warpPerspective(image, M, (W, H), flags=cv2.INTER_LINEAR)
    return patch, M


def perspective_transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_ = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts_, M)
    return out.reshape(-1, 2)


# ---------- rotation helpers (no resize) ----------
def rotate_points_90cw(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    # (x,y) -> (H-1 - y, x); new size: (H, W)
    pts = points.copy().astype(np.float32)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    pts[:, 0] = (H - 1) - y
    pts[:, 1] = x
    return pts, H, W


def rotate_points_90ccw(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    # (x,y) -> (y, W-1 - x); new size: (H, W) -> (H', W') = (W, H)
    pts = points.copy().astype(np.float32)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    pts[:, 0] = y
    pts[:, 1] = (W - 1) - x
    return pts, H, W


def rotate_points_180(points: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, int, int]:
    # (x,y) -> (W-1 - x, H-1 - y); size unchanged
    pts = points.copy().astype(np.float32)
    pts[:, 0] = (W - 1) - pts[:, 0]
    pts[:, 1] = (H - 1) - pts[:, 1]
    return pts, W, H


def choose_best_rotation(img: np.ndarray, qr_center: np.ndarray):
    """
    Chọn xoay 0/90cw/180/270cw sao cho tâm QR gần góc phải-dưới nhất.
    Trả về (rotated_img, rot_code, qr_center_rotated).
    rot_code in {"rot0", "rot90cw", "rot180", "rot270cw"}.
    """
    H, W = img.shape[:2]
    candidates = []

    # 0 deg
    br0 = np.array([W - 1, H - 1], dtype=np.float32)
    d0 = float(np.linalg.norm(qr_center - br0))
    candidates.append(("rot0", d0))

    # 90 cw
    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    qr90, H90, W90 = rotate_points_90cw(qr_center.reshape(1, 2), W, H)
    br90 = np.array([W90 - 1, H90 - 1], dtype=np.float32)
    d90 = float(np.linalg.norm(qr90.reshape(2) - br90))
    candidates.append(("rot90cw", d90))

    # 180
    img180 = cv2.rotate(img, cv2.ROTATE_180)
    qr180, W180, H180 = rotate_points_180(qr_center.reshape(1, 2), W, H)
    br180 = np.array([W180 - 1, H180 - 1], dtype=np.float32)
    d180 = float(np.linalg.norm(qr180.reshape(2) - br180))
    candidates.append(("rot180", d180))

    # 270 cw (i.e., 90 ccw)
    img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    qr270, H270, W270 = rotate_points_90ccw(qr_center.reshape(1, 2), W, H)
    br270 = np.array([W270 - 1, H270 - 1], dtype=np.float32)
    d270 = float(np.linalg.norm(qr270.reshape(2) - br270))
    candidates.append(("rot270cw", d270))

    # pick best
    best = min(candidates, key=lambda x: x[1])[0]
    if best == "rot0":
        return img, "rot0", qr_center
    elif best == "rot90cw":
        return img90, "rot90cw", qr90.reshape(2)
    elif best == "rot180":
        return img180, "rot180", qr180.reshape(2)
    else:
        return img270, "rot270cw", qr270.reshape(2)


def process_one(image_path: Path, json_path: Path, out_stage1: Path, out_final: Path,
                save_meta: bool = False):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seg_raw = meta.get("segment_corners", [])
    if not seg_raw:
        print(f"[WARN] No segment_corners in {json_path.name}, skipping.")
        return None

    seg_pts = _flatten_points(seg_raw)
    rect4 = corners_of_min_area_rect(seg_pts)
    patch, Hmat = warp_rect_patch(img, rect4)
    Hh, Hw = patch.shape[:2]

    out_stage1.mkdir(parents=True, exist_ok=True)
    stage1_path = out_stage1 / (image_path.stem + ".png")
    cv2.imwrite(str(stage1_path), patch)

    # QR center in rectified space
    qr_raw = meta.get("qr_corners", [])
    if len(qr_raw) >= 4:
        qr_pts = _flatten_points(qr_raw)[:4]
        qr_rect_pts = perspective_transform_points(Hmat, qr_pts)
        qr_center_rect = qr_rect_pts.mean(axis=0)
        # choose best rotation (0/90/180/270) to bring QR closest to BR
        final_img, rot_code, qr_center_final = choose_best_rotation(patch, qr_center_rect.astype(np.float32))
    else:
        # No QR info -> keep patch
        final_img, rot_code = patch, "rot0"
        qr_rect_pts = None
        qr_center_final = None

    out_final.mkdir(parents=True, exist_ok=True)
    final_path = out_final / (image_path.stem + ".png")
    cv2.imwrite(str(final_path), final_img)

    if save_meta:
        summary = {
            "source_image": str(image_path),
            "source_json": str(json_path),
            "stage1_image": str(stage1_path),
            "final_image": str(final_path),
            "rect4_src_ordered_tl_tr_br_bl": rect4.tolist(),
            "rectified_size_wh": [int(Hw), int(Hh)],
            "canonical_rotation": rot_code,
        }
        if qr_rect_pts is not None:
            summary["qr_rectified"] = qr_rect_pts.tolist()
            summary["qr_center_rectified"] = qr_rect_pts.mean(axis=0).tolist()
            summary["qr_center_final"] = qr_center_final.tolist()
        final_json = out_final / (image_path.stem + ".json")
        with open(final_json, "w", encoding="utf-8") as fw:
            json.dump(summary, fw, ensure_ascii=False, indent=2)

    return True


def main():
    ap = argparse.ArgumentParser(description="Crop & rectify boxes, rotate 0/90/180/270 so QR is at bottom-right.")
    ap.add_argument("--root", type=str, required=True, help="Root folder containing images + JSON metas (recursive).")
    ap.add_argument("--out", type=str, required=True, help="Output root folder.")
    ap.add_argument("--save-meta", action="store_true", help="Save transformed JSON metadata for each final image.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_stage1 = out_root / "stage1_rectified"
    out_final = out_root / "final_canonical"

    pairs = find_image_json_pairs(root)
    if not pairs:
        print("[ERROR] No (image,json) pairs found under:", root)
        return

    print(f"[INFO] Found {len(pairs)} pairs. Processing...")
    for img_path, json_path in tqdm(pairs, desc="Rectifying"):
        try:
            process_one(img_path, json_path, out_stage1, out_final, save_meta=args.save_meta)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("[DONE] All pairs processed.")
    print(f"  Stage1 images -> {out_stage1}")
    print(f"  Final images  -> {out_final}")


if __name__ == "__main__":
    main()
