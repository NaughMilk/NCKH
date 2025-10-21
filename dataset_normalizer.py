#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_normalizer_strict.py
--------------------------------
Strict square-ROI normalizer with *no implicit resize*.

What it does:
1) Read JSON fields:
   - segment_square_corners: required for strict mode (4 pts, any order)
   - segment_corners: optional (polygon); used only if you set --mask_mode=polygon
   - qr_corners: optional (4 pts); used to enforce orientation

2) Mask step (default = square):
   - --mask_mode square  => mask outside the *square* ROI (segment_square_corners)
   - --mask_mode polygon => mask outside the original polygon (segment_corners)
   Masking happens BEFORE warp so background is guaranteed black.

3) Warp step:
   - Perspective warp from the 4-pt square to an axis-aligned rectangle.
   - --force_square true => force output to perfect square with side=max(w,h)
   - --force_square false => keep (w,h) from the box geometry
   *No final resize* unless you set --final_size > 0

4) Orientation step:
   - Detect which image corner (TL/TR/BR/BL) is closest to the *centroid* of QR (after warp)
   - With --rot_mode only180 (default): if QR at TR -> rotate 180°, else keep
   - With --rot_mode any90: rotate by 0/90/180/270 so QR ends at BL

Usage (single file):
  python dataset_normalizer_strict.py single \
    --image /path/img.jpg \
    --json  /path/img.json \
    --out   /path/out.jpg \
    --mask_mode square --force_square true --rot_mode only180 --final_size 0

Usage (dataset root: images/{train,val} + meta/*.json):
  python dataset_normalizer_strict.py dataset \
    --root /path/dataset \
    --out_root /path/out \
    --mask_mode square --force_square true --rot_mode only180 --final_size 0
"""
from __future__ import annotations
import os, json, cv2, numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
import argparse, math

# ---------- helpers ----------
def _to_points(arr):
    pts = np.array(arr, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts[:,0,:]
    return pts

def order_box_pts(box4):
    """
    Order 4 corner points to TL, TR, BR, BL
    Use simple min/max approach for stability
    """
    pts = np.array(box4, dtype=np.float32)
    
    # Find TL (min Y, min X among min Y)
    y_coords = pts[:, 1]
    min_y = np.min(y_coords)
    min_y_indices = np.where(y_coords == min_y)[0]
    if len(min_y_indices) == 1:
        tl_idx = min_y_indices[0]
    else:
        # If multiple points have min Y, choose the one with min X
        x_coords_min_y = pts[min_y_indices, 0]
        tl_idx = min_y_indices[np.argmin(x_coords_min_y)]
    
    # Find BR (max Y, max X among max Y)
    max_y = np.max(y_coords)
    max_y_indices = np.where(y_coords == max_y)[0]
    if len(max_y_indices) == 1:
        br_idx = max_y_indices[0]
    else:
        # If multiple points have max Y, choose the one with max X
        x_coords_max_y = pts[max_y_indices, 0]
        br_idx = max_y_indices[np.argmax(x_coords_max_y)]
    
    # Find TR and BL from remaining points
    remaining_indices = [i for i in range(4) if i != tl_idx and i != br_idx]
    
    if len(remaining_indices) == 2:
        # TR should be top-right (smaller Y, larger X)
        # BL should be bottom-left (larger Y, smaller X)
        p1, p2 = remaining_indices[0], remaining_indices[1]
        
        # Check which is more "top-right" vs "bottom-left"
        if pts[p1, 1] < pts[p2, 1]:  # p1 is higher (more top)
            if pts[p1, 0] > pts[p2, 0]:  # p1 is more right
                tr_idx, bl_idx = p1, p2
            else:
                tr_idx, bl_idx = p2, p1
        else:  # p2 is higher
            if pts[p2, 0] > pts[p1, 0]:  # p2 is more right
                tr_idx, bl_idx = p2, p1
            else:
                tr_idx, bl_idx = p1, p2
    
    return np.array([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]], np.float32)

def mask_poly(image, poly):
    h, w = image.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    else:
        mask[:] = 255
    return cv2.bitwise_and(image, image, mask=mask)

def persp_from_box(image, box4, force_square=True, force_horizontal=True):
    """
    Warp image using perspective transform
    
    Args:
        image: Input image
        box4: 4 corner points (will be ordered to TL,TR,BR,BL)
        force_square: Force output to be square
        force_horizontal: For rectangle mode, ensure width > height (horizontal layout)
    
    Returns:
        (warped_image, perspective_matrix, (out_width, out_height))
    """
    box4 = order_box_pts(box4)  # TL, TR, BR, BL
    
    # Calculate width and height from box edges
    w = float(np.linalg.norm(box4[1]-box4[0]))  # TL to TR
    h = float(np.linalg.norm(box4[2]-box4[1]))  # TR to BR
    
    # Safety check
    if w < 1: w = 1.0
    if h < 1: h = 1.0

    if force_square:
        side = int(round(max(w, h)))
        dst = np.array([[0,0],[side,0],[side,side],[0,side]], np.float32)
        out_size = (side, side)
    else:
        # Rectangle mode - PRESERVE NATURAL ASPECT RATIO
        # Don't force horizontal by swapping dimensions - this causes resizing!
        # Just use the natural dimensions from the bounding box
        out_size = (int(round(w)), int(round(h)))
        dst = np.array([
            [0, 0],
            [out_size[0], 0],
            [out_size[0], out_size[1]],
            [0, out_size[1]]
        ], np.float32)
    
    # Ensure minimum size
    out_size = (max(1, out_size[0]), max(1, out_size[1]))
    
    H = cv2.getPerspectiveTransform(box4.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, H, out_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return warped, H, out_size

def warp_pts_persp(pts, H):
    pts = np.array(pts, np.float32)
    ones = np.ones((pts.shape[0],1), np.float32)
    hom = np.hstack([pts, ones])
    dst = (H @ hom.T)
    dst = (dst[:2,:] / dst[2:,:]).T
    return dst

def closest_corner(x, y, W, H):
    corners = {"TL": (0,0), "TR": (W-1,0), "BR": (W-1,H-1), "BL": (0,H-1)}
    d2 = {k: (x-cx)**2 + (y-cy)**2 for k,(cx,cy) in corners.items()}
    return min(d2, key=d2.get)

def rotate_90mul(img, deg):
    deg = int(deg) % 360
    if deg == 0:   return img
    if deg == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # fallback
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), deg, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

# ---------- core ----------
def load_meta(json_path: Path):
    m = json.loads(Path(json_path).read_text(encoding="utf-8"))
    seg_poly = _to_points(m.get("segment_corners", [])) if "segment_corners" in m else None
    seg_square = m.get("segment_square_corners", None)
    if seg_square is not None and len(seg_square) == 4:
        seg_square = order_box_pts(_to_points(seg_square))
    else:
        seg_square = None
    qr_data = m.get("qr_corners", [])
    qr = _to_points(qr_data) if qr_data and len(qr_data) == 4 else None
    return m, seg_poly, seg_square, qr

def normalize_one(
    img_path: Path,
    json_path: Path,
    out_img_path: Path,
    out_json_path: Optional[Path]=None,
    mask_mode: str="square",
    force_square: bool=True,
    rot_mode: str="only180",
    final_size: int=0
) -> bool:
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] {stem}: cannot read image {img_path}")
        return False
    try:
        meta, seg_poly, seg_square, qr = load_meta(json_path)
    except Exception as e:
        print(f"[ERROR] {stem}: cannot read json {json_path} — {e}")
        return False

    if seg_square is None:
        print(f"[ERROR] {stem}: segment_square_corners missing — strict mode requires it")
        return False

    # 1) mask outside — default uses the square
    if mask_mode == "polygon":
        img_mask = mask_poly(img, seg_poly)
    else:
        img_mask = mask_poly(img, seg_square)  # strict square mask

    # 2) warp to axis plane using the same square
    warped, H, (Wdst, Hdst) = persp_from_box(img_mask, seg_square, force_square=force_square)

    # 3) transform QR points
    qr_warp = warp_pts_persp(qr, H) if qr is not None else None

    # 4) orientation - SIMPLE QR POSITIONING (no forced horizontal)
    final = warped
    extra_rot = 0
    qr_corner = "N/A"
    if qr_warp is not None and len(qr_warp)==4:
        qx, qy = float(np.mean(qr_warp[:,0])), float(np.mean(qr_warp[:,1]))
        Hc, Wc = final.shape[:2]
        qr_corner = closest_corner(qx, qy, Wc, Hc)
        
        if rot_mode == "only180":
            # Only rotate 180° if QR is at TR
            if qr_corner == "TR":
                final = rotate_90mul(final, 180); extra_rot = 180; qr_corner = "BL"
        else:  # any90
            # Only use 180° rotation to avoid flip/mirror issues
            if qr_corner == "TR":
                final = rotate_90mul(final, 180); extra_rot = 180; qr_corner = "BL"
            # For TL and BR, keep original orientation to avoid flip/mirror

    # 5) optional final resize (disabled by default)
    if isinstance(final_size, int) and final_size > 0:
        final = cv2.resize(final, (final_size, final_size), interpolation=cv2.INTER_AREA)

    # 6) save outputs
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), final)

    if out_json_path is not None:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        meta2 = dict(meta)
        meta2.setdefault("normalization", {})
        meta2["normalization"].update({
            "mask_mode": mask_mode,
            "force_square": bool(force_square),
            "rot_mode": rot_mode,
            "extra_rotation_deg": int(extra_rot),
            "qr_corner_after": qr_corner,
            "original_shape": list(img.shape[:2]),
            "normalized_shape": list(final.shape[:2]),
            "timestamp": datetime.now().isoformat()
        })
        Path(out_json_path).write_text(json.dumps(meta2, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {stem} -> {out_img_path.name} | shape={final.shape[:2]} | qr={qr_corner} | rot={extra_rot}")
    return True

# ---------- dataset runner ----------
def run_dataset(
    root: Path, out_root: Path,
    mask_mode: str="square", force_square: bool=True,
    rot_mode: str="only180", final_size: int=0
):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    for sub in ["images/train","images/val","meta"]:
        (out_root/sub).mkdir(parents=True, exist_ok=True)

    total=0; ok=0
    for split in ["train","val"]:
        img_dir = root/"images"/split
        meta_dir = root/"meta"
        if not img_dir.exists():
            print(f"[WARN] missing dir: {img_dir}"); continue
        for ip in sorted(list(img_dir.glob("*.jpg"))+list(img_dir.glob("*.png"))+list(img_dir.glob("*.jpeg"))):
            total += 1
            jp = meta_dir/f"{ip.stem}.json"
            if not jp.exists():
                print(f"[WARN] missing json for {ip.name}"); continue
            out_img = out_root/"images"/split/ip.name
            out_meta = out_root/"meta"/f"{ip.stem}.json"
            if normalize_one(ip, jp, out_img, out_meta, mask_mode, force_square, rot_mode, final_size):
                ok += 1
    print(f"[DONE] OK={ok}/{total}")

def main():
    ap = argparse.ArgumentParser(description="Strict square-ROI normalizer (no implicit resize)")
    sp = ap.add_subparsers(dest="mode")
    p1 = sp.add_parser("single", help="Process a single image+json")
    p1.add_argument("--image", required=True)
    p1.add_argument("--json", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--out_meta", default=None)
    p1.add_argument("--mask_mode", choices=["square","polygon"], default="square")
    p1.add_argument("--force_square", type=lambda s:s.lower()!='false', default=True)
    p1.add_argument("--rot_mode", choices=["only180","any90"], default="only180")
    p1.add_argument("--final_size", type=int, default=0)

    p2 = sp.add_parser("dataset", help="Process dataset root (images/{train,val}+meta)")
    p2.add_argument("--root", required=True)
    p2.add_argument("--out_root", required=True)
    p2.add_argument("--mask_mode", choices=["square","polygon"], default="square")
    p2.add_argument("--force_square", type=lambda s:s.lower()!='false', default=True)
    p2.add_argument("--rot_mode", choices=["only180","any90"], default="only180")
    p2.add_argument("--final_size", type=int, default=0)

    args = ap.parse_args()
    if args.mode == "single":
        normalize_one(Path(args.image), Path(args.json), Path(args.out),
                      Path(args.out_meta) if args.out_meta else None,
                      mask_mode=args.mask_mode, force_square=args.force_square,
                      rot_mode=args.rot_mode, final_size=args.final_size)
    elif args.mode == "dataset":
        run_dataset(Path(args.root), Path(args.out_root),
                    mask_mode=args.mask_mode, force_square=args.force_square,
                    rot_mode=args.rot_mode, final_size=args.final_size)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
