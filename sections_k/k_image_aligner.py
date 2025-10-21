#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_image_aligner.py
------------------
Image alignment by box ID with consistent QR positioning.

This module:
1. Groups images by box ID (from metadata)
2. Selects N random images per ID
3. Aligns them so QR codes are in the same corner
4. Exports organized folders

Author: SDY Pipeline Team
"""

from __future__ import annotations
import os
import json
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

# Import normalization utilities from dataset_normalizer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_normalizer import (
    load_meta, 
    mask_poly, 
    persp_from_box, 
    warp_pts_persp,
    closest_corner,
    rotate_90mul,
    order_box_pts,
    _to_points
)


class ImageAligner:
    """Handle image alignment by box ID"""
    
    def __init__(self, dataset_root: str, output_root: str):
        """
        Initialize ImageAligner
        
        Args:
            dataset_root: Path to dataset (should have images/train, images/val, meta/)
            output_root: Path to output folder
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset exists
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")
        
        self.images_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def scan_dataset(self) -> Dict[str, int]:
        """
        Scan dataset and group images by box ID
        
        Returns:
            Dictionary with box_id -> count
        """
        print("[INFO] Scanning dataset...")
        
        meta_dir = self.dataset_root / "meta"
        if not meta_dir.exists():
            print(f"[ERROR] Meta directory not found: {meta_dir}")
            return {}
        
        # Scan all JSON files
        json_files = list(meta_dir.glob("*.json"))
        print(f"[INFO] Found {len(json_files)} metadata files")
        
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # Get box ID (try multiple fields)
                box_id = None
                if 'id_qr' in meta:
                    box_id = str(meta['id_qr'])
                elif 'box_id' in meta:
                    box_id = str(meta['box_id'])
                elif 'qr_data' in meta:
                    # qr_data can be a string (direct ID) or a JSON dict
                    qr_data_raw = meta['qr_data']
                    if isinstance(qr_data_raw, str):
                        # Try to parse as JSON first
                        try:
                            qr_data_dict = json.loads(qr_data_raw)
                            if isinstance(qr_data_dict, dict):
                                box_id = str(qr_data_dict.get('id_qr', qr_data_dict.get('box_id', '')))
                            else:
                                # It's just a string ID
                                box_id = str(qr_data_raw)
                        except:
                            # Not JSON, treat as direct string ID
                            box_id = str(qr_data_raw)
                    elif isinstance(qr_data_raw, dict):
                        box_id = str(qr_data_raw.get('id_qr', qr_data_raw.get('box_id', '')))
                    else:
                        box_id = str(qr_data_raw)
                
                if not box_id:
                    print(f"[WARN] No box ID found in {json_path.name}")
                    continue
                
                # Find corresponding image
                stem = json_path.stem
                img_path = None
                for split in ['train', 'val']:
                    for ext in ['.jpg', '.png', '.jpeg']:
                        candidate = self.dataset_root / "images" / split / f"{stem}{ext}"
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path:
                        break
                
                if not img_path or not img_path.exists():
                    print(f"[WARN] Image not found for {json_path.name}")
                    continue
                
                # Store image info
                self.images_by_id[box_id].append({
                    'image_path': img_path,
                    'json_path': json_path,
                    'stem': stem,
                    'meta': meta
                })
                
            except Exception as e:
                print(f"[ERROR] Failed to process {json_path.name}: {e}")
                continue
        
        # Summary
        summary = {box_id: len(imgs) for box_id, imgs in self.images_by_id.items()}
        print(f"[INFO] Found {len(summary)} unique box IDs")
        for box_id, count in sorted(summary.items()):
            print(f"  - {box_id}: {count} images")
        
        return summary
    
    def align_images(
        self,
        box_id: str,
        num_images: int = 3,
        target_qr_corner: str = "BL",
        mask_mode: str = "square",
        force_square: bool = True,
        final_size: int = 0
    ) -> Tuple[List[np.ndarray], List[str], str]:
        """
        Align N images from the same box ID
        
        Args:
            box_id: Box ID to process
            num_images: Number of images to select (default: 3)
            target_qr_corner: Target QR corner position (TL/TR/BR/BL, default: BL)
            mask_mode: Mask mode (square/polygon)
            force_square: Force square output
            final_size: Final resize (0 = no resize)
        
        Returns:
            (aligned_images, image_names, log_message)
        """
        if box_id not in self.images_by_id:
            return [], [], f"[ERROR] Box ID '{box_id}' not found"
        
        available_images = self.images_by_id[box_id]
        if len(available_images) < num_images:
            print(f"[WARN] Box {box_id} has only {len(available_images)} images, using all")
            selected = available_images
        else:
            # Random selection
            selected = random.sample(available_images, num_images)
        
        aligned_images = []
        image_names = []
        log_messages = []
        
        log_messages.append(f"[INFO] Processing Box ID: {box_id}")
        log_messages.append(f"[INFO] Selected {len(selected)} images")
        
        for idx, img_info in enumerate(selected, 1):
            try:
                img_path = img_info['image_path']
                json_path = img_info['json_path']
                stem = img_info['stem']
                
                log_messages.append(f"\n[{idx}/{len(selected)}] Processing: {stem}")
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    log_messages.append(f"[ERROR] Cannot read image: {img_path}")
                    continue
                
                # Load metadata
                try:
                    meta, seg_poly, seg_square, qr = load_meta(json_path)
                except Exception as e:
                    log_messages.append(f"[ERROR] Cannot load metadata: {e}")
                    continue
                
                if seg_square is None:
                    log_messages.append(f"[ERROR] Missing segment_square_corners")
                    continue
                
                # 1) Mask outside
                if mask_mode == "polygon":
                    img_mask = mask_poly(img, seg_poly)
                else:
                    img_mask = mask_poly(img, seg_square)
                
                # 2) Warp to axis plane
                warped, H, (Wdst, Hdst) = persp_from_box(img_mask, seg_square, force_square=force_square)
                
                # 3) Transform QR points
                qr_warp = warp_pts_persp(qr, H) if qr is not None else None
                
                # 4) Orientation - align to target QR corner
                final = warped
                extra_rot = 0
                qr_corner = "N/A"
                
                if qr_warp is not None and len(qr_warp) == 4:
                    qx, qy = float(np.mean(qr_warp[:, 0])), float(np.mean(qr_warp[:, 1]))
                    Hc, Wc = final.shape[:2]
                    qr_corner = closest_corner(qx, qy, Wc, Hc)
                    
                    # Calculate rotation needed to move QR to target corner
                    corner_map = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
                    current_idx = corner_map.get(qr_corner, 0)
                    target_idx = corner_map.get(target_qr_corner, 3)
                    
                    # Calculate rotation (each step = 90 degrees)
                    rot_steps = (target_idx - current_idx) % 4
                    extra_rot = rot_steps * 90
                    
                    if extra_rot > 0:
                        final = rotate_90mul(final, extra_rot)
                        qr_corner = target_qr_corner
                    
                    log_messages.append(f"  QR corner detected: {qr_corner} -> rotated {extra_rot}° to {target_qr_corner}")
                else:
                    log_messages.append(f"  [WARN] QR corners not found, no rotation applied")
                
                # 5) Optional final resize
                if isinstance(final_size, int) and final_size > 0:
                    final = cv2.resize(final, (final_size, final_size), interpolation=cv2.INTER_AREA)
                
                # Convert BGR to RGB for display
                final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                
                aligned_images.append(final_rgb)
                image_names.append(f"{stem}_{idx}.jpg")
                
                log_messages.append(f"  ✓ Success: shape={final.shape[:2]}")
                
            except Exception as e:
                log_messages.append(f"[ERROR] Failed to process {img_info['stem']}: {e}")
                import traceback
                log_messages.append(traceback.format_exc())
                continue
        
        log_message = "\n".join(log_messages)
        return aligned_images, image_names, log_message
    
    def export_aligned_images(
        self,
        box_id: str,
        aligned_images: List[np.ndarray],
        image_names: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Export aligned images to output folder
        
        Args:
            box_id: Box ID
            aligned_images: List of aligned images (RGB)
            image_names: List of image names
        
        Returns:
            (output_folder_path, list_of_saved_paths)
        """
        if not aligned_images:
            return "", []
        
        # Create output folder for this box ID
        box_folder = self.output_root / box_id
        box_folder.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for img_rgb, name in zip(aligned_images, image_names):
            # Convert RGB back to BGR for saving
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            output_path = box_folder / name
            cv2.imwrite(str(output_path), img_bgr)
            saved_paths.append(str(output_path))
            print(f"[SAVE] {output_path}")
        
        return str(box_folder), saved_paths


def align_images_by_id(
    dataset_root: str,
    output_root: str,
    box_id: str,
    num_images: int = 3,
    target_qr_corner: str = "BL",
    mask_mode: str = "square",
    force_square: bool = True,
    final_size: int = 0
) -> Tuple[List[np.ndarray], str, str]:
    """
    Main function to align images by box ID
    
    Args:
        dataset_root: Path to dataset
        output_root: Path to output folder
        box_id: Box ID to process (or "ALL" to process all)
        num_images: Number of images to select per ID
        target_qr_corner: Target QR corner (TL/TR/BR/BL)
        mask_mode: Mask mode (square/polygon)
        force_square: Force square output
        final_size: Final resize (0 = no resize)
    
    Returns:
        (aligned_images, output_folder, log_message)
    """
    try:
        aligner = ImageAligner(dataset_root, output_root)
        summary = aligner.scan_dataset()
        
        if not summary:
            return [], "", "[ERROR] No images found in dataset"
        
        if box_id == "ALL":
            # Process all IDs
            all_images = []
            all_logs = []
            
            for bid in summary.keys():
                aligned_imgs, img_names, log = aligner.align_images(
                    bid, num_images, target_qr_corner, mask_mode, force_square, final_size
                )
                
                if aligned_imgs:
                    folder, _ = aligner.export_aligned_images(bid, aligned_imgs, img_names)
                    all_images.extend(aligned_imgs)
                    all_logs.append(log)
            
            log_message = "\n\n".join(all_logs)
            return all_images, str(output_root), log_message
        
        else:
            # Process single ID
            aligned_imgs, img_names, log = aligner.align_images(
                box_id, num_images, target_qr_corner, mask_mode, force_square, final_size
            )
            
            if aligned_imgs:
                folder, _ = aligner.export_aligned_images(box_id, aligned_imgs, img_names)
                return aligned_imgs, folder, log
            else:
                return [], "", log
    
    except Exception as e:
        import traceback
        error_msg = f"[ERROR] {e}\n{traceback.format_exc()}"
        return [], "", error_msg


def process_dataset_alignment(
    dataset_root: str,
    output_root: str,
    num_images: int = 3,
    target_qr_corner: str = "BL"
) -> Tuple[List[np.ndarray], str]:
    """
    Process entire dataset - align images for all box IDs
    
    Args:
        dataset_root: Path to dataset
        output_root: Path to output folder
        num_images: Number of images per ID
        target_qr_corner: Target QR corner position
    
    Returns:
        (preview_images, summary_message)
    """
    try:
        aligner = ImageAligner(dataset_root, output_root)
        summary = aligner.scan_dataset()
        
        if not summary:
            return [], "[ERROR] No images found in dataset"
        
        preview_images = []
        total_processed = 0
        total_ids = len(summary)
        
        summary_lines = [
            f"[INFO] Processing {total_ids} box IDs",
            f"[INFO] Selecting {num_images} images per ID",
            f"[INFO] Target QR corner: {target_qr_corner}",
            ""
        ]
        
        for idx, box_id in enumerate(sorted(summary.keys()), 1):
            print(f"\n[{idx}/{total_ids}] Processing Box ID: {box_id}")
            
            aligned_imgs, img_names, log = aligner.align_images(
                box_id, num_images, target_qr_corner
            )
            
            if aligned_imgs:
                folder, saved_paths = aligner.export_aligned_images(box_id, aligned_imgs, img_names)
                total_processed += len(aligned_imgs)
                
                # Add first image to preview
                if aligned_imgs:
                    preview_images.append(aligned_imgs[0])
                
                summary_lines.append(f"✓ {box_id}: {len(aligned_imgs)} images saved to {folder}")
            else:
                summary_lines.append(f"✗ {box_id}: Failed (see logs)")
        
        summary_lines.append("")
        summary_lines.append(f"[DONE] Processed {total_processed} images from {total_ids} box IDs")
        summary_lines.append(f"[OUTPUT] {output_root}")
        
        summary_message = "\n".join(summary_lines)
        return preview_images, summary_message
    
    except Exception as e:
        import traceback
        error_msg = f"[ERROR] {e}\n{traceback.format_exc()}"
        return [], error_msg


if __name__ == "__main__":
    # Test
    import argparse
    
    ap = argparse.ArgumentParser(description="Align images by box ID")
    ap.add_argument("--dataset", required=True, help="Dataset root path")
    ap.add_argument("--output", required=True, help="Output root path")
    ap.add_argument("--box_id", default="ALL", help="Box ID to process (or ALL)")
    ap.add_argument("--num_images", type=int, default=3, help="Number of images per ID")
    ap.add_argument("--target_corner", default="BL", choices=["TL", "TR", "BR", "BL"], help="Target QR corner")
    
    args = ap.parse_args()
    
    aligned_imgs, output_folder, log = align_images_by_id(
        args.dataset,
        args.output,
        args.box_id,
        args.num_images,
        args.target_corner
    )
    
    print("\n" + "="*50)
    print("ALIGNMENT LOG")
    print("="*50)
    print(log)
    print("\n" + "="*50)
    print(f"Output: {output_folder}")
    print(f"Total aligned images: {len(aligned_imgs)}")
    print("="*50)

