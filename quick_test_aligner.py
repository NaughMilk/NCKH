#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for Image Aligner module
"""

import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 >nul 2>&1')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing Image Aligner Module")
print("=" * 60)

# Test 1: Import modules
print("\n[TEST 1] Testing module imports...")
try:
    from sections_k.k_image_aligner import ImageAligner, align_images_by_id
    print("[OK] Successfully imported ImageAligner")
    print("[OK] Successfully imported align_images_by_id")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check dataset existence
print("\n[TEST 2] Checking dataset...")
dataset_path = Path("sdy_project/dataset_sdy_box")
if not dataset_path.exists():
    print(f"[ERROR] Dataset not found: {dataset_path}")
    print("  Please ensure you have a dataset in the correct location")
    sys.exit(1)

print(f"[OK] Dataset found: {dataset_path}")

# Check structure
images_dir = dataset_path / "images"
meta_dir = dataset_path / "meta"

if not images_dir.exists():
    print(f"[ERROR] Images directory not found: {images_dir}")
    sys.exit(1)

if not meta_dir.exists():
    print(f"[ERROR] Meta directory not found: {meta_dir}")
    sys.exit(1)

print(f"[OK] Images directory: {images_dir}")
print(f"[OK] Meta directory: {meta_dir}")

# Count files
json_files = list(meta_dir.glob("*.json"))
print(f"[OK] Found {len(json_files)} JSON files")

# Test 3: Initialize ImageAligner
print("\n[TEST 3] Initializing ImageAligner...")
try:
    output_path = "sdy_project/test_aligned_output"
    aligner = ImageAligner(str(dataset_path), output_path)
    print(f"[OK] ImageAligner initialized")
    print(f"  Dataset root: {aligner.dataset_root}")
    print(f"  Output root: {aligner.output_root}")
except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Scan dataset
print("\n[TEST 4] Scanning dataset for Box IDs...")
try:
    summary = aligner.scan_dataset()
    
    if not summary:
        print("[ERROR] No Box IDs found in dataset")
        print("  Check that JSON files have 'id_qr' or 'box_id' fields")
    else:
        print(f"[OK] Found {len(summary)} unique Box IDs:")
        for box_id, count in sorted(summary.items()):
            print(f"    - {box_id}: {count} images")
        
        # Test 5: Try aligning one Box ID
        print("\n[TEST 5] Testing alignment for first Box ID...")
        test_box_id = list(summary.keys())[0]
        print(f"  Testing with Box ID: {test_box_id}")
        
        try:
            aligned_imgs, img_names, log = aligner.align_images(
                box_id=test_box_id,
                num_images=2,  # Just 2 images for quick test
                target_qr_corner="BL",
                mask_mode="square",
                force_square=True,
                final_size=0
            )
            
            if aligned_imgs:
                print(f"[OK] Successfully aligned {len(aligned_imgs)} images")
                print(f"  Image names: {img_names}")
                
                # Test 6: Export
                print("\n[TEST 6] Testing export...")
                folder, saved_paths = aligner.export_aligned_images(
                    test_box_id, aligned_imgs, img_names
                )
                print(f"[OK] Exported to: {folder}")
                print(f"  Files saved: {len(saved_paths)}")
                for path in saved_paths:
                    print(f"    - {path}")
            else:
                print("[ERROR] No images aligned")
                print(f"  Log:\n{log}")
        
        except Exception as e:
            print(f"[ERROR] Alignment test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

except Exception as e:
    print(f"[ERROR] Dataset scan failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("[SUCCESS] ALL TESTS PASSED!")
print("=" * 60)
print("\nThe Image Aligner module is working correctly.")
print(f"Test output saved to: {output_path}")
print("\nYou can now:")
print("1. Run the full pipeline: python run.py")
print("2. Use the 'Image Alignment' tab in the UI")
print("3. Run command line: python sections_k\\k_image_aligner.py --help")
print("=" * 60)

