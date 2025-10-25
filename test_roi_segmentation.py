#!/usr/bin/env python3
"""
Test script for ROI-based U²-Net segmentation in warehouse
"""

import os
import sys
import cv2
import numpy as np
import time
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_roi_segmentation():
    """Test the new ROI-based segmentation approach"""
    print("🧪 Testing ROI-based U²-Net segmentation...")
    
    try:
        # Import required modules
        from sections_a.a_config import CFG, _log_info, _log_success, _log_warning
        from sections_h.h_warehouse_core import warehouse_check_frame
        
        # Create a test image with a box
        test_image = create_test_image()
        
        # Test with mock model paths (if models exist)
        yolo_path = os.path.join(CFG.project_dir, "warehouse_models", "yolo_model.pt")
        u2net_path = os.path.join(CFG.project_dir, "warehouse_models", "u2net_model.pth")
        
        if not os.path.exists(yolo_path) or not os.path.exists(u2net_path):
            print("⚠️  Model files not found. Please upload models first.")
            print(f"   YOLO: {yolo_path}")
            print(f"   U²-Net: {u2net_path}")
            return False
        
        print(f"✅ Model files found:")
        print(f"   YOLO: {yolo_path}")
        print(f"   U²-Net: {u2net_path}")
        
        # Test warehouse check
        print("\n🔍 Running warehouse check with ROI segmentation...")
        start_time = time.time()
        
        visualizations, log_message, results = warehouse_check_frame(
            test_image, 
            yolo_model_path=yolo_path,
            u2net_model_path=u2net_path,
            enable_deskew=False,
            enable_force_rectangle=False
        )
        
        processing_time = time.time() - start_time
        
        if visualizations is not None:
            print(f"✅ Warehouse check completed in {processing_time*1000:.1f}ms")
            print(f"📊 Results: {len(visualizations)} visualizations")
            print(f"📝 Log: {log_message}")
            
            # Save test results
            save_test_results(visualizations, log_message, results)
            return True
        else:
            print(f"❌ Warehouse check failed: {log_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """Create a test image with a box for testing"""
    # Create a simple test image with a box
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a box
    cv2.rectangle(img, (100, 100), (300, 200), (0, 0, 0), 2)
    cv2.putText(img, "Test Box", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add some noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def save_test_results(visualizations, log_message, results):
    """Save test results for inspection"""
    try:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save log message
        with open(os.path.join(output_dir, "test_log.txt"), 'w', encoding='utf-8') as f:
            f.write(log_message)
        
        # Save visualizations
        for i, (vis_img, title) in enumerate(visualizations):
            if isinstance(vis_img, Image.Image):
                vis_img.save(os.path.join(output_dir, f"visualization_{i}_{title.replace(' ', '_')}.png"))
            else:
                cv2.imwrite(os.path.join(output_dir, f"visualization_{i}_{title.replace(' ', '_')}.png"), vis_img)
        
        print(f"💾 Test results saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Could not save test results: {e}")

if __name__ == "__main__":
    print("🚀 Starting ROI Segmentation Test")
    print("=" * 50)
    
    success = test_roi_segmentation()
    
    print("=" * 50)
    if success:
        print("✅ ROI segmentation test completed successfully!")
    else:
        print("❌ ROI segmentation test failed!")
    
    print("\n📋 Key improvements with ROI segmentation:")
    print("   • Faster processing (only segment ROI, not full image)")
    print("   • More accurate segmentation (focused on relevant area)")
    print("   • Better memory efficiency")
    print("   • Reduced noise from background")
