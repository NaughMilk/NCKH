#!/usr/bin/env python3
"""
Script chạy White Ring Segmentation Demo
========================================
"""

import subprocess
import sys
import os

def install_requirements():
    """Cài đặt dependencies"""
    print("🔧 Đang cài đặt dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_gradio.txt"])
        print("✅ Cài đặt dependencies thành công!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi cài đặt dependencies: {e}")
        return False

def run_demo():
    """Chạy demo"""
    print("🚀 Đang khởi động White Ring Segmentation Demo...")
    try:
        from white_ring_gradio_demo import create_gradio_interface
        demo = create_gradio_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Lỗi khởi động demo: {e}")
        print("💡 Hãy thử cài đặt dependencies trước:")
        print("   pip install -r requirements_gradio.txt")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 WHITE RING SEGMENTATION DEMO")
    print("=" * 60)
    
    # Kiểm tra dependencies
    try:
        import gradio
        import cv2
        import numpy as np
        print("✅ Tất cả dependencies đã sẵn sàng!")
    except ImportError as e:
        print(f"⚠️  Thiếu dependency: {e}")
        if not install_requirements():
            sys.exit(1)
    
    # Chạy demo
    run_demo()
