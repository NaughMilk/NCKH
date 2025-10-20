#!/usr/bin/env python3
"""
Run Pipeline - Wrapper để chạy unified pipeline
===============================================

File này cung cấp interface đơn giản để chạy pipeline thống nhất
tương đương với NCC_PIPELINE_NEW.py gốc.
"""

import sys
import os
import socket
import webbrowser
from pathlib import Path

# Thêm thư mục hiện tại vào Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_unified_pipeline():
    """Chạy unified pipeline"""
    try:
        # Import final unified pipeline
        from final_unified_pipeline import main, create_final_pipeline
        
        print("[START] Starting Unified Pipeline...")
        print("=" * 50)
        
        # Chạy main
        success = main()
        
        if success:
            print("\n[OK] Pipeline loaded successfully!")
            
            # Thống kê components
            functions = [name for name, obj in globals().items() 
                        if not name.startswith('_') and callable(obj) and not isinstance(obj, type)]
            classes = [name for name, obj in globals().items() 
                      if not name.startswith('_') and isinstance(obj, type)]
            
            print(f"[INFO] Functions: {len(functions)}")
            print(f"[INFO] Classes: {len(classes)}")
            
            # Import key components để sử dụng
            import final_unified_pipeline
            
            print("\n[COMPONENTS] Key components available:")
            print("  - Config: Configuration class")
            print("  - SDYPipeline: Main pipeline class") 
            print("  - build_ui: Gradio UI builder")
            print("  - All other functions from original file")
            
            return True
        else:
            print("[ERROR] Failed to load pipeline!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error loading unified pipeline: {e}")
        return False

def run_gradio_ui():
    """Chạy Gradio UI (tương đương với file gốc)"""
    try:
        # Import final unified pipeline
        import final_unified_pipeline
        
        print("[UI] Starting Gradio UI...")
        
        # Kiểm tra xem build_ui có sẵn không
        if hasattr(final_unified_pipeline, 'build_ui'):
            ui = final_unified_pipeline.build_ui()
            # Tự chọn port rảnh từ 7860-7890 nếu 7862 bận
            port_candidates = list(range(7860, 7891))
            chosen_port = None
            for p in port_candidates:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind(("127.0.0.1", p))
                        chosen_port = p
                        break
                    except OSError:
                        continue
            if chosen_port is None:
                raise RuntimeError("No free port available in 7860-7890")
            url = f"http://127.0.0.1:{chosen_port}"
            print(f"[UI] Launching at {url}")
            # Tự mở trình duyệt
            try:
                webbrowser.open(url)
            except Exception:
                pass
            ui.queue().launch(server_name="127.0.0.1", server_port=chosen_port)
        else:
            print("[ERROR] build_ui function not found!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running Gradio UI: {e}")
        return False

def show_help():
    """Hiển thị help"""
    print("""
UNIFIED PIPELINE - Help
======================

Cach su dung:
1. Chay pipeline: python run_pipeline.py
2. Chay UI: python run_pipeline.py --ui
3. Test: python run_pipeline.py --test

Options:
  --ui         Chay Gradio UI
  --test       Test pipeline loading
  --show-help  Hien thi help nay

Examples:
  python run_pipeline.py --ui
  python run_pipeline.py --test
""")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Pipeline Runner')
    parser.add_argument('--ui', action='store_true', help='Run Gradio UI')
    parser.add_argument('--test', action='store_true', help='Test pipeline loading')
    parser.add_argument('--show-help', action='store_true', help='Show help')
    
    args = parser.parse_args()
    
    if args.show_help:
        show_help()
        return
    
    if args.test:
        print("[TEST] Testing pipeline loading...")
        success = run_unified_pipeline()
        if success:
            print("[OK] Test passed!")
        else:
            print("[ERROR] Test failed!")
        return
    
    if args.ui:
        print("[UI] Starting UI mode...")
        run_gradio_ui()
    else:
        # Mặc định khi ấn Run trong editor: mở UI trực tiếp
        print("[INFO] No args provided -> launching UI by default")
        run_gradio_ui()

if __name__ == "__main__":
    main()
