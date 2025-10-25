# ========================= SIMPLE RUNNER ========================= #
# Just double-click this file to run the pipeline!

import os
import sys
import webbrowser
import socket

def find_free_port(start_port=7860, max_port=7890):
    """Find a free port"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return 7860  # fallback

def main():
    print("=" * 50)
    print("SDY Pipeline - Starting...")
    print("=" * 50)
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Load all sections
        print("Loading sections...")
        sections = [
            "sections_a", "sections_b", "sections_c", "sections_d", "sections_e",
            "sections_f", "sections_g", "sections_h", "sections_i", "sections_j", "sections_k"
        ]
        
        for section in sections:
            try:
                __import__(section)
                print(f"[OK] {section}")
            except Exception as e:
                print(f"[ERROR] {section}: {e}")
        
        # Import and build UI
        print("\nBuilding UI...")
        from sections_j.j_ui_builder import build_ui
        
        # Find free port
        port = find_free_port()
        print(f"Using port: {port}")
        
        # Build UI
        print("Creating UI components...")
        ui = build_ui()
        print("UI created successfully!")
        
        # Queue for better performance
        ui.queue()
        
        # Launch
        url = f"http://127.0.0.1:{port}"
        print(f"\n[LAUNCH] Starting UI at: {url}")
        print("Opening browser...")
        
        # Auto open browser
        webbrowser.open(url)
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Pipeline is running!")
        print(f"[URL] Open: {url}")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Launch UI
        ui.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Pipeline stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
