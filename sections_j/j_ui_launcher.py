# ========================= SECTION J: UI LAUNCHER ========================= #

import gradio as gr
import webbrowser
import socket
from typing import Optional

def find_free_port(start_port: int = 7860, max_port: int = 7890) -> int:
    """Find a free port in the given range"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Cannot find empty port in range: {start_port}-{max_port}")

def launch_ui(server_name: str = "127.0.0.1", start_port: int = 7860, 
              max_port: int = 7890, auto_open: bool = True) -> None:
    """
    Launch the Gradio UI with automatic port selection and browser opening
    
    Args:
        server_name: Server address (default: 127.0.0.1)
        start_port: Starting port to try (default: 7860)
        max_port: Maximum port to try (default: 7890)
        auto_open: Whether to automatically open browser (default: True)
    """
    try:
        # Find a free port
        port = find_free_port(start_port, max_port)
        
        # Import build_ui here to avoid circular imports
        from sections_j.j_ui_builder import build_ui
        
        # Build the UI
        ui = build_ui()
        
        # Launch with queue for better performance
        ui.queue()
        
        # Launch the interface
        print(f"🚀 Launching SDY Pipeline UI on http://{server_name}:{port}")
        
        if auto_open:
            # Open browser automatically
            url = f"http://{server_name}:{port}"
            print(f"🌐 Opening browser: {url}")
            webbrowser.open(url)
        
        ui.launch(
            server_name=server_name,
            server_port=port,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")
        raise

def launch_ui_simple():
    """Simple launcher with default settings"""
    launch_ui()

if __name__ == "__main__":
    launch_ui_simple()
