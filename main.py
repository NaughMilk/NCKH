# ========================= MAIN PIPELINE RUNNER ========================= #
# Connect all 10 sections and provide easy run interface

import os
import sys
import argparse
import webbrowser
import socket
from typing import Optional

def setup_environment():
    """Setup environment and paths"""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print(f"Working directory: {current_dir}")
    print(f"Python path updated")

def load_all_sections():
    """Load all 10 sections in correct order"""
    print("Loading all sections...")
    
    # Import order is important for dependencies
    sections = [
        ("sections_a", "Section A - Config & Utils"),
        ("sections_b", "Section B - GroundingDINO"),
        ("sections_c", "Section C - Background Removal"),
        ("sections_d", "Section D - U²-Net Architecture"),
        ("sections_e", "Section E - QR Helpers"),
        ("sections_f", "Section F - Dataset Writer"),
        ("sections_g", "Section G - SDY Pipeline"),
        ("sections_h", "Section H - Warehouse Checker"),
        ("sections_i", "Section I - UI Handlers"),
        ("sections_j", "Section J - UI Builder")
    ]
    
    loaded_sections = []
    
    for section_name, description in sections:
        try:
            print(f"Loading {section_name} - {description}")
            __import__(section_name)
            loaded_sections.append(section_name)
            print(f"OK: {section_name} loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load {section_name}: {e}")
            return False
    
    print(f"SUCCESS: All {len(loaded_sections)} sections loaded successfully!")
    return True

def initialize_pipeline():
    """Initialize the main pipeline"""
    try:
        print("Initializing pipeline...")
        
        # Import main components
        from sections_a.a_config import CFG, _log_info, _log_success
        from sections_i.i_model_init import init_models
        
        # Initialize models
        init_result = init_models()
        print(f"Model initialization: {init_result}")
        
        _log_success("Pipeline", "Pipeline initialized successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Pipeline initialization failed: {e}")
        return False

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

def launch_ui():
    """Launch the Gradio UI"""
    try:
        print("Launching Gradio UI...")
        
        # Import UI components
        from sections_j.j_ui_builder import build_ui
        
        # Find free port
        port = find_free_port()
        print(f"Using port: {port}")
        
        # Build UI
        ui = build_ui()
        ui.queue()
        
        # Launch
        url = f"http://127.0.0.1:{port}"
        print(f"Launching UI at: {url}")
        
        # Auto open browser
        webbrowser.open(url)
        
        ui.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"ERROR: Failed to launch UI: {e}")
        return False

def run_test():
    """Run pipeline test"""
    try:
        print("Running pipeline test...")
        
        # Import test components
        from sections_a.a_config import CFG
        from sections_g.g_sdy_core import SDYPipeline
        
        # Test basic functionality
        print(f"OK: Config loaded: {CFG.project_dir}")
        print(f"OK: SDYPipeline available: {SDYPipeline is not None}")
        
        # Test UI builder
        from sections_j.j_ui_builder import build_ui
        print(f"OK: UI builder import successful")
        
        print("SUCCESS: All tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False

def show_help():
    """Show help information"""
    help_text = """
SDY Pipeline - Smart Dataset & Training System

USAGE:
    python main.py [OPTIONS]

OPTIONS:
    --ui, -u          Launch Gradio UI (default)
    --test, -t         Run pipeline test
    --help, -h         Show this help message

EXAMPLES:
    python main.py                 # Launch UI (default)
    python main.py --ui            # Launch UI explicitly
    python main.py --test          # Run test only
    python main.py --help          # Show help

FEATURES:
    Dataset Creation: GroundingDINO + QR validation + Background Removal
    Model Training: YOLOv8 (detection) + U2-Net (segmentation)
    Warehouse Check: QR decode + YOLO detect + U2-Net segment
    QR Generation: Generate QR codes for boxes
    Advanced Settings: Configurable parameters

SECTIONS:
    A: Config & Utils          F: Dataset Writer
    B: GroundingDINO          G: SDY Pipeline  
    C: Background Removal      H: Warehouse Checker
    D: U2-Net Architecture     I: UI Handlers
    E: QR Helpers              J: UI Builder
    """
    print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SDY Pipeline Runner")
    parser.add_argument("--ui", "-u", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--test", "-t", action="store_true", help="Run pipeline test")
    
    args = parser.parse_args()
    
    # Show help if requested
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    print("SDY Pipeline Starting...")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Load all sections
    if not load_all_sections():
        print("ERROR: Failed to load sections. Exiting.")
        return
    
    # Initialize pipeline
    if not initialize_pipeline():
        print("ERROR: Failed to initialize pipeline. Exiting.")
        return
    
    # Run based on arguments
    if args.test:
        run_test()
    else:
        # Default: launch UI
        launch_ui()

if __name__ == "__main__":
    main()
