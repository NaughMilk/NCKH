# ========================= SECTION I: DEXINED FUNCTIONS ========================= #

import os
import urllib.request

# Import dependencies
from sections_a.a_config import CFG
from sections_a.a_edges import EDGE, CUDA_AVAILABLE

def get_system_status():
    """Get current system status"""
    gpu_status = "[GPU] Available" if CUDA_AVAILABLE else "[CPU] Only"
    dexined_status = "[DexiNed] Ready" if EDGE.dexi and EDGE.dexi.available() else "[DexiNed] Not Initialized"
    backend_type = "ONNX" if EDGE.dexi and EDGE.dexi.onnx_sess else "PyTorch" if EDGE.dexi and EDGE.dexi.torch_model else "Canny"
    
    return f"{gpu_status} | {dexined_status} | Backend: {backend_type}"

def auto_download_dexined():
    """Auto-download DexiNed weights if not available"""
    onnx_path = "weights/dexined.onnx"
    torch_path = "weights/dexined.pth"
    
    # Create weights directory if not exists
    os.makedirs("weights", exist_ok=True)
    
    downloaded = []
    
    # Try to download ONNX model (using a more reliable source)
    if not os.path.isfile(onnx_path):
        try:
            # Try multiple sources for ONNX model
            onnx_urls = [
                "https://github.com/xavysp/DexiNed/releases/download/v1.0/dexined.onnx",
                "https://huggingface.co/xavysp/DexiNed/resolve/main/dexined.onnx"
            ]
            
            for onnx_url in onnx_urls:
                try:
                    print(f"Downloading DexiNed ONNX from {onnx_url}...")
                    urllib.request.urlretrieve(onnx_url, onnx_path)
                    downloaded.append("ONNX")
                    break
                except:
                    continue
        except Exception as e:
            print(f"Failed to download ONNX: {e}")
    
    # Try to download PyTorch model
    if not os.path.isfile(torch_path):
        try:
            # Try multiple sources for PyTorch model
            torch_urls = [
                "https://github.com/xavysp/DexiNed/releases/download/v1.0/dexined.pth",
                "https://huggingface.co/xavysp/DexiNed/resolve/main/dexined.pth"
            ]
            
            for torch_url in torch_urls:
                try:
                    print(f"Downloading DexiNed PyTorch from {torch_url}...")
                    urllib.request.urlretrieve(torch_url, torch_path)
                    downloaded.append("PyTorch")
                    break
                except:
                    continue
        except Exception as e:
            print(f"Failed to download PyTorch: {e}")
    
    return downloaded

def auto_init_dexined():
    """Auto-initialize DexiNed with auto-download"""
    try:
        # First try to auto-download if needed
        downloaded = auto_download_dexined()
        
        # Try to initialize with existing or downloaded weights
        onnx_path = "weights/dexined.onnx"
        torch_path = "weights/dexined.pth"
        short_side = 1024
        
        EDGE.init_dexi(onnx_path, torch_path, short_side)
        ok = EDGE.dexi is not None and EDGE.dexi.available()
        
        status = f"✅ DexiNed Auto-Init: {'SUCCESS' if ok else 'FAILED'}\n"
        if downloaded:
            status += f"📥 Downloaded: {', '.join(downloaded)}\n"
        status += f"🧠 Backend: {'ONNX' if EDGE.dexi and EDGE.dexi.onnx_sess else 'PyTorch' if EDGE.dexi and EDGE.dexi.torch_model else 'Canny'}\n"
        status += f"🚀 GPU: {'ON' if EDGE.use_gpu else 'OFF'}\n"
        status += f"📊 System: {get_system_status()}"
        
        return status
    except Exception as e:
        return f"❌ DexiNed Auto-Init Failed: {e}\n📊 System: {get_system_status()}"

def init_dexined_backend(onnx_path, torch_path, short_side):
    """Initialize DexiNed backend"""
    try:
        EDGE.init_dexi(onnx_path, torch_path, short_side)
        ok = EDGE.dexi is not None and EDGE.dexi.available()
        return f"DexiNed ready: {ok}"
    except Exception as e:
        return f"DexiNed init failed: {e}"

def set_gpu_mode(use_gpu):
    """Enable/disable GPU acceleration"""
    EDGE.set_gpu_mode(use_gpu)
    return f"GPU mode: {'ON' if EDGE.use_gpu else 'OFF'}"
