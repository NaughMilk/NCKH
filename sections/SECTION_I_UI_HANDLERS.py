# ========================= SECTION I: UI HANDLERS ========================= #
# ========================= SECTION I: UI HANDLERS ========================= #

import os
import sys
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional



import os
import sys
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
pipe: Optional[SDYPipeline] = None

def init_models() -> str:
    """Initialize all models and ensure complete dataset structure"""
    global pipe
    try:
        ensure_dir(CFG.project_dir)
        pipe = SDYPipeline(CFG)
        
        # CRITICAL: Ensure complete dataset structure is created
        _log_info("Init Models", "Creating complete dataset structure...")
        
        # 1. Create registry directory and initialize
        registry_dir = os.path.join(CFG.project_dir, "registry")
        ensure_dir(registry_dir)
        registry = DatasetRegistry(CFG.project_dir)
        _log_success("Init Models", "Dataset registry initialized")
        
        # 2. Create default session directories to ensure structure exists
        default_session = make_session_id()  # Creates vYYYYMMDD-HHMMSS
        default_yolo_root = os.path.join(CFG.project_dir, "datasets", "yolo", default_session)
        default_u2net_root = os.path.join(CFG.project_dir, "datasets", "u2net", default_session)
        
        # Create all required directories
        required_dirs = [
            # YOLO directories
            os.path.join(default_yolo_root, "images", "train"),
            os.path.join(default_yolo_root, "images", "val"),
            os.path.join(default_yolo_root, "labels", "train"),
            os.path.join(default_yolo_root, "labels", "val"),
            os.path.join(default_yolo_root, "meta"),
            # U²-Net directories
            os.path.join(default_u2net_root, "images", "train"),
            os.path.join(default_u2net_root, "images", "val"),
            os.path.join(default_u2net_root, "masks", "train"),
            os.path.join(default_u2net_root, "masks", "val"),
            os.path.join(default_u2net_root, "meta"),
            # Training directories
            os.path.join(CFG.project_dir, "runs_sdy"),
            os.path.join(CFG.project_dir, CFG.u2_runs_dir),
            os.path.join(CFG.project_dir, CFG.rejected_images_dir)
        ]
        
        for dir_path in required_dirs:
            ensure_dir(dir_path)
        
        _log_success("Init Models", f"Created default session: {default_session}")
        _log_success("Init Models", f"YOLO structure: {default_yolo_root}")
        _log_success("Init Models", f"U²-Net structure: {default_u2net_root}")
        
        # 3. Create initial data.yaml files
        yolo_yaml_path = os.path.join(default_yolo_root, "data.yaml")
        yolo_yaml_content = f"""path: {os.path.abspath(default_yolo_root)}
train: images/train
val: images/val

nc: 22
names: {list(range(22))}
"""
        atomic_write_text(yolo_yaml_path, yolo_yaml_content)
        _log_success("Init Models", f"Created YOLO data.yaml: {yolo_yaml_path}")
        
        gpu_status = ""
        if torch.cuda.is_available():
            if not _suppress_all_cuda_logs:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_status = f"\n\n[GPU] {gpu_name} ({gpu_mem:.1f} GB)"
        
        return f"[SUCCESS] Models loaded successfully on {CFG.device}\n[INFO] Project: {os.path.abspath(CFG.project_dir)}\n[INFO] Default session: {default_session}\n[SUCCESS] Complete dataset structure created{gpu_status}"
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"

def update_gdino_params(prompt: str, box_thr: float, text_thr: float, hand_detection_thr: float, bg_model: str, feather: int, 
                       use_white_ring: bool, seg_mode: str, edge_backend: str, dexi_thr: float, canny_lo: int, canny_hi: int,
                       dilate_px: int, close_px: int, ban_border: int, min_area_ratio: float, 
                       rect_score_min: float, ar_min: float, ar_max: float, center_cov_min: float, 
                       erode_inner: int, min_comp_area: int, smooth_mode: str, smooth_iterations: int,
                       gaussian_kernel: int, use_shadow_robust_edges: bool, force_rectify: str,
                       rect_pad: int, use_convex_hull: bool, use_gpu: bool):
    """Update GroundingDINO, Background Removal, and White-ring params"""
    try:
        CFG.current_prompt = prompt.strip() or CFG.gdino_prompt
        CFG.current_box_thr = max(0.01, min(1.0, box_thr))
        CFG.current_text_thr = max(0.01, min(1.0, text_thr))
        CFG.current_hand_detection_thr = max(0.01, min(1.0, hand_detection_thr))
        
        # White-ring segmentation params (priority)
        CFG.use_white_ring_seg = use_white_ring
        CFG.seg_mode = seg_mode
        
        # DexiNed Edge Detection (Main Backend)
        CFG.video_backend = edge_backend
        CFG.video_dexi_thr = max(0.05, min(0.8, dexi_thr))
        CFG.canny_lo = max(10, min(100, canny_lo))
        CFG.canny_hi = max(50, min(300, canny_hi))
        
        # GPU Settings
        CFG.video_use_gpu = use_gpu
        EDGE.set_gpu_mode(use_gpu)
        
        # Morphology
        CFG.dilate_px = max(0, min(5, dilate_px))
        CFG.close_px = max(0, min(30, close_px))
        
        # Contour filtering
        CFG.ban_border_px = max(1, min(50, ban_border))
        CFG.min_area_ratio = max(0.1, min(0.8, min_area_ratio))
        CFG.rect_score_min = max(0.3, min(1.0, rect_score_min))
        
        # Shape constraints
        CFG.ar_min = max(0.1, min(1.0, ar_min))
        CFG.ar_max = max(1.0, min(3.0, ar_max))
        CFG.center_cov_min = max(0.1, min(1.0, center_cov_min))
        
        # Final processing
        CFG.erode_inner_px = max(0, min(10, erode_inner))
        CFG.min_comp_area = max(500, min(20000, min_comp_area))
        
        # Edge smoothing
        CFG.smooth_mode = smooth_mode
        CFG.smooth_iterations = max(0, min(5, smooth_iterations))
        CFG.gaussian_kernel = max(3, min(15, gaussian_kernel))
        
        # Post-processing
        CFG.use_shadow_robust_edges = use_shadow_robust_edges
        CFG.force_rectify = force_rectify
        CFG.rect_pad = max(0, min(20, rect_pad))
        CFG.use_convex_hull = use_convex_hull
        
        # Legacy background removal params (only if white-ring is disabled)
        if not use_white_ring:
            CFG.bg_removal_model = bg_model
            CFG.feather_px = max(0, min(20, feather))
        
        status_msg = f"✅ Updated:\nPrompt: {CFG.current_prompt}\nBox: {CFG.current_box_thr}\nText: {CFG.current_text_thr}\n"
        
        if use_white_ring:
            status_msg += f"🧠 DexiNed White-ring: ENABLED ({seg_mode})\n"
            status_msg += f"   - Backend: {edge_backend} (DexiNed thr={dexi_thr:.2f}, Canny {canny_lo}-{canny_hi})\n"
            status_msg += f"   - GPU: {'ON' if use_gpu else 'OFF'}\n"
            status_msg += f"   - Morphology: dilate={dilate_px}, close={close_px}\n"
            status_msg += f"   - Filtering: ban_border={ban_border}px, min_area={min_area_ratio:.2f}\n"
            status_msg += f"   - Shape: AR={ar_min:.1f}-{ar_max:.1f}, rect_score={rect_score_min:.2f}\n"
            status_msg += f"   - Center: {center_cov_min:.2f}, erode={erode_inner}px\n"
            status_msg += f"   - Min Comp Area: {min_comp_area}\n"
            status_msg += f"   - Smooth: {smooth_mode} ({smooth_iterations} iter, kernel {gaussian_kernel})\n"
            status_msg += f"   - Shadow Robust: {use_shadow_robust_edges}\n"
            status_msg += f"   - Force Rectify: {force_rectify} (pad {rect_pad}px)\n"
            status_msg += f"   - Convex Hull: {use_convex_hull}\n"
            status_msg += f"🎨 Legacy BG Removal: DISABLED (DexiNed White-ring active)"
        else:
            status_msg += f"🎨 Legacy BG Removal: {bg_model}\n"
            status_msg += f"🔲 White-ring: DISABLED"
        
        return status_msg
    except Exception as e:
        return f"❌ Error: {e}"

def update_video_params(backend, canny_lo, canny_hi, dexi_thr, dilate_iters, close_kernel,
                       min_area_ratio, rect_score_min, ar_min, ar_max, erode_inner,
                       smooth_close, smooth_open, use_hull, rectify_mode, rect_pad,
                       expand_factor, mode, min_comp_area, show_green_frame,
                       frame_step, max_frames, keep_only_detected, use_pair_filter,
                       pair_min_gap, pair_max_gap, lock_enable, lock_n_warmup,
                       lock_trim, lock_pad, use_gpu):
    """Update video processing parameters"""
    try:
        # Video Settings
        CFG.video_frame_step = max(1, min(20, frame_step))
        CFG.video_max_frames = max(0, min(500, max_frames))
        CFG.video_keep_only_detected = keep_only_detected
        
        # Edge Detection Backend
        CFG.video_backend = backend
        CFG.video_dexi_thr = max(0.05, min(0.8, dexi_thr))
        CFG.video_canny_lo = max(0, min(255, canny_lo))
        CFG.video_canny_hi = max(0, min(255, canny_hi))
        
        # Morphology & Filtering
        CFG.video_dilate_iters = max(0, min(5, dilate_iters))
        CFG.video_close_kernel = max(3, min(31, close_kernel))
        CFG.video_min_area_ratio = max(5, min(80, min_area_ratio))
        CFG.video_rect_score_min = max(0.3, min(0.95, rect_score_min))
        CFG.video_ar_min = max(0.4, min(1.0, ar_min))
        CFG.video_ar_max = max(1.0, min(3.0, ar_max))
        CFG.video_erode_inner = max(0, min(10, erode_inner))
        
        # Pair-edge Filter
        CFG.video_use_pair_filter = use_pair_filter
        CFG.video_pair_min_gap = max(2, min(20, pair_min_gap))
        CFG.video_pair_max_gap = max(8, min(40, pair_max_gap))
        
        # Smooth & Rectify
        CFG.video_smooth_close = max(0, min(31, smooth_close))
        CFG.video_smooth_open = max(0, min(15, smooth_open))
        CFG.video_use_hull = use_hull
        CFG.video_rectify_mode = rectify_mode
        CFG.video_rect_pad = max(0, min(20, rect_pad))
        CFG.video_expand_factor = max(0.5, min(2.0, expand_factor))
        
        # Display Mode
        CFG.video_mode = mode
        CFG.video_min_comp_area = max(0, min(10000, min_comp_area))
        CFG.video_show_green_frame = show_green_frame
        
        # Size-Lock Controls
        CFG.video_lock_enable = lock_enable
        CFG.video_lock_n_warmup = max(10, min(200, lock_n_warmup))
        CFG.video_lock_trim = max(0.0, min(0.3, lock_trim))
        CFG.video_lock_pad = max(0, min(20, lock_pad))
        
        # GPU Acceleration
        CFG.video_use_gpu = use_gpu
        
        status_msg = f"✅ Video Processing Parameters Updated:\n"
        status_msg += f"🎬 Backend: {backend} | DexiNed thr={dexi_thr:.2f} | Canny {canny_lo}-{canny_hi}\n"
        status_msg += f"🔧 Morphology: dilate={dilate_iters}, close={close_kernel}, min_area={min_area_ratio}%\n"
        status_msg += f"[SHAPE] AR={ar_min:.1f}-{ar_max:.1f}, rect_score≥{rect_score_min:.2f}\n"
        status_msg += f"[SMOOTH] close={smooth_close}, open={smooth_open}, hull={use_hull}\n"
        status_msg += f"[RECTIFY] {rectify_mode} (pad {rect_pad}px, expand {expand_factor:.1f}x)\n"
        status_msg += f"[FILTER] pair_edge={use_pair_filter} ({pair_min_gap}-{pair_max_gap}px)\n"
        status_msg += f"[SIZE-LOCK] {'ON' if lock_enable else 'OFF'} (warmup={lock_n_warmup}, trim={lock_trim:.2f})\n"
        status_msg += f"[SETTINGS] step={frame_step}, max_frames={max_frames}, keep_detected={keep_only_detected}\n"
        status_msg += f"[GPU] {'ON' if use_gpu else 'OFF'}"
        
        return status_msg
    except Exception as e:
        return f"[ERROR] Error updating video params: {e}"

def get_system_status():
    """Get current system status"""
    gpu_status = "[GPU] Available" if CUDA_AVAILABLE else "[CPU] Only"
    dexined_status = "[DexiNed] Ready" if EDGE.dexi and EDGE.dexi.available() else "[DexiNed] Not Initialized"
    backend_type = "ONNX" if EDGE.dexi and EDGE.dexi.onnx_sess else "PyTorch" if EDGE.dexi and EDGE.dexi.torch_model else "None"
    
    return f"{gpu_status} | {dexined_status} | Backend: {backend_type}"

def auto_download_dexined():
    """Auto-download DexiNed weights if not available"""
    import urllib.request
    import os
    
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
        status += f"🧠 Backend: {'ONNX' if EDGE.dexi and EDGE.dexi.onnx_sess else 'PyTorch' if EDGE.dexi and EDGE.dexi.torch_model else 'None'}\n"
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

def decode_qr_info(qr_file):
    """Decode an uploaded QR image and report the info that would be used by GroundingDINO"""
    try:
        p = _get_path(qr_file)
        if not p:
            return "[ERROR] No file provided"
        img = cv2.imread(p)
        if img is None:
            return f"[ERROR] Cannot read image: {p}"
        qr = QR()
        s, pts = qr.decode(img)
        if not s:
            return "[ERROR] No QR detected in image"
        parsed = parse_qr_payload(s)
        qr_id = (parsed.get("_qr") if isinstance(parsed, dict) else None) or str(s).strip()
        meta_path = os.path.join(CFG.project_dir, CFG.qr_meta_dir, f"{qr_id}.json")
        lines = []
        lines.append(f"QR Raw: {s}")
        lines.append(f"Parsed ID: {qr_id}")
        lines.append(f"QR Points: {pts.tolist() if pts is not None else 'None'}")
        lines.append("")
        lines.append(f"Meta Path: {meta_path}")
        fruits = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # Prefer full fruits map; fallback to single fruit_name/quantity
                if isinstance(meta.get('fruits'), dict) and len(meta['fruits']) > 0:
                    fruits = meta['fruits']
                else:
                    fname = meta.get('fruit_name')
                    qty = meta.get('quantity', 0)
                    if fname:
                        fruits = {str(fname): int(qty)}
                lines.append(f"Loaded Meta: {json.dumps(meta, ensure_ascii=False)}")
            except Exception as e:
                lines.append(f"[WARN] Failed to read meta: {e}")
        else:
            lines.append("[WARN] Meta file not found")
        lines.append("")
        items = list(fruits.keys()) if fruits else []
        prompt = (" . ".join(items) + " .") if items else "(none)"
        lines.append(f"GDINO Items: {items}")
        lines.append(f"GDINO Prompt: {prompt}")
        lines.append(f"Quantities: {fruits}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"

def handle_capture(cam_image, img_upload, video_path, supplier_id=None):
    """Handle image/video capture and processing with session support"""
    if pipe is None:
        return None, "[ERROR] Models not initialized", None
    
    # Create session-specific pipeline
    session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
    
    try:
        previews, metas, saved = [], [], []
        
        # Process webcam
        if cam_image is not None:
            bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
            # Only process if QR decode was successful (has QR items)
            # Debug: Print meta structure
            if meta:
                print(f"DEBUG: meta keys: {list(meta.keys())}")
                if meta.get('qr'):
                    print(f"DEBUG: meta['qr'] keys: {list(meta['qr'].keys())}")
                    if meta['qr'].get('parsed'):
                        print(f"DEBUG: meta['qr']['parsed'] keys: {list(meta['qr']['parsed'].keys())}")
            
            has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
            if vis_bbox is not None and vis_seg is not None and has_qr_items:
                # Add both visualizations with labels - ensure proper format for Gradio Gallery
                previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                saved.append(f"WEBCAM→ {img_path}")
            elif meta and not has_qr_items:
                # QR decode failed - skip this image
                _log_warning("Dataset", "Skipping image: QR decode failed")
        
        # Process single upload
        if img_upload is not None:
            bgr = cv2.cvtColor(img_upload, cv2.COLOR_RGB2BGR)
            vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
            # Only process if QR decode was successful (has QR items)
            has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
            if vis_bbox is not None and vis_seg is not None and has_qr_items:
                # Add both visualizations with labels - ensure proper format for Gradio Gallery
                previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                saved.append(f"UPLOAD→ {img_path}")
            elif meta and not has_qr_items:
                # QR decode failed - skip this image
                _log_warning("Dataset", "Skipping image: QR decode failed")
        
        # Process video
        if video_path:
            video_file_path = _get_path(video_path)
            cap = cv2.VideoCapture(video_file_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(CFG.min_frame_step, int(total // CFG.frames_per_video) if total > 0 else 10)
                idx, grabbed = 0, 0
                
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % step == 0:
                        vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(frame, return_both_visualizations=True)
                        # Only process if QR decode was successful (has QR items)
                        has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
                        if vis_bbox is not None and vis_seg is not None and has_qr_items:
                            # Add both visualizations with labels - ensure proper format for Gradio Gallery
                            previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                            metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                            saved.append(f"VIDEO→ {img_path}")
                            grabbed += 1
                            if grabbed >= CFG.frames_per_video:
                                break
                        elif meta and not has_qr_items:
                            # QR decode failed - skip this frame
                            _log_warning("Dataset", f"Skipping frame {idx}: QR decode failed")
                    idx += 1
                cap.release()
        
        if not previews:
            return None, "[WARN] No valid frames processed", None
        
        # Create ZIP from original dataset (contains all data + meta files)
        # Create temporary directory with original dataset structure
        temp_export_dir = os.path.join(CFG.project_dir, f"temp_export_{session_pipe.ds.session_id}")
        ensure_dir(temp_export_dir)
        
        # Copy original dataset directory (contains images, labels, masks, meta)
        original_dataset_src = session_pipe.ds.root
        original_dataset_dst = os.path.join(temp_export_dir, "dataset")
        if os.path.exists(original_dataset_src):
            shutil.copytree(original_dataset_src, original_dataset_dst, dirs_exist_ok=True)
        
        # Copy registry directory for reference
        registry_src = os.path.join(CFG.project_dir, "registry")
        registry_dst = os.path.join(temp_export_dir, "registry")
        if os.path.exists(registry_src):
            shutil.copytree(registry_src, registry_dst, dirs_exist_ok=True)
        
        # Create ZIP
        zip_name = f"dataset_export_{session_pipe.ds.session_id}"
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, zip_name), 'zip', temp_export_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_export_dir, ignore_errors=True)
        
        return previews, "\n\n".join(metas), zip_path
        
    except Exception as e:
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

def handle_multiple_uploads(images, videos, supplier_id=None):
    """Handle multiple image/video uploads and processing with session support"""
    if pipe is None:
        return None, "[ERROR] Models not initialized", None
    
    # Create session-specific pipeline
    session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
    
    try:
        previews, metas, saved = [], [], []
        total_processed = 0
        total_success = 0
        
        # Process multiple images
        if images:
            _log_info("Multi Upload", f"Processing {len(images)} images...")
            for i, img in enumerate(images):
                try:
                    _log_info("Multi Upload", f"Processing image {i+1}/{len(images)}")
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
                    
                    has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
                    if vis_bbox is not None and vis_seg is not None and has_qr_items:
                        previews.extend([(vis_bbox, f"Image {i+1} - GroundingDINO"), (vis_seg, f"Image {i+1} - White-ring")])
                        metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                        saved.append(f"IMAGE {i+1}→ {img_path}")
                        total_success += 1
                    else:
                        _log_warning("Multi Upload", f"Image {i+1}: QR decode failed or no valid detection")
                        saved.append(f"IMAGE {i+1}→ SKIPPED (QR failed)")
                    
                    total_processed += 1
                except Exception as e:
                    _log_error("Multi Upload", e, f"Failed to process image {i+1}")
                    saved.append(f"IMAGE {i+1}→ ERROR: {str(e)}")
                    total_processed += 1
        
        # Process multiple videos with enhanced video processing
        if videos:
            _log_info("Multi Upload", f"Processing {len(videos)} videos with enhanced video processing...")
            
            # Use the new multi-video processing function
            video_results = process_multiple_videos(videos, CFG)
            
            for video_name, result in video_results["results"].items():
                i = videos.index(result["video_path"]) + 1
                
                if result["success"]:
                    # Add video frames to previews
                    for j, frame_img in enumerate(result["images"]):
                        previews.append((frame_img, f"Video {i} Frame {j+1} - White-ring"))
                    
                    # Add metadata for this video
                    video_meta = {
                        "video_name": video_name,
                        "video_path": result["video_path"],
                        "frame_count": result["frame_count"],
                        "processing_info": result["message"]
                    }
                    metas.append(json.dumps(video_meta, ensure_ascii=False, indent=2))
                    saved.append(f"VIDEO {i} ({video_name})→ {result['frame_count']} frames processed")
                    total_success += 1
                else:
                    _log_warning("Multi Upload", f"Video {i} ({video_name}): {result['message']}")
                    saved.append(f"VIDEO {i} ({video_name})→ ERROR: {result['message']}")
                
                total_processed += 1
        
        # Summary
        summary = f"📊 MULTIPLE UPLOAD SUMMARY:\n"
        summary += f"✅ Total processed: {total_processed}\n"
        summary += f"✅ Total successful: {total_success}\n"
        summary += f"📁 Images: {len(images) if images else 0}\n"
        summary += f"🎬 Videos: {len(videos) if videos else 0}\n"
        
        if videos:
            summary += f"\n🎥 Enhanced Video Processing:\n"
            summary += f"   📊 Processed: {video_results['summary']['processed_videos']}/{video_results['summary']['total_videos']} videos\n"
            summary += f"   🖼️ Total frames: {video_results['summary']['total_frames']}\n"
            summary += f"   ✅ Success rate: {video_results['summary']['success_rate']}\n"
            summary += f"   🔒 Size-lock: {'Enabled' if CFG.video_lock_enable else 'Disabled'}\n"
        
        # FIXED: Create ZIP file from original dataset (contains all data + meta files)
        # Create temporary directory with original dataset structure
        temp_export_dir = os.path.join(CFG.project_dir, f"temp_export_{session_pipe.ds.session_id}")
        ensure_dir(temp_export_dir)
        
        # Copy original dataset directory (contains images, labels, masks, meta)
        original_dataset_src = session_pipe.ds.root
        original_dataset_dst = os.path.join(temp_export_dir, "dataset")
        if os.path.exists(original_dataset_src):
            shutil.copytree(original_dataset_src, original_dataset_dst, dirs_exist_ok=True)
        
        # Copy registry directory for reference
        registry_src = os.path.join(CFG.project_dir, "registry")
        registry_dst = os.path.join(temp_export_dir, "registry")
        if os.path.exists(registry_src):
            shutil.copytree(registry_src, registry_dst, dirs_exist_ok=True)
        
        # Create ZIP
        zip_name = f"dataset_export_{session_pipe.ds.session_id}"
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, zip_name), 'zip', temp_export_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_export_dir, ignore_errors=True)
        
        return previews, json.dumps(metas, ensure_ascii=False, indent=2), zip_path
    except Exception as e:
        _log_error("Multi Upload", e)
        return None, f"[ERROR] {e}", None

def handle_qr_generation(box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count, 
                        fruit_type="", quantity=0, note=""):
    """Generate QR code (id-only payload) and save per-id JSON metadata skeleton"""
    try:
        fruits = {}
        for name, count in [(fruit1_name, fruit1_count), (fruit2_name, fruit2_count)]:
            if name.strip() and count > 0:
                fruits[name.strip()] = int(count)
        
        # Generate QR with metadata
        qr_image, qr_content, meta_file = generate_qr_with_metadata(
            CFG, box_id, fruits, fruit_type, quantity, note
        )
        
        # Save QR image
        qr_filename = f"qr_{box_id.replace('#', '').replace(CFG.box_name_prefix, '')}_{int(time.time())}.png"
        qr_path = os.path.join(CFG.project_dir, qr_filename)
        ensure_dir(CFG.project_dir)
        Image.fromarray(qr_image).save(qr_path)
        
        return qr_image, qr_content, qr_path, meta_file
    except Exception as e:
        return None, f"[ERROR] {e}", None, None

def train_sdy_btn():
    """Train YOLOv8"""
    if pipe is None:
        return "[ERROR] Models not initialized", None
    try:
        # FIXED: Clean up empty dataset folders first
        _log_info("YOLO Training", "Cleaning up empty dataset folders...")
        cleanup_empty_dataset_folders(CFG.project_dir)
        
        # FIXED: Clean dataset BEFORE training to convert class_id = 99 to class_id = 1
        _log_info("YOLO Training", "Cleaning dataset class IDs...")
        clean_dataset_class_ids(CFG.project_dir, old_class_id=99, new_class_id=1)
        
        data_yaml = pipe.ds.write_yaml()
        w, wdir = pipe.train_sdy(data_yaml)
        if not w:
            return "[ERROR] No weights found", None
        
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "sdy_weights"), 'zip', wdir)
        return f"✅ Trained! Weights: {w}", zip_path
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None

def train_u2net_btn():
    """Train U²-Net with ONNX export"""
    if pipe is None:
        return "[ERROR] Models not initialized", None, None
    try:
        best, run_dir, onnx_path = pipe.train_u2net()
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "u2net_weights"), 'zip', run_dir)
        return f"✅ Trained! Best: {best}", zip_path, onnx_path
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None, None

def handle_warehouse_upload(uploaded_image, enable_deskew=False):
    """Handle warehouse upload check with optional deskew"""
    if uploaded_image is None:
        return None, "[ERROR] No image uploaded", None
    
    try:
        frame_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)
        vis_images, log_msg, results = warehouse_check_frame(frame_bgr, enable_deskew)
        return vis_images, log_msg, results
    except Exception as e:
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

