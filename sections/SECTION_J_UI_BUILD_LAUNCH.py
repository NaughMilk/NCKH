# ========================= SECTION J: UI BUILD & LAUNCH ========================= #
# ========================= SECTION J: UI BUILD & LAUNCH ========================= #

import gradio as gr



import gradio as gr

def _get_path(f):
    """Safe wrapper to get file path from Gradio File component"""
    if f is None: 
        return ""
    return getattr(f, "name", f)  # f.name if object, otherwise f is already path string

def validate_yolo_label(class_id: int, x_center: float, y_center: float, width: float, height: float) -> bool:
    """
    Validate YOLO label values before writing
    Returns True if valid, False if invalid
    """
    # Check class_id is in valid range [0, 1] for 2-class dataset
    if not (0 <= class_id <= 1):
        _log_warning("Label Validation", f"Invalid class_id: {class_id} (must be 0 or 1)")
        return False
    
    # Check bbox values are in [0, 1] range
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        _log_warning("Label Validation", f"Invalid center coordinates: ({x_center:.3f}, {y_center:.3f}) (must be in [0, 1])")
        return False
    
    # Check width and height are positive and reasonable
    if width <= 0 or height <= 0:
        _log_warning("Label Validation", f"Invalid dimensions: width={width:.3f}, height={height:.3f} (must be > 0)")
        return False
    
    # Check for tiny boxes (width/height < 0.01)
    if width < 0.01 or height < 0.01:
        _log_warning("Label Validation", f"Tiny box detected: width={width:.3f}, height={height:.3f} (min size: 0.01)")
        return False
    
    # Check bbox doesn't extend outside image bounds
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    
    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
        _log_warning("Label Validation", f"Bbox extends outside image: ({x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f})")
        return False
    
    return True

def cleanup_empty_dataset_folders(dataset_root: str):
    """
    Clean up empty dataset folders to avoid confusion
    Only keep folders that actually contain images
    """
    import glob
    import shutil
    
    _log_info("Dataset Cleanup", f"Cleaning up empty dataset folders in: {dataset_root}")
    
    # Find all versioned folders
    yolo_folders = glob.glob(os.path.join(dataset_root, "datasets", "yolo", "v*"))
    u2net_folders = glob.glob(os.path.join(dataset_root, "datasets", "u2net", "v*"))
    
    total_removed = 0
    
    # Clean YOLO folders
    for folder in yolo_folders:
        train_images = glob.glob(os.path.join(folder, "images", "train", "*.jpg"))
        if len(train_images) == 0:
            _log_info("Dataset Cleanup", f"Removing empty YOLO folder: {os.path.basename(folder)}")
            shutil.rmtree(folder, ignore_errors=True)
            total_removed += 1
    
    # Clean U²-Net folders
    for folder in u2net_folders:
        train_images = glob.glob(os.path.join(folder, "images", "train", "*.jpg"))
        if len(train_images) == 0:
            _log_info("Dataset Cleanup", f"Removing empty U²-Net folder: {os.path.basename(folder)}")
            shutil.rmtree(folder, ignore_errors=True)
            total_removed += 1
    
    _log_success("Dataset Cleanup", f"Removed {total_removed} empty dataset folders")
    return total_removed

def clean_dataset_class_ids(dataset_root: str, old_class_id: int = 99, new_class_id: int = 1):
    """
    Clean dataset by converting old class_id to new_class_id in all .txt files
    This fixes the issue where class_id = 99 causes Ultralytics to drop all labels
    """
    import glob
    
    _log_info("Dataset Cleaner", f"Cleaning dataset: {old_class_id} -> {new_class_id}")
    
    # Find all .txt files in labels directories - FIXED: Include all possible paths
    label_patterns = [
        # Original dataset paths
        os.path.join(dataset_root, "labels", "train", "*.txt"),
        os.path.join(dataset_root, "labels", "val", "*.txt"),
        # YOLO-specific dataset paths
        os.path.join(dataset_root, "datasets", "yolo", "*", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "datasets", "yolo", "*", "labels", "val", "*.txt"),
        # Legacy paths
        os.path.join(dataset_root, "sdy_project", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "sdy_project", "labels", "val", "*.txt"),
        # Any other nested paths
        os.path.join(dataset_root, "**", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "**", "labels", "val", "*.txt")
    ]
    
    total_files = 0
    total_lines = 0
    converted_lines = 0
    
    for pattern in label_patterns:
        for txt_file in glob.glob(pattern):
            total_files += 1
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    total_lines += 1
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id == old_class_id:
                            # Convert old class_id to new_class_id
                            parts[0] = str(new_class_id)
                            converted_lines += 1
                            _log_info("Dataset Cleaner", f"Converted {old_class_id}->{new_class_id} in {os.path.basename(txt_file)}")
                    new_lines.append(' '.join(parts) + '\n')
                
                # Write back the cleaned file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                    
            except Exception as e:
                _log_error("Dataset Cleaner", f"Error processing {txt_file}: {e}")
    
    _log_success("Dataset Cleaner", f"Cleaned {total_files} files: {converted_lines}/{total_lines} lines converted")
    return total_files, converted_lines

def build_ui():
    with gr.Blocks(title="NCC Pipeline — Dataset → Train → Warehouse Check", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # **NCC Pipeline — Full End-to-End System**
        *Dataset Creation → Model Training → Warehouse Quality Check*
        
        ## 🔄 Pipeline Overview
        1. **NHÀ CUNG CẤP**: Upload ảnh/video → GroundingDINO + QR + Background Removal → Dataset (ZIP)
        2. **TRUNG GIAN**: Train YOLOv8-seg + U²-Net từ dataset → Export weights
        3. **KHO**: Load models → QR decode + YOLO detect + U²-Net segment → Check & Export
        """)
        
        # Init section
        with gr.Row():
            init_btn = gr.Button("🚀 Init Models", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
        init_btn.click(fn=init_models, outputs=init_status)
        
        # Config section
        with gr.Row():
            with gr.Column():
                # Settings Toggle
                settings_visible = gr.State(False)
                settings_toggle = gr.Button("⚙️ Show/Hide Advanced Settings", variant="secondary")
                
                with gr.Group(visible=False) as settings_group:
                    gr.Markdown("### ⚙️ GroundingDINO Config")
                    gdino_prompt = gr.Textbox(label="Prompt", value=CFG.gdino_prompt)
                    
                    gr.Markdown("#### 📊 GroundingDINO Thresholds")
                    with gr.Row():
                        gdino_box_thr = gr.Slider(0.01, 1.0, CFG.gdino_box_thr, label="Box Threshold")
                        gdino_text_thr = gr.Slider(0.01, 1.0, CFG.gdino_text_thr, label="Text Threshold")
                        hand_detection_thr = gr.Slider(0.01, 1.0, CFG.current_hand_detection_thr, label="Hand Detection Threshold")
                    
                    gr.Markdown("### 🧠 DexiNed White-ring Segmentation (Main Backend)")
                    use_white_ring = gr.Checkbox(label="Use DexiNed White-ring Segmentation (Recommended)", value=CFG.use_white_ring_seg)
                    seg_mode = gr.Dropdown(["single", "components"], value=CFG.seg_mode, label="Segmentation Mode")
                    
                    # DexiNed Backend Settings
                    gr.Markdown("#### 🧠 DexiNed Backend")
                    with gr.Row():
                        auto_init_dexined_btn = gr.Button("🚀 Auto-Init DexiNed", variant="primary", size="lg")
                        system_status_btn = gr.Button("📊 System Status", variant="secondary")
                    
                    with gr.Accordion("🔧 Advanced DexiNed Settings", open=False):
                        dexi_onnx_path = gr.Textbox("weights/dexined.onnx", label="DexiNed ONNX path")
                        dexi_torch_path = gr.Textbox("weights/dexined.pth", label="DexiNed PyTorch path")
                        dexi_short_side = gr.Slider(512, 1280, 1024, step=32, label="DexiNed short-side resize")
                        init_dexined_btn = gr.Button("🧠 Manual Init", variant="secondary")
                    
                    # Edge Detection (DexiNed + Canny fallback)
                    gr.Markdown("#### 🎯 Edge Detection")
                    edge_backend = gr.Radio(["DexiNed", "Canny"], value="DexiNed", label="Edge Backend")
                    dexi_thr = gr.Slider(0.05, 0.8, CFG.video_dexi_thr, step=0.01, label="DexiNed Threshold")
                    canny_lo = gr.Slider(10, 100, CFG.canny_lo, label="Canny Low Threshold (Fallback)")
                    canny_hi = gr.Slider(50, 300, CFG.canny_hi, label="Canny High Threshold (Fallback)")
                    
                    # Morphology & Filtering
                    gr.Markdown("#### 🔧 Morphology & Filtering")
                    dilate_px = gr.Slider(0, 5, CFG.dilate_px, label="Dilate Iterations")
                    close_px = gr.Slider(0, 30, CFG.close_px, label="Close Kernel Size")
                    ban_border = gr.Slider(1, 50, CFG.ban_border_px, label="Ban Border Distance (px)")
                    min_area_ratio = gr.Slider(0.1, 0.8, CFG.min_area_ratio, label="Min Area Ratio")
                    rect_score_min = gr.Slider(0.3, 1.0, CFG.rect_score_min, label="Rect Score Min")
                    
                    # Shape Constraints
                    gr.Markdown("#### 📐 Shape Constraints")
                    ar_min = gr.Slider(0.1, 1.0, CFG.ar_min, label="Aspect Ratio Min")
                    ar_max = gr.Slider(1.0, 3.0, CFG.ar_max, label="Aspect Ratio Max")
                    center_cov_min = gr.Slider(0.1, 1.0, CFG.center_cov_min, label="Center Coverage Min")
                    
                    # Final Processing
                    gr.Markdown("#### ✂️ Final Processing")
                    erode_inner = gr.Slider(0, 10, CFG.erode_inner_px, label="Erode Inner (px)")
                    min_comp_area = gr.Slider(500, 20000, CFG.min_comp_area, label="Min Component Area")
                    
                    # Edge Smoothing
                    gr.Markdown("#### 🎨 Edge Smoothing")
                    smooth_mode = gr.Radio(
                        choices=["Off", "Light", "Medium", "Strong"],
                        value=CFG.smooth_mode,
                        label="Smooth Mode"
                    )
                    smooth_iterations = gr.Slider(0, 5, CFG.smooth_iterations, label="Smooth Iterations")
                    gaussian_kernel = gr.Slider(3, 15, CFG.gaussian_kernel, label="Gaussian Kernel")
                    
                    # Post-processing
                    gr.Markdown("#### 🔧 Post-processing")
                    use_shadow_robust_edges = gr.Checkbox(label="Shadow Robust Edges", value=CFG.use_shadow_robust_edges)
                    force_rectify = gr.Radio(
                        choices=["Off", "Square", "Rectangle", "Robust (erode-fit-pad)"],
                        value=CFG.force_rectify,
                        label="Force Rectify"
                    )
                    rect_pad = gr.Slider(0, 20, CFG.rect_pad, label="Rectify Padding (px)")
                    use_convex_hull = gr.Checkbox(label="Use Convex Hull", value=CFG.use_convex_hull)
                    
                    # GPU Settings
                    gr.Markdown("#### 🚀 GPU Settings")
                    use_gpu = gr.Checkbox(CFG.video_use_gpu, label="GPU Acceleration")
                    
                    gr.Markdown("### 🎨 Legacy Background Removal Config (Disabled when White-ring is ON)")
                    bg_model = gr.Dropdown(
                         ["u2netp", "u2net", "u2net_human_seg"], 
                         value=CFG.bg_removal_model, 
                         label="Model",
                         info="u2net: U²-Net full | u2netp: U²-Net lite | u2net_human_seg: Human segmentation",
                         interactive=False
                     )
                    feather = gr.Slider(0, 20, CFG.feather_px, label="Feather (px)", interactive=False)
                    
                    update_btn = gr.Button("🔄 Update Config", variant="secondary")
                    config_status = gr.Textbox(label="Config Status", interactive=False)
                
                # Toggle settings visibility
                def toggle_settings(visible):
                    return gr.update(visible=not visible), not visible
                
                settings_toggle.click(
                    fn=toggle_settings,
                    inputs=[settings_visible],
                    outputs=[settings_group, settings_visible]
                )
                
                # Enable/disable legacy config based on white-ring checkbox
                def toggle_legacy_config(use_white_ring):
                    return gr.update(interactive=not use_white_ring)
                
                use_white_ring.change(
                    fn=toggle_legacy_config,
                    inputs=[use_white_ring],
                    outputs=[bg_model, feather]
                )
                
                # DexiNed Event Handlers
                auto_init_dexined_btn.click(
                    fn=auto_init_dexined,
                    inputs=[],
                    outputs=[config_status]
                )
                
                init_dexined_btn.click(
                    fn=init_dexined_backend,
                    inputs=[dexi_onnx_path, dexi_torch_path, dexi_short_side],
                    outputs=[config_status]
                )
                
                system_status_btn.click(
                    fn=get_system_status,
                    inputs=[],
                    outputs=[config_status]
                )
                
                update_btn.click(
                    fn=update_gdino_params,
                    inputs=[gdino_prompt, gdino_box_thr, gdino_text_thr, hand_detection_thr, bg_model, feather, use_white_ring, seg_mode, 
                           edge_backend, dexi_thr, canny_lo, canny_hi, dilate_px, close_px, ban_border, min_area_ratio, rect_score_min,
                           ar_min, ar_max, center_cov_min, erode_inner, min_comp_area, smooth_mode, 
                           smooth_iterations, gaussian_kernel, use_shadow_robust_edges, force_rectify, 
                           rect_pad, use_convex_hull, use_gpu],
                    outputs=[config_status]
                )
        
        # Tab 1: Dataset Creation
        with gr.Tab("📦 Create Dataset"):
            gr.Markdown("### Upload ảnh/video để tạo dataset")
            
            # Unified Upload Section
            with gr.Group():
                gr.Markdown(f"""
                #### 📁 Upload Media Files
                **🚀 System Status**: {'GPU Available' if CUDA_AVAILABLE else 'CPU Only'} | **🧠 DexiNed**: {'Ready' if EDGE.dexi and EDGE.dexi.available() else 'Not Initialized'}
                """)
                
                with gr.Row():
                    # Single Upload
                    with gr.Column():
                        gr.Markdown("### 📸 Single Upload")
                        supplier_input = gr.Textbox(label="Supplier/Batch ID", placeholder="e.g., supplier_A, batch_001", info="Optional: for dataset versioning")
                        cam = gr.Image(label="Webcam", sources=["webcam"])
                        img_upload = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
                        vid_upload = gr.File(label="Upload Video", file_types=["video"])
                        run_btn = gr.Button("🧰 Process Single", variant="primary")
                    
                    # Multiple Upload
                    with gr.Column():
                        gr.Markdown("### 📁 Multiple Upload (Batch)")
                        multi_supplier_input = gr.Textbox(label="Supplier/Batch ID", placeholder="e.g., supplier_A, batch_001", info="Optional: for dataset versioning")
                        multi_img_upload = gr.File(
                            label="Upload Multiple Images", 
                            file_types=["image"], 
                            file_count="multiple"
                        )
                        multi_vid_upload = gr.File(
                            label="Upload Multiple Videos", 
                            file_types=["video"], 
                            file_count="multiple"
                        )
                        run_multi_btn = gr.Button("🚀 Process Multiple", variant="primary", size="lg")
            
            
            gr.Markdown("""
            ### 📸 Preview Full Pipeline
            **2 ảnh preview sẽ hiển thị (chỉ khi QR decode thành công):**
            1. **GroundingDINO Detection**: Bbox detection của hộp và trái cây
            2. **White-ring Segmentation**: 
               - **🔲 Enhanced White-ring**: Viền trắng + mask container với contour filtering, edge smoothing, force rectify
               - **❌ QR Failed**: Ảnh sẽ bị loại bỏ nếu không đọc được QR code
            
            ### 📁 Multiple Upload Features
            - **Batch Processing**: Upload nhiều ảnh/video cùng lúc
            - **Progress Tracking**: Hiển thị tiến độ xử lý từng file
            - **Error Handling**: Báo lỗi chi tiết cho từng file thất bại
            - **Summary Report**: Tổng kết số lượng thành công/thất bại
            - **Individual Previews**: Preview riêng cho từng file được xử lý thành công
            
            ### 📊 Dataset Configuration
            - **Train/Val Split**: 70%/30% (tăng validation data)
            - **Frames per Video**: 390 frames (tăng 30% từ 300)
            - **Step Size**: 2 (lấy mỗi 2 frame)
            - **Enhanced White-ring Features**:
              - **Contour Area Filtering**: Chỉ giữ contour lớn nhất (loại bỏ fragments)
              - **Edge Smoothing**: Medium mode với 2 iterations, kernel 7
              - **Force Rectify**: Rectangle mode với anti-aliasing
              - **Shadow Robust Edges**: Kháng bóng đổ
              - **Single/Components**: 1 mask toàn bộ hoặc tách components
            - **Legacy U²-Net**: Chỉ active khi White-ring tắt
            """)
            
            # Output Components
            gallery = gr.Gallery(label="Preview Full Pipeline", columns=2, height=400, show_label=True)
            meta_box = gr.Textbox(label="Metadata", lines=8)
            ds_zip = gr.File(label="Download Dataset (ZIP)")
            
            # Multiple Upload Output Components
            multi_gallery = gr.Gallery(label="Multiple Upload Preview", columns=2, height=400, show_label=True)
            multi_meta_box = gr.Textbox(label="Multiple Upload Metadata", lines=8)
            multi_ds_zip = gr.File(label="Download Multiple Dataset (ZIP)")
            
            # Event Handlers
            run_btn.click(fn=handle_capture, inputs=[cam, img_upload, vid_upload, supplier_input], outputs=[gallery, meta_box, ds_zip])
            
            # Multiple Upload Handler
            def process_multiple_files(multi_images, multi_videos, supplier_id):
                """Process multiple uploaded files"""
                if not multi_images and not multi_videos:
                    return None, "No files uploaded", None
                
                # Convert file paths to images for processing
                images = []
                videos = []
                
                # Process multiple images
                if multi_images:
                    for img_file in multi_images:
                        try:
                            img_path = _get_path(img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                images.append(img_rgb)
                        except Exception as e:
                            _log_error("Multi Upload", e, f"Failed to load image: {img_path}")
                
                # Process multiple videos (just store paths)
                if multi_videos:
                    videos = [_get_path(vid_file) for vid_file in multi_videos]
                
                return handle_multiple_uploads(images, videos, supplier_id)
            
            run_multi_btn.click(
                fn=process_multiple_files,
                inputs=[multi_img_upload, multi_vid_upload, multi_supplier_input],
                outputs=[multi_gallery, multi_meta_box, multi_ds_zip]
            )
            
            
            
        
        # (Removed advanced QR Generator tab; keeping only Simple)
        
        # Tab 3: Real-time Camera Processing
        with gr.Tab("📹 Real-time Camera"):
            gr.Markdown(f"""
            ### 📹 Real-time Camera Processing
            **Tính năng:**
            - **Real-time processing** với camera
            - **DexiNed/Canny edge detection** với auto-download
            - **GPU acceleration** nếu có
            - **Live parameter adjustment**
            - **4 Rectify Modes**: Off, Rectangle, Robust (erode-fit-pad), Square
            - **Pair-edge Filter**: Advanced edge filtering
            
            **🚀 System Status**: {'GPU Available' if CUDA_AVAILABLE else 'CPU Only'} | **🧠 DexiNed**: {'Ready' if EDGE.dexi and EDGE.dexi.available() else 'Not Initialized'}
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Camera Controls
                    gr.Markdown("### 📹 Camera Controls")
                    camera_input = gr.Image(label="Camera Input", streaming=True)
                    
                    # Processing Parameters
                    gr.Markdown("### ⚙️ Processing Parameters")
                    cam_backend = gr.Radio(["DexiNed", "Canny"], value=CFG.video_backend, label="Edge Backend")
                    cam_dexi_thr = gr.Slider(0.05, 0.8, CFG.video_dexi_thr, step=0.01, label="DexiNed Threshold")
                    cam_canny_lo = gr.Slider(0, 255, CFG.video_canny_lo, step=1, label="Canny Low")
                    cam_canny_hi = gr.Slider(0, 255, CFG.video_canny_hi, step=1, label="Canny High")
                    
                    # Morphology
                    cam_dilate_iters = gr.Slider(0, 5, CFG.video_dilate_iters, step=1, label="Dilate Iterations")
                    cam_close_kernel = gr.Slider(3, 31, CFG.video_close_kernel, step=2, label="Close Kernel")
                    cam_min_area_ratio = gr.Slider(5, 80, CFG.video_min_area_ratio, step=5, label="Min Area Ratio (%)")
                    cam_rect_score_min = gr.Slider(0.3, 0.95, CFG.video_rect_score_min, step=0.05, label="Rect Score Min")
                    
                    # Shape Filtering
                    cam_ar_min = gr.Slider(0.4, 1.0, CFG.video_ar_min, step=0.1, label="AR Min")
                    cam_ar_max = gr.Slider(1.0, 3.0, CFG.video_ar_max, step=0.1, label="AR Max")
                    cam_erode_inner = gr.Slider(0, 10, CFG.video_erode_inner, step=1, label="Erode Inner (px)")
                    
                    # Smoothing
                    cam_smooth_close = gr.Slider(0, 31, CFG.video_smooth_close, step=1, label="Smooth Close")
                    cam_smooth_open = gr.Slider(0, 15, CFG.video_smooth_open, step=1, label="Smooth Open")
                    cam_use_hull = gr.Checkbox(CFG.video_use_hull, label="Use Convex Hull")
                    
                    # Rectification
                    cam_rectify_mode = gr.Radio(["Off", "Rectangle", "Robust (erode-fit-pad)", "Square"], 
                                               value=CFG.video_rectify_mode, label="Rectify Mode")
                    cam_rect_pad = gr.Slider(0, 20, CFG.video_rect_pad, step=1, label="Rectify Padding (px)")
                    cam_expand_factor = gr.Slider(0.5, 2.0, CFG.video_expand_factor, step=0.1, label="Expand Factor")
                    
                    # Display
                    cam_mode = gr.Radio(["Mask Only", "Components Inside"], value=CFG.video_mode, label="Display Mode")
                    cam_min_comp_area = gr.Slider(0, 10000, CFG.video_min_comp_area, step=500, label="Min Component Area")
                    cam_show_green_frame = gr.Checkbox(CFG.video_show_green_frame, label="Show Green Frame")
                    
                    # Pair-edge Filter
                    cam_use_pair_filter = gr.Checkbox(CFG.video_use_pair_filter, label="Use Pair-edge Filter")
                    cam_pair_min_gap = gr.Slider(2, 20, CFG.video_pair_min_gap, step=1, label="Pair Min Gap (px)")
                    cam_pair_max_gap = gr.Slider(8, 40, CFG.video_pair_max_gap, step=1, label="Pair Max Gap (px)")
                    
                    # GPU
                    cam_use_gpu = gr.Checkbox(CFG.video_use_gpu, label="GPU Acceleration")
                    
                    # DexiNed Auto-Init for Camera
                    gr.Markdown("### 🧠 DexiNed Setup")
                    cam_auto_init_btn = gr.Button("🚀 Auto-Init DexiNed", variant="primary")
                    
                    # Control Buttons
                    start_camera_btn = gr.Button("📹 Start Camera Processing", variant="primary", size="lg")
                    stop_camera_btn = gr.Button("⏹️ Stop Camera", variant="stop")
                    
                    # Status
                    camera_status = gr.Textbox(label="Camera Status", lines=2, interactive=False)
                
                with gr.Column(scale=2):
                    # Camera Output
                    gr.Markdown("### 📊 Camera Output")
                    camera_output = gr.Image(label="Processed Camera Feed", height=600)
                    camera_info = gr.Textbox(label="Processing Info", lines=3, interactive=False)
            
            # Camera processing function
            def process_camera_live(frame, backend, dexi_thr, canny_lo, canny_hi,
                                  dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                                  ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                                  rectify_mode, rect_pad, expand_factor, mode, min_comp_area,
                                  show_green_frame, use_pair_filter, pair_min_gap, pair_max_gap, use_gpu):
                """Process camera frame in real-time"""
                if frame is None:
                    return None, "No camera input"
                
                try:
                    # Set GPU mode
                    EDGE.set_gpu_mode(use_gpu)
                    
                    # Process frame
                    processed_frame = process_camera_frame(
                        frame, backend, canny_lo, canny_hi, dexi_thr,
                        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                        ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                        rectify_mode, rect_pad, min_comp_area, mode, show_green_frame, expand_factor,
                        use_pair_filter, pair_min_gap, pair_max_gap, None
                    )
                    
                    if processed_frame is not None:
                        gpu_info = "[GPU]" if EDGE.use_gpu else "[CPU]"
                        info = f"Backend: {backend} | GPU: {gpu_info} | Real-time processing active"
                        return processed_frame, info
                    else:
                        return None, "Processing failed"
                        
                except Exception as e:
                    return None, f"Error: {str(e)}"
            
            # Event handlers
            cam_auto_init_btn.click(
                fn=auto_init_dexined,
                inputs=[],
                outputs=[camera_status]
            )
            
            # Event handlers for real-time processing
            camera_input.change(
                fn=process_camera_live,
                inputs=[
                    camera_input, cam_backend, cam_dexi_thr, cam_canny_lo, cam_canny_hi,
                    cam_dilate_iters, cam_close_kernel, cam_min_area_ratio, cam_rect_score_min,
                    cam_ar_min, cam_ar_max, cam_erode_inner, cam_smooth_close, cam_smooth_open,
                    cam_use_hull, cam_rectify_mode, cam_rect_pad, cam_expand_factor, cam_mode,
                    cam_min_comp_area, cam_show_green_frame, cam_use_pair_filter, cam_pair_min_gap,
                    cam_pair_max_gap, cam_use_gpu
                ],
                outputs=[camera_output, camera_info]
            )
        
        # Tab 4: QR Generator (Simple)
        with gr.Tab("🎯 QR Generator (Simple)"):
            gr.Markdown("### Đơn giản: Box ID (tùy chọn), Tên trái cây, Số lượng → QR id-only")
            
            with gr.Row():
                with gr.Column():
                    box_id_input_s = gr.Textbox(label="Box ID", placeholder="Auto-generate if empty")
                    fruit_name_s = gr.Textbox(label="Fruit Name", value="Orange")
                    quantity_s = gr.Number(label="Quantity", value=1)
                    
                    generate_qr_btn_s = gr.Button("🎯 Generate QR + Save Metadata", variant="primary")
                
                with gr.Column():
                    qr_image = gr.Image(label="QR Code", type="numpy")
                    qr_content = gr.Textbox(label="QR ID (payload)", lines=2)
                    json_path_simple = gr.Textbox(label="JSON Path (edit this file)", lines=2)
                    download_qr = gr.File(label="Download QR")
            
            with gr.Row():
                gr.Markdown("### 🔎 Decode QR (Upload ảnh QR để xem log)")
                qr_upload = gr.File(label="Upload QR Image", file_types=["image"])
                qr_decode_log = gr.Textbox(label="QR Decode Log", lines=12, interactive=False)
            
            def _wrap_handle_qr_generation(box_id, fruit_name_in, qty, dummy2=None, dummy3=None, dummy4=None, dummy5=None, dummy6=None):
                # Keep signature flexible; map simple inputs to original handler
                return handle_qr_generation(box_id, fruit_name_in, int(qty or 0), "", 0, "", int(qty or 0), "")

            # Main tab button wiring removed (only Simple tab retained)

            # Simple tab button wiring
            generate_qr_btn_s.click(
                fn=lambda b, f, q: _wrap_handle_qr_generation(b, f, q),
                inputs=[box_id_input_s, fruit_name_s, quantity_s],
                outputs=[qr_image, qr_content, json_path_simple, download_qr]
            )

            # Decode uploaded QR and show GDINO-related info
            qr_upload.change(
                fn=decode_qr_info,
                inputs=[qr_upload],
                outputs=[qr_decode_log]
            )
        
        # Tab 3: Train SDY (YOLOv8)
        with gr.Tab("🏋️ Train YOLOv8"):
            gr.Markdown("### Train YOLOv8 model (SDY) với Hyperparameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ Training Parameters")
                    yolo_epochs = gr.Slider(10, 200, CFG.yolo_epochs, step=10, label="Epochs", info="Số vòng lặp đào tạo")
                    yolo_batch = gr.Slider(1, 32, CFG.yolo_batch, step=1, label="Batch Size", info="Số ảnh mỗi bước")
                    yolo_imgsz = gr.Slider(320, 1280, CFG.yolo_imgsz, step=32, label="Image Size", info="Kích thước ảnh huấn luyện")
                    
                    gr.Markdown("#### 📈 Learning Rate")
                    yolo_lr0 = gr.Slider(0.001, 0.1, CFG.yolo_lr0, step=0.001, label="Initial LR", info="Tốc độ học ban đầu")
                    yolo_lrf = gr.Slider(0.001, 0.1, CFG.yolo_lrf, step=0.001, label="Final LR", info="Tốc độ học cuối")
                    yolo_weight_decay = gr.Slider(0.0001, 0.01, CFG.yolo_weight_decay, step=0.0001, label="Weight Decay", info="Hệ số suy giảm trọng số")
                    
                    gr.Markdown("#### 🔄 Augmentation")
                    yolo_mosaic = gr.Checkbox(CFG.yolo_mosaic, label="Mosaic", info="Ghép 4 ảnh thành 1")
                    yolo_flip = gr.Checkbox(CFG.yolo_flip, label="Horizontal Flip", info="Lật ngang ảnh")
                    yolo_hsv = gr.Checkbox(CFG.yolo_hsv, label="HSV Augmentation", info="Thay đổi màu sắc")
                    
                    gr.Markdown("#### ⚡ Performance")
                    yolo_workers = gr.Slider(1, 16, CFG.yolo_workers, step=1, label="Workers", info="Số luồng xử lý dữ liệu")
                
                with gr.Column():
                    train_sdy = gr.Button("🏋️ Train YOLOv8", variant="primary", size="lg")
                    sdy_log = gr.Textbox(label="Training Log", lines=8)
                    sdy_zip = gr.File(label="Download Weights (ZIP)")
            
            # Update config button for YOLO
            update_yolo_btn = gr.Button("🔄 Update YOLO Config", variant="secondary")
            yolo_config_status = gr.Textbox(label="Config Status", interactive=False)
            
            train_sdy.click(fn=train_sdy_btn, outputs=[sdy_log, sdy_zip])
            
            # Update YOLO config function
            def update_yolo_config(epochs, batch, imgsz, lr0, lrf, weight_decay, mosaic, flip, hsv, workers):
                try:
                    CFG.yolo_epochs = int(epochs)
                    CFG.yolo_batch = int(batch)
                    CFG.yolo_imgsz = int(imgsz)
                    CFG.yolo_lr0 = float(lr0)
                    CFG.yolo_lrf = float(lrf)
                    CFG.yolo_weight_decay = float(weight_decay)
                    CFG.yolo_mosaic = bool(mosaic)
                    CFG.yolo_flip = bool(flip)
                    CFG.yolo_hsv = bool(hsv)
                    CFG.yolo_workers = int(workers)
                    
                    return f"✅ YOLO Config Updated:\nEpochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}\nLR0: {lr0}, LRF: {lrf}, WeightDecay: {weight_decay}\nMosaic: {mosaic}, Flip: {flip}, HSV: {hsv}\nWorkers: {workers}"
                except Exception as e:
                    return f"❌ Error updating YOLO config: {str(e)}"
            
            update_yolo_btn.click(
                fn=update_yolo_config,
                inputs=[yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay, 
                       yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers],
                outputs=[yolo_config_status]
            )
        
        # Tab 5: Train U²-Net
        with gr.Tab("🎓 Train U²-Net"):
            gr.Markdown("### Train U²-Net for Background Removal (from scratch) với Hyperparameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ Training Parameters")
                    u2_epochs = gr.Slider(10, 200, CFG.u2_epochs, step=10, label="Epochs", info="Số vòng lặp đào tạo")
                    u2_batch = gr.Slider(1, 32, CFG.u2_batch, step=1, label="Batch Size", info="Số ảnh mỗi bước")
                    u2_imgsz = gr.Slider(256, 512, CFG.u2_imgsz, step=32, label="Image Size", info="Kích thước ảnh huấn luyện")
                    
                    gr.Markdown("#### 📈 Learning & Optimization")
                    u2_lr = gr.Slider(0.0001, 0.01, CFG.u2_lr, step=0.0001, label="Learning Rate", info="Tốc độ học")
                    u2_optimizer = gr.Dropdown(["AdamW", "SGD"], value=CFG.u2_optimizer, label="Optimizer", info="Thuật toán tối ưu")
                    u2_loss = gr.Dropdown(["BCEDice", "BCE", "Dice"], value=CFG.u2_loss, label="Loss Function", info="Hàm mất mát")
                    
                    gr.Markdown("#### ⚡ Performance")
                    u2_workers = gr.Slider(1, 8, CFG.u2_workers, step=1, label="Workers", info="Số luồng xử lý dữ liệu")
                    u2_amp = gr.Checkbox(CFG.u2_amp, label="Mixed Precision", info="Sử dụng AMP để tăng tốc")
                    
                    gr.Markdown("#### 🎯 Advanced Settings")
                    u2_weight_decay = gr.Slider(0.0001, 0.01, CFG.u2_weight_decay, step=0.0001, label="Weight Decay", info="Hệ số suy giảm trọng số")
                    u2_use_edge_loss = gr.Checkbox(CFG.u2_use_edge_loss, label="Edge Loss", info="Sử dụng edge loss để cải thiện boundary")
                    u2_edge_loss_weight = gr.Slider(0.01, 0.5, CFG.u2_edge_loss_weight, step=0.01, label="Edge Loss Weight", info="Trọng số cho edge loss")
                
                with gr.Column():
                    train_u2 = gr.Button("🏋️ Train U²-Net", variant="primary", size="lg")
                    u2_log = gr.Textbox(label="Training Log", lines=8)
                    
                    with gr.Row():
                        u2_zip = gr.File(label="Download Weights (ZIP)")
                        u2_onnx = gr.File(label="Download ONNX")
            
            # Update config button for U²-Net
            update_u2_btn = gr.Button("🔄 Update U²-Net Config", variant="secondary")
            u2_config_status = gr.Textbox(label="Config Status", interactive=False)
            
            train_u2.click(fn=train_u2net_btn, outputs=[u2_log, u2_zip, u2_onnx])
            
            # Update U²-Net config function
            def update_u2net_config(epochs, batch, imgsz, lr, optimizer, loss, workers, amp, weight_decay, use_edge_loss, edge_loss_weight):
                try:
                    CFG.u2_epochs = int(epochs)
                    CFG.u2_batch = int(batch)
                    CFG.u2_imgsz = int(imgsz)
                    CFG.u2_lr = float(lr)
                    CFG.u2_optimizer = str(optimizer)
                    CFG.u2_loss = str(loss)
                    CFG.u2_workers = int(workers)
                    CFG.u2_amp = bool(amp)
                    CFG.u2_weight_decay = float(weight_decay)
                    CFG.u2_use_edge_loss = bool(use_edge_loss)
                    CFG.u2_edge_loss_weight = float(edge_loss_weight)
                    
                    return f"✅ U²-Net Config Updated:\nEpochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}\nLR: {lr}, Optimizer: {optimizer}, Loss: {loss}\nWorkers: {workers}, AMP: {amp}\nWeightDecay: {weight_decay}, EdgeLoss: {use_edge_loss}, EdgeWeight: {edge_loss_weight}"
                except Exception as e:
                    return f"❌ Error updating U²-Net config: {str(e)}"
            
            update_u2_btn.click(
                fn=update_u2net_config,
                inputs=[u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_amp, 
                       u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight],
                outputs=[u2_config_status]
            )
        
        # Tab 6: WAREHOUSE CHECK (MỚI)
        with gr.Tab("🏭 Kho – Kiểm tra & Lọc"):
            gr.Markdown("""
            ### 🏭 Kiểm tra Hàng Tại Kho
            **Pipeline**: QR decode → YOLO detect (box + fruits) → **QR-YOLO Validation** → U²-Net segment → Check & Export
            
            **Quy trình chi tiết**:
            1. **QR Decode**: Đọc thông tin box và danh sách trái cây từ QR code
            2. **YOLO Detection**: Phát hiện box và các trái cây trong ảnh
            3. **🆕 QR-YOLO Validation**: So sánh số lượng trái cây từ QR với YOLO detection
               - ✅ **Pass**: Số lượng khớp (cho phép sai lệch ±20%) → Tiếp tục U²-Net
               - ❌ **Fail**: Số lượng không khớp → Bỏ qua U²-Net, hiển thị lỗi
            4. **U²-Net Segmentation**: Chỉ chạy khi validation passed, segment box region
            5. **Deskew** (tùy chọn): Xoay thẳng hộp nghiêng
            6. **Export**: Xuất kết quả detection + segmentation
            
            **Hướng dẫn**:
            1. Load YOLO model (best.pt từ tab Train YOLOv8)
            2. Load U²-Net model (best.pth từ tab Train U²-Net)
            3. Upload ảnh hoặc chụp từ webcam
            4. Xem kết quả validation + detection + segmentation
            5. Export kết quả nếu cần
            """)
            
            # Model loading section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Load YOLO Model")
                    yolo_upload = gr.File(label="Upload YOLO weights (.pt)", file_types=[".pt"])
                    load_yolo_btn = gr.Button("🔄 Load YOLO", variant="secondary")
                    yolo_status = gr.Textbox(label="YOLO Status", lines=2, interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 🎨 Load U²-Net Model")
                    u2net_upload = gr.File(label="Upload U²-Net weights (.pth)", file_types=[".pth"])
                    load_u2net_btn = gr.Button("🔄 Load U²-Net", variant="secondary")
                    u2net_status = gr.Textbox(label="U²-Net Status", lines=2, interactive=False)
            
            # Input section
            gr.Markdown("### 📷 Input")
            with gr.Row():
                warehouse_cam = gr.Image(label="Webcam", type="numpy")
                warehouse_upload = gr.Image(label="Upload Image", type="numpy")
            
            # Deskew options
            gr.Markdown("### 🔄 Processing Options")
            with gr.Row():
                enable_deskew = gr.Checkbox(CFG.enable_deskew, label="Deskew Box (remove BG + rotate 90° align)", 
                                          info="Tự động xoay thẳng hộp nghiêng")
                deskew_method = gr.Dropdown(["minAreaRect", "PCA", "heuristic"], value=CFG.deskew_method, 
                                          label="Deskew Method", info="Phương pháp tính góc xoay")
            
            check_btn = gr.Button("🔍 Run Warehouse Check", variant="primary", size="lg")
            
            # Output section
            warehouse_gallery = gr.Gallery(label="Results", columns=2, height=400)
            warehouse_log = gr.Textbox(label="Check Log", lines=10, interactive=False)
            
            # Event handlers
            load_yolo_btn.click(
                fn=lambda f: load_warehouse_yolo(_get_path(f))[1],
                inputs=[yolo_upload],
                outputs=[yolo_status]
            )
            
            load_u2net_btn.click(
                fn=lambda f: load_warehouse_u2net(_get_path(f))[1],
                inputs=[u2net_upload],
                outputs=[u2net_status]
            )
            
            def handle_warehouse_check(cam_img, upload_img, deskew_enabled, deskew_meth):
                # Update global config
                CFG.enable_deskew = deskew_enabled
                CFG.deskew_method = deskew_meth
                
                # Prioritize upload over cam
                input_img = upload_img if upload_img is not None else cam_img
                if input_img is None:
                    return None, "[ERROR] No image provided"
                
                # FIXED: Only return first 2 values, ignore results
                vis_images, log_msg, _ = handle_warehouse_upload(input_img, deskew_enabled)
                return vis_images, log_msg
            
            check_btn.click(
                fn=handle_warehouse_check,
                inputs=[warehouse_cam, warehouse_upload, enable_deskew, deskew_method],
                outputs=[warehouse_gallery, warehouse_log]
            )
        
        gr.Markdown("""
        ---
        **Pipeline Summary**:
        - **Dataset**: GroundingDINO + QR validation + Background Removal → Clean dataset (images + masks)
        - **Training**: YOLOv8 (box detection) + U²-Net (segmentation from scratch)
        - **Warehouse**: QR decode + YOLO detect + U²-Net segment → Quality check
        
        **Key Features**:
        - ✅ Full end-to-end pipeline
        - ✅ Train from scratch (no pretrained weights for U²-Net)
        - ✅ Warehouse quality control with dual models
        - ✅ Export all results (dataset ZIP, weights ZIP, check results)
        """)
    
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(server_name="127.0.0.1", server_port=7860)

