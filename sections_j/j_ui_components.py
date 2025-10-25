# ========================= SECTION J: UI COMPONENTS ========================= #

import gradio as gr
from sections_a.a_config import CFG
from sections_a.a_edges import EDGE, CUDA_AVAILABLE

def create_settings_components():
    """Create settings components - COPY FROM SEGMENT_GRADIO.py"""
    with gr.Group(visible=False) as settings_group:
        # Model Initialization Section
        gr.Markdown("### Model Initialization")
        with gr.Row():
            auto_init_dexined_btn = gr.Button("Auto-Init DexiNed", variant="primary")
            init_dexined_btn = gr.Button("Manual Init DexiNed", variant="secondary")
            system_status_btn = gr.Button("System Status", variant="secondary")
        
        # DexiNed Settings
        gr.Markdown("#### DexiNed Backend")
        dexi_onnx_path = gr.Textbox("weights/dexined.onnx", label="DexiNed ONNX path")
        dexi_torch_path = gr.Textbox("weights/dexined.pth", label="DexiNed PyTorch path")
        dexi_short_side = gr.Slider(512, 1280, 1024, step=32, label="DexiNed short-side resize")
        
        # GroundingDINO Settings
        gr.Markdown("#### GroundingDINO Settings")
        gdino_prompt = gr.Textbox(CFG.current_prompt, label="Prompt")
        gdino_box_thr = gr.Slider(0.01, 1.0, CFG.current_box_thr, step=0.01, label="Box Threshold")
        gdino_text_thr = gr.Slider(0.01, 1.0, CFG.current_text_thr, step=0.01, label="Text Threshold")
        hand_detection_thr = gr.Slider(0.01, 1.0, CFG.current_hand_detection_thr, step=0.01, label="Hand Detection Threshold")
        
        # White-ring Segmentation Settings
        gr.Markdown("#### White-ring Segmentation Settings")
        use_white_ring = gr.Checkbox(CFG.use_white_ring_seg, label="Enable White-ring Segmentation")
        seg_mode = gr.Radio(["single", "components"], value=CFG.seg_mode, label="Segmentation Mode")
        
        # Edge Detection Settings - COPY FROM SEGMENT_GRADIO.py
        gr.Markdown("#### Edge Detection Settings")
        edge_backend = gr.Radio(["DexiNed","Canny"], value="DexiNed", label="Edge Backend")
        dexi_thr = gr.Slider(0.05, 0.8, 0.42, step=0.01, label="DexiNed threshold")
        canny_lo = gr.Slider(0,255,29, label="Canny low (fallback)")
        canny_hi = gr.Slider(0,255,119, label="Canny high (fallback)")

        # Morphology & filtering - COPY FROM SEGMENT_GRADIO.py
        gr.Markdown("### Morphology & filtering")
        dilate_iters = gr.Slider(0,5,3, label="Dilate iters")
        close_kernel = gr.Slider(3,31,18, label="Close kernel")
        min_area_ratio = gr.Slider(5,80,20, label="Min area ratio (%)")
        rect_score_min = gr.Slider(0.3, 0.95, 0.44, step=0.01, label="Rect score min (area/rect)")
        aspect_ratio_min = gr.Slider(0.4, 1.0, 0.6, step=0.01, label="Aspect ratio min")
        aspect_ratio_max = gr.Slider(1.0, 3.0, 1.8, step=0.01, label="Aspect ratio max")
        erode_inner = gr.Slider(0, 10, 0, label="Erode inner (px)")

        # Pair-edge Filter - COPY FROM SEGMENT_GRADIO.py
        gr.Markdown("### Pair-edge Filter")
        ring_pair_edge_filter = gr.Checkbox(False, label="Ring pair-edge filter")
        pair_min_gap = gr.Slider(2, 20, 4, step=1, label="Pair min gap")
        pair_max_gap = gr.Slider(8, 40, 18, step=1, label="Pair max gap")

        # Smooth & Rectify - COPY FROM SEGMENT_GRADIO.py
        gr.Markdown("### Smooth & Rectify")
        smooth_close = gr.Slider(0,31,26, label="Smooth close")
        smooth_open = gr.Slider(0,15,9, label="Smooth open")
        convex_hull = gr.Checkbox(True, label="Convex hull")
        force_rectify = gr.Radio(["Off","Rectangle","Robust (erode-fit-pad)","Square"], value="Rectangle", label="Force Rectify")
        rectify_padding = gr.Slider(0,20,12, label="Rectify padding (px)")
        rectangle_expansion_factor = gr.Slider(0.5, 2.0, 0.5, step=0.1, label="Rectangle expansion factor")

        # Display Mode - COPY FROM SEGMENT_GRADIO.py
        mode = gr.Radio(["Mask Only","Components Inside"], value="Mask Only", label="Display mode")
        min_component_area = gr.Slider(0, 10000, 0, step=500, label="Min component area")
        show_green_frame = gr.Checkbox(True, label="Show green frame")
        
        # Lock Size Settings
        gr.Markdown("### Lock Size Settings")
        lock_size_enable = gr.Checkbox(False, label="Enable Lock Size")
        lock_size_long = gr.Slider(100, 2000, 800, step=10, label="Lock Long Side (px)")
        lock_size_short = gr.Slider(100, 2000, 600, step=10, label="Lock Short Side (px)")
        lock_size_pad = gr.Slider(0, 50, 10, step=1, label="Lock Padding (px)")
        
        # GPU Settings
        gr.Markdown("#### GPU Settings")
        use_gpu = gr.Checkbox(CFG.video_use_gpu, label="GPU Acceleration")
        
        gr.Markdown("### Legacy Background Removal Config (Disabled when White-ring is ON)")
        bg_model = gr.Dropdown(
             ["u2netp", "u2net", "u2net_human_seg"], 
             value=CFG.bg_removal_model, 
             label="Model",
             info="u2net: U²-Net full | u2netp: U²-Net lite | u2net_human_seg: Human segmentation",
             interactive=False
         )
        feather = gr.Slider(0, 20, CFG.feather_px, label="Feather (px)", interactive=False)
        
        update_btn = gr.Button("Update Config", variant="secondary")
        config_status = gr.Textbox(label="Config Status", interactive=False)
    
    return (settings_group, auto_init_dexined_btn, init_dexined_btn, system_status_btn,
            dexi_onnx_path, dexi_torch_path, dexi_short_side,
            gdino_prompt, gdino_box_thr, gdino_text_thr, hand_detection_thr,
            use_white_ring, seg_mode, edge_backend, dexi_thr, canny_lo, canny_hi,
            dilate_iters, close_kernel, min_area_ratio, rect_score_min,
            aspect_ratio_min, aspect_ratio_max, erode_inner,
            ring_pair_edge_filter, pair_min_gap, pair_max_gap,
            smooth_close, smooth_open, convex_hull, force_rectify, rectify_padding, rectangle_expansion_factor,
            mode, min_component_area, show_green_frame,
            lock_size_enable, lock_size_long, lock_size_short, lock_size_pad,
            use_gpu, bg_model, feather, update_btn, config_status)

def create_dataset_components():
    """Create dataset creation components"""
    with gr.Group():
        gr.Markdown("### Upload Images/Videos to create dataset")
        
        # Model Initialization Section
        gr.Markdown("### Model Initialization")
        with gr.Row():
            init_models_btn = gr.Button("Initialize Models", variant="primary")
            system_status_btn = gr.Button("System Status", variant="secondary")
        init_status = gr.Textbox(label="Initialization Status", interactive=False)
        
        gr.Markdown(f"""
        #### Upload Media Files
        **System Status**: {'GPU Available' if CUDA_AVAILABLE else 'CPU Only'} | **DexiNed**: {'Ready' if EDGE.dexi and EDGE.dexi.available() else 'Not Initialized'}
        """)
        
        with gr.Row():
            # Single Upload
            with gr.Column():
                gr.Markdown("### Single Upload")
                supplier_input = gr.Textbox(label="Supplier/Batch ID", placeholder="e.g., supplier_A, batch_001", info="Optional: for dataset versioning")
                cam = gr.Image(label="Webcam", sources=["webcam"])
                img_upload = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
                vid_upload = gr.File(label="Upload Video", file_types=["video"])
                run_btn = gr.Button("Process Single", variant="primary")
            
            # Multiple Upload
            with gr.Column():
                gr.Markdown("### Multiple Upload (Batch)")
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
                run_multi_btn = gr.Button("Process Multiple", variant="primary", size="lg")
        
        # Output Components
        gallery = gr.Gallery(label="Preview Full Pipeline", columns=2, height=400, show_label=True)
        meta_box = gr.Textbox(label="Metadata", lines=8)
        ds_zip = gr.File(label="Download Dataset (ZIP)")
        
        # Multiple Upload Output Components
        multi_gallery = gr.Gallery(label="Multiple Upload Preview", columns=2, height=400, show_label=True)
        multi_meta_box = gr.Textbox(label="Multiple Upload Metadata", lines=8)
        multi_ds_zip = gr.File(label="Download Multiple Dataset (ZIP)")
    
    return (init_models_btn, system_status_btn, init_status,
            supplier_input, cam, img_upload, vid_upload, run_btn,
            multi_supplier_input, multi_img_upload, multi_vid_upload, run_multi_btn,
            gallery, meta_box, ds_zip, multi_gallery, multi_meta_box, multi_ds_zip)

def create_training_components():
    """Create training components"""
    with gr.Group():
        gr.Markdown("### YOLOv8 Training")
        with gr.Row():
            yolo_epochs = gr.Number(100, label="Epochs")
            yolo_batch = gr.Number(16, label="Batch Size")
            yolo_imgsz = gr.Number(640, label="Image Size")
        with gr.Row():
            yolo_lr0 = gr.Number(0.01, label="Initial LR")
            yolo_lrf = gr.Number(0.1, label="Final LR Factor")
            yolo_weight_decay = gr.Number(0.0005, label="Weight Decay")
        with gr.Row():
            yolo_mosaic = gr.Checkbox(True, label="Mosaic")
            yolo_flip = gr.Checkbox(True, label="Flip")
            yolo_hsv = gr.Checkbox(True, label="HSV Aug")
            yolo_workers = gr.Number(8, label="Workers")
        
        with gr.Row():
            yolo_update_config_btn = gr.Button("Update YOLO Config", variant="secondary")
            yolo_train_btn = gr.Button("Train YOLOv8", variant="primary")
        sdy_status = gr.Textbox(label="YOLO Training Status", lines=5, interactive=False)
        sdy_weights = gr.File(label="Download YOLO Weights")
        sdy_folder = gr.Textbox(label="YOLO Models Folder", interactive=False, visible=False)
        
        gr.Markdown("### U2-Net Training")
        with gr.Row():
            u2_epochs = gr.Number(100, label="Epochs")
            u2_batch = gr.Number(2, label="Batch Size")
            u2_imgsz = gr.Number(320, label="Image Size")
        with gr.Row():
            u2_lr = gr.Number(0.00001, label="Learning Rate")
            u2_optimizer = gr.Dropdown(["Adam", "AdamW", "SGD"], value="AdamW", label="Optimizer")
            u2_loss = gr.Dropdown(["BCEDice", "BCE", "Dice"], value="BCEDice", label="Loss")
            u2_workers = gr.Number(4, label="Workers")
        with gr.Row():
            u2_variant = gr.Dropdown(["u2net", "u2netp", "u2net_lite"], value="u2net", label="U2Net Variant")
        with gr.Row():
            u2_amp = gr.Checkbox(False, label="AMP")
            u2_weight_decay = gr.Number(0.0001, label="Weight Decay")
            u2_use_edge_loss = gr.Checkbox(False, label="Edge Loss")
            u2_edge_loss_weight = gr.Number(0.5, label="Edge Loss Weight")
        
        with gr.Row():
            u2_update_config_btn = gr.Button("Update U2-Net Config", variant="secondary")
            u2_train_btn = gr.Button("Train U2-Net", variant="primary")
        u2_status = gr.Textbox(label="U2-Net Training Status", lines=5, interactive=False)
        u2_weights = gr.File(label="Download U2-Net Weights")
        u2_folder = gr.Textbox(label="U2-Net Models Folder", interactive=False, visible=False)
        u2_onnx = gr.File(label="Download U2-Net ONNX")
    
    return (yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay,
            yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers, yolo_update_config_btn, yolo_train_btn, sdy_status, sdy_weights, sdy_folder,
            u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_variant,
            u2_amp, u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight,
            u2_update_config_btn, u2_train_btn, u2_status, u2_weights, u2_folder, u2_onnx)

def create_qr_components():
    """Create QR generation components"""
    with gr.Group():
        gr.Markdown("### QR Generator")
        box_id = gr.Textbox(label="Box ID", placeholder="Auto-generate if empty")
        
        gr.Markdown("#### Fruit 1")
        with gr.Row():
            fruit1_name = gr.Textbox(label="Fruit Name", value="Orange")
            fruit1_count = gr.Number(label="Count", value=1)
        
        gr.Markdown("#### Fruit 2 (Optional)")
        with gr.Row():
            fruit2_name = gr.Textbox(label="Fruit Name", placeholder="Leave empty if not needed")
            fruit2_count = gr.Number(label="Count", value=0)
        
        gr.Markdown("#### Additional Info")
        fruit_type = gr.Textbox(label="Fruit Type", placeholder="e.g., Citrus")
        quantity = gr.Number(label="Total Quantity", value=1)
        note = gr.Textbox(label="Note", placeholder="Optional note")
        
        generate_qr_btn = gr.Button("Generate QR", variant="primary")
        
        qr_image = gr.Image(label="QR Code", type="numpy")
        qr_content = gr.Textbox(label="QR Content", lines=2, interactive=False)
        qr_file = gr.File(label="Download QR Image")
        qr_meta_file = gr.File(label="Download QR Metadata (JSON)")
    
    return (box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count,
            fruit_type, quantity, note, generate_qr_btn,
            qr_image, qr_content, qr_file, qr_meta_file)

def create_warehouse_components():
    """Create warehouse checking components"""
    with gr.Group():
        gr.Markdown("### Warehouse Checker")
        
        # Model Upload Section
        gr.Markdown("#### 1. Upload Trained Models")
        with gr.Row():
            yolo_model_file = gr.File(label="YOLO Model (.pt)", file_types=[".pt"])
            u2net_model_file = gr.File(label="U²-Net Model (.pth)", file_types=[".pth"])
        
        upload_models_btn = gr.Button("Upload Models", variant="secondary")
        model_upload_status = gr.Textbox(label="Model Upload Status", lines=2, interactive=False)
        
        # Image Input Section
        gr.Markdown("#### 2. Input Image")
        warehouse_cam = gr.Image(label="Webcam", sources=["webcam"], type="numpy")
        warehouse_upload = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
        
        # Processing Settings
        gr.Markdown("#### 3. Processing Settings")
        enable_deskew = gr.Checkbox(label="Enable Deskew", value=False)
        deskew_method = gr.Radio(["minAreaRect", "PCA", "heuristic"], value="minAreaRect", label="Deskew Method")
        enable_force_rectangle = gr.Checkbox(label="Force Rectangular Mask", value=True)
        
        check_btn = gr.Button("Run Warehouse Check", variant="primary")
        
        # Results
        warehouse_gallery = gr.Gallery(label="Results", columns=2, height=400)
        warehouse_log = gr.Textbox(label="Check Log", lines=10, interactive=False)
    
    return (yolo_model_file, u2net_model_file, upload_models_btn, model_upload_status,
            warehouse_cam, warehouse_upload, enable_deskew, deskew_method, enable_force_rectangle,
            check_btn, warehouse_gallery, warehouse_log)

def create_image_aligner_components():
    """Create image aligner components"""
    with gr.Group():
        gr.Markdown("""
        ### Image Alignment by Box ID
        
        Upload trained weights and align images from dataset by box ID.
        This ensures all images from the same box have QR codes in the same position.
        """)
        
        # Weight Upload Section
        gr.Markdown("#### 1. Upload Trained Weights (Optional)")
        with gr.Row():
            yolo_weight = gr.File(label="YOLO Weight (.pt)", file_types=[".pt"])
            u2net_weight = gr.File(label="U2-Net Weight (.pth)", file_types=[".pth"])
        
        # Dataset Selection
        gr.Markdown("#### 2. Select Dataset & Settings")
        dataset_path = gr.Textbox(
            label="Dataset Path",
            value="sdy_project/dataset_sdy_box",
            placeholder="Path to dataset (with images/ and meta/ folders)"
        )
        output_path = gr.Textbox(
            label="Output Path",
            value="sdy_project/aligned_images",
            placeholder="Output folder for aligned images"
        )
        
        with gr.Row():
            box_id_input = gr.Textbox(
                label="Box ID",
                placeholder="Enter specific Box ID or 'ALL' to process all",
                value="ALL",
                info="Use 'ALL' to process all box IDs in dataset"
            )
            num_images = gr.Slider(
                1, 10, 3, step=1,
                label="Number of Images per ID",
                info="How many images to select randomly for each box ID"
            )
        
        with gr.Row():
            target_corner = gr.Radio(
                ["TL", "TR", "BR", "BL"],
                value="BL",
                label="Target QR Corner",
                info="All images will be rotated so QR code is in this corner"
            )
            force_square = gr.Checkbox(
                value=True,
                label="Force Square Output",
                info="Make output images square"
            )
        
        with gr.Row():
            mask_mode = gr.Radio(
                ["square", "polygon"],
                value="square",
                label="Mask Mode"
            )
            final_size = gr.Slider(
                0, 1024, 0, step=32,
                label="Final Resize (0 = no resize)",
                info="Resize aligned images to this size"
            )
        
        # Action Buttons
        gr.Markdown("#### 3. Run Alignment")
        with gr.Row():
            scan_btn = gr.Button("Scan Dataset", variant="secondary")
            align_btn = gr.Button("Align Images", variant="primary", size="lg")
        
        # Output Section
        gr.Markdown("#### 4. Results")
        scan_output = gr.Textbox(
            label="Dataset Scan Results",
            lines=5,
            interactive=False,
            placeholder="Click 'Scan Dataset' to see available box IDs..."
        )
        
        align_gallery = gr.Gallery(
            label="Aligned Images Preview",
            columns=3,
            height=400,
            show_label=True
        )
        
        align_log = gr.Textbox(
            label="Alignment Log",
            lines=10,
            interactive=False
        )
        
        output_folder = gr.Textbox(
            label="Output Folder",
            interactive=False,
            placeholder="Aligned images will be saved here..."
        )
    
    return (yolo_weight, u2net_weight, dataset_path, output_path, box_id_input,
            num_images, target_corner, force_square, mask_mode, final_size,
            scan_btn, align_btn, scan_output, align_gallery, align_log, output_folder)