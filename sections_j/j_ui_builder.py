# ========================= SECTION J: UI BUILDER ========================= #

import gradio as gr
import os

# Import dependencies
from sections_a.a_config import CFG
from sections_i.i_config_updates import update_gdino_params
from sections_i.i_dexined import auto_init_dexined, init_dexined_backend, get_system_status
from sections_i.i_handlers import handle_capture, handle_multiple_uploads, handle_qr_generation, handle_warehouse_upload, handle_warehouse_model_upload
from sections_i.i_utils import decode_qr_info
from sections_j.j_ui_components import (
    create_settings_components,
    create_dataset_components,
    create_training_components,
    create_qr_components,
    create_warehouse_components,
    create_image_aligner_components
)
from sections_j.j_ui_utils import _get_path

def build_ui():
    """Build the main Gradio UI with proper tab structure"""
    with gr.Blocks(title="SDY Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # SDY Pipeline - Smart Dataset & Training System
        
        **Complete end-to-end pipeline for dataset creation and model training**
        """)
        
        # Settings Toggle (Global - outside tabs)
        settings_toggle = gr.Button("Advanced Settings", variant="secondary")
        settings_visible = gr.State(False)
        
        # Settings Group (Global - can be toggled from any tab)
        settings_components = create_settings_components()
        (settings_group, auto_init_dexined_btn, init_dexined_btn, system_status_btn,
         dexi_onnx_path, dexi_torch_path, dexi_short_side,
         gdino_prompt, gdino_box_thr, gdino_text_thr, hand_detection_thr,
         use_white_ring, seg_mode, edge_backend, dexi_thr, canny_lo, canny_hi,
         dilate_iters, close_kernel, min_area_ratio, rect_score_min,
         aspect_ratio_min, aspect_ratio_max, erode_inner,
         ring_pair_edge_filter, pair_min_gap, pair_max_gap,
         smooth_close, smooth_open, convex_hull, force_rectify, rectify_padding, rectangle_expansion_factor,
         mode, min_component_area, show_green_frame,
         lock_size_enable, lock_size_long, lock_size_short, lock_size_pad,
         use_gpu, bg_model, feather, update_btn, config_status) = settings_components
        
        # Settings toggle handler
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
                   edge_backend, dexi_thr, canny_lo, canny_hi, dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                   aspect_ratio_min, aspect_ratio_max, erode_inner,
                   ring_pair_edge_filter, pair_min_gap, pair_max_gap,
                   smooth_close, smooth_open, convex_hull, force_rectify, rectify_padding, rectangle_expansion_factor,
                   mode, min_component_area, show_green_frame,
                   lock_size_enable, lock_size_long, lock_size_short, lock_size_pad,
                   use_gpu],
            outputs=[config_status]
        )
        
        # Main Tabs Container
        with gr.Tabs():
            # Tab 1: Create Dataset
            with gr.Tab("Create Dataset"):
                dataset_components = create_dataset_components()
                (init_models_btn, dataset_system_status_btn, init_status,
                 supplier_input, cam, img_upload, vid_upload, run_btn,
                 multi_supplier_input, multi_img_upload, multi_vid_upload, run_multi_btn,
                 gallery, meta_box, ds_zip, multi_gallery, multi_meta_box, multi_ds_zip) = dataset_components
                
                # Model Initialization Event Handlers
                def init_models():
                    try:
                        from sections_i.i_model_init import init_models as init_models_func
                        result = init_models_func()
                        return f"Models initialized: {result}"
                    except Exception as e:
                        return f"Error initializing models: {e}"
                
                def get_dataset_status():
                    try:
                        from sections_i.i_dexined import get_system_status as get_status_func
                        return get_status_func()
                    except Exception as e:
                        return f"Error getting status: {e}"
                
                init_models_btn.click(fn=init_models, inputs=[], outputs=[init_status])
                dataset_system_status_btn.click(fn=get_dataset_status, inputs=[], outputs=[init_status])
                
                # Dataset Event Handlers
                run_btn.click(fn=handle_capture, inputs=[cam, img_upload, vid_upload, supplier_input], outputs=[gallery, meta_box, ds_zip])
                
                # Multiple Upload Handler
                def process_multiple_files(multi_images, multi_videos, supplier_id):
                    """Process multiple uploaded files"""
                    if not multi_images and not multi_videos:
                        return None, "No files uploaded", None
                    
                    # Convert file paths to images for processing
                    images = []
                    videos = []
                    
                    if multi_images:
                        for img_file in multi_images:
                            img_path = _get_path(img_file)
                            if img_path:
                                import cv2
                                img = cv2.imread(img_path)
                                if img is not None:
                                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    if multi_videos:
                        for vid_file in multi_videos:
                            vid_path = _get_path(vid_file)
                            if vid_path:
                                videos.append(vid_path)
                    
                    return handle_multiple_uploads(images, videos, supplier_id)
                
                run_multi_btn.click(
                    fn=process_multiple_files,
                    inputs=[multi_img_upload, multi_vid_upload, multi_supplier_input],
                    outputs=[multi_gallery, multi_meta_box, multi_ds_zip]
                )
            
            # Tab 2: Training
            with gr.Tab("Training"):
                training_components = create_training_components()
                (yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay,
                 yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers, yolo_update_config_btn, yolo_train_btn, sdy_status, sdy_weights, sdy_folder,
                 u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_variant,
                 u2_amp, u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight,
                 u2_update_config_btn, u2_train_btn, u2_status, u2_weights, u2_folder, u2_onnx) = training_components
                
                # Training Event Handlers
                from sections_i.i_training import train_sdy_btn as train_yolo_fn, train_u2net_btn as train_u2_fn
                from sections_i.i_training import update_yolo_config_only, update_u2net_config_only
                
                # YOLO Update Config Button
                yolo_update_config_btn.click(
                    fn=update_yolo_config_only,
                    inputs=[yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay,
                           yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers],
                    outputs=[sdy_status]
                )
                
                # U²-Net Update Config Button
                u2_update_config_btn.click(
                    fn=update_u2net_config_only,
                    inputs=[u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_variant,
                           u2_amp, u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight],
                    outputs=[u2_status]
                )
                
                yolo_train_btn.click(
                    fn=train_yolo_fn,
                    inputs=[yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay,
                           yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers],
                    outputs=[sdy_status, sdy_weights, sdy_folder]
                )
                
                u2_train_btn.click(
                    fn=train_u2_fn,
                    inputs=[u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_variant,
                           u2_amp, u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight],
                    outputs=[u2_status, u2_weights, u2_folder, u2_onnx]
                )
            
            # Tab 3: QR Generator
            with gr.Tab("QR Generator"):
                qr_components = create_qr_components()
                (box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count,
                 fruit_type, quantity, note, generate_qr_btn,
                 qr_image, qr_content, qr_file, qr_meta_file) = qr_components
                
                # QR Generation Event Handlers
                generate_qr_btn.click(
                    fn=handle_qr_generation,
                    inputs=[box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count, fruit_type, quantity, note],
                    outputs=[qr_image, qr_content, qr_file, qr_meta_file]
                )
            
            # Tab 4: Warehouse Check
            with gr.Tab("Warehouse Check"):
                warehouse_components = create_warehouse_components()
                (yolo_model_file, u2net_model_file, upload_models_btn, model_upload_status,
                 warehouse_cam, warehouse_upload, enable_deskew, deskew_method, enable_force_rectangle,
                 check_btn, warehouse_gallery, warehouse_log) = warehouse_components
                
                # Model Upload Event Handler
                def handle_model_upload(yolo_file, u2net_file):
                    """Handle model file uploads"""
                    from sections_i.i_handlers import handle_warehouse_model_upload
                    success, message = handle_warehouse_model_upload(yolo_file, u2net_file)
                    return message
                
                upload_models_btn.click(
                    fn=handle_model_upload,
                    inputs=[yolo_model_file, u2net_model_file],
                    outputs=[model_upload_status]
                )
                
                # Warehouse Check Event Handlers
                def handle_warehouse_check(cam_img, upload_img, deskew_enabled, deskew_method, force_rectangle):
                    """Handle warehouse check with either camera or upload"""
                    input_img = cam_img if cam_img is not None else upload_img
                    if input_img is None:
                        return None, "[ERROR] No image provided"
                    
                    # Use uploaded models if available
                    yolo_path = None
                    u2net_path = None
                    
                    # Try to find latest trained models
                    # Look for latest YOLO model in runs_sdy
                    runs_dir = os.path.join(CFG.project_dir, "runs_sdy")
                    yolo_path = None
                    if os.path.exists(runs_dir):
                        for run_folder in sorted(os.listdir(runs_dir), reverse=True):
                            if run_folder.startswith("train_"):
                                weights_dir = os.path.join(runs_dir, run_folder, "weights")
                                best_pt = os.path.join(weights_dir, "best.pt")
                                if os.path.exists(best_pt):
                                    yolo_path = best_pt
                                    break
                    
                    # Look for latest U2Net model
                    u2net_path = os.path.join(CFG.project_dir, "u2_runs", "u2net_best.pth")
                    
                    if not os.path.exists(yolo_path) or not os.path.exists(u2net_path):
                        return None, "[ERROR] Please upload both YOLO and U²-Net models first"
                    
                    # FIXED: Pass model paths to handler
                    vis_images, log_msg, _ = handle_warehouse_upload(input_img, yolo_path, u2net_path, deskew_enabled, deskew_method, force_rectangle)
                    return vis_images, log_msg
                
                check_btn.click(
                    fn=handle_warehouse_check,
                    inputs=[warehouse_cam, warehouse_upload, enable_deskew, deskew_method, enable_force_rectangle],
                    outputs=[warehouse_gallery, warehouse_log]
                )
            
            # Tab 5: Image Alignment by ID
            with gr.Tab("Image Alignment"):
                aligner_components = create_image_aligner_components()
                (yolo_weight, u2net_weight, dataset_path, output_path, box_id_input,
                 num_images, target_corner, force_square, mask_mode, final_size,
                 scan_btn, align_btn, scan_output, align_gallery, align_log, output_folder) = aligner_components
                
                # Image Alignment Event Handlers
                def handle_scan_dataset(dataset_root):
                    """Scan dataset and return summary of box IDs"""
                    try:
                        from sections_k.k_image_aligner import ImageAligner
                        from pathlib import Path
                        
                        if not dataset_root:
                            return "[ERROR] Please provide dataset path"
                        
                        dataset_path_obj = Path(dataset_root)
                        if not dataset_path_obj.exists():
                            return f"[ERROR] Dataset path does not exist: {dataset_root}"
                        
                        aligner = ImageAligner(dataset_root, "temp_output")
                        summary = aligner.scan_dataset()
                        
                        if not summary:
                            return "[ERROR] No images found in dataset"
                        
                        lines = [
                            f"[SUCCESS] Found {len(summary)} unique box IDs:",
                            ""
                        ]
                        for box_id, count in sorted(summary.items()):
                            lines.append(f"  • {box_id}: {count} images")
                        
                        lines.append("")
                        lines.append(f"[INFO] Total unique IDs: {len(summary)}")
                        lines.append(f"[INFO] You can enter a specific Box ID or use 'ALL' to process all")
                        
                        return "\n".join(lines)
                    
                    except Exception as e:
                        import traceback
                        return f"[ERROR] {e}\n{traceback.format_exc()}"
                
                def handle_align_images(yolo_wt, u2_wt, ds_path, out_path, box_id, 
                                       n_imgs, tgt_corner, f_square, m_mode, f_size):
                    """Handle image alignment"""
                    try:
                        from sections_k.k_image_aligner import align_images_by_id
                        
                        if not ds_path:
                            return None, "[ERROR] Please provide dataset path", ""
                        
                        if not out_path:
                            return None, "[ERROR] Please provide output path", ""
                        
                        if not box_id:
                            return None, "[ERROR] Please provide Box ID or 'ALL'", ""
                        
                        # Store weights if provided (for future use)
                        if yolo_wt:
                            print(f"[INFO] YOLO weight uploaded: {yolo_wt}")
                        if u2_wt:
                            print(f"[INFO] U2-Net weight uploaded: {u2_wt}")
                        
                        # Run alignment
                        aligned_imgs, output_dir, log = align_images_by_id(
                            dataset_root=ds_path,
                            output_root=out_path,
                            box_id=box_id,
                            num_images=int(n_imgs),
                            target_qr_corner=tgt_corner,
                            mask_mode=m_mode,
                            force_square=f_square,
                            final_size=int(f_size)
                        )
                        
                        if not aligned_imgs:
                            return None, log if log else "[ERROR] No images aligned", ""
                        
                        return aligned_imgs, log, output_dir
                    
                    except Exception as e:
                        import traceback
                        error_msg = f"[ERROR] {e}\n{traceback.format_exc()}"
                        return None, error_msg, ""
                
                # Connect handlers
                scan_btn.click(
                    fn=handle_scan_dataset,
                    inputs=[dataset_path],
                    outputs=[scan_output]
                )
                
                align_btn.click(
                    fn=handle_align_images,
                    inputs=[yolo_weight, u2net_weight, dataset_path, output_path, box_id_input,
                           num_images, target_corner, force_square, mask_mode, final_size],
                    outputs=[align_gallery, align_log, output_folder]
                )
        
        gr.Markdown("""
        ---
        **Pipeline Summary**:
        - **Dataset**: GroundingDINO + QR validation + White-ring segmentation → Clean dataset
        - **Training**: YOLOv8 (box detection) + U²-Net (segmentation)
        - **Warehouse**: QR decode + YOLO detect + U²-Net segment → Quality check
        """)
    
    return demo
