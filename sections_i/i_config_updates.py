# ========================= SECTION I: CONFIG UPDATES ========================= #

# Import dependencies
from sections_a.a_config import CFG, _log_info, _log_success, _log_error
from sections_a.a_edges import EDGE

def update_gdino_params(prompt: str, box_thr: float, text_thr: float, hand_detection_thr: float, bg_model: str, feather: int, 
                       use_white_ring: bool, seg_mode: str, edge_backend: str, dexi_thr: float, canny_lo: int, canny_hi: int,
                       dilate_iters: int, close_kernel: int, min_area_ratio: float, 
                       rect_score_min: float, aspect_ratio_min: float, aspect_ratio_max: float, erode_inner: int,
                       ring_pair_edge_filter: bool, pair_min_gap: int, pair_max_gap: int,
                       smooth_close: int, smooth_open: int, convex_hull: bool, force_rectify: str, rectify_padding: int, rectangle_expansion_factor: float,
                       mode: str, min_component_area: int, show_green_frame: bool,
                       lock_size_enable: bool, lock_size_long: int, lock_size_short: int, lock_size_pad: int,
                       use_gpu: bool):
    """Update GroundingDINO, Background Removal, and White-ring params - COPY FROM SEGMENT_GRADIO.py"""
    try:
        CFG.current_prompt = prompt.strip() or CFG.gdino_prompt
        CFG.current_box_thr = max(0.01, min(1.0, box_thr))
        CFG.current_text_thr = max(0.01, min(1.0, text_thr))
        CFG.current_hand_detection_thr = max(0.01, min(1.0, hand_detection_thr))
        
        # White-ring segmentation params (priority)
        CFG.use_white_ring_seg = use_white_ring
        CFG.seg_mode = seg_mode
        
        # Edge Detection Backend - COPY FROM SEGMENT_GRADIO.py
        CFG.edge_backend = edge_backend
        CFG.dexined_threshold = max(0.05, min(0.8, dexi_thr))
        CFG.canny_low = max(0, min(255, canny_lo))
        CFG.canny_high = max(0, min(255, canny_hi))
        
        # Update old config attributes for compatibility
        CFG.canny_lo = CFG.canny_low
        CFG.canny_hi = CFG.canny_high
        CFG.video_backend = edge_backend
        CFG.video_dexi_thr = CFG.dexined_threshold
        
        # Morphology & Filtering - COPY FROM SEGMENT_GRADIO.py
        CFG.dilate_iters = max(0, min(5, dilate_iters))
        CFG.close_kernel = max(3, min(31, close_kernel))
        CFG.min_area_ratio = max(5, min(80, min_area_ratio))
        CFG.rect_score_min = max(0.3, min(0.95, rect_score_min))
        CFG.aspect_ratio_min = max(0.4, min(1.0, aspect_ratio_min))
        CFG.aspect_ratio_max = max(1.0, min(3.0, aspect_ratio_max))
        CFG.erode_inner = max(0, min(10, erode_inner))
        
        # Update old config attributes for compatibility
        CFG.dilate_px = CFG.dilate_iters
        CFG.close_px = CFG.close_kernel
        CFG.ar_min = CFG.aspect_ratio_min
        CFG.ar_max = CFG.aspect_ratio_max
        CFG.erode_inner_px = CFG.erode_inner
        
        # Pair-edge Filter - COPY FROM SEGMENT_GRADIO.py
        CFG.ring_pair_edge_filter = ring_pair_edge_filter
        CFG.pair_min_gap = max(2, min(20, pair_min_gap))
        CFG.pair_max_gap = max(8, min(40, pair_max_gap))
        
        # Smooth & Rectify - COPY FROM SEGMENT_GRADIO.py
        CFG.smooth_close = max(0, min(31, smooth_close))
        CFG.smooth_open = max(0, min(15, smooth_open))
        CFG.convex_hull = convex_hull
        CFG.force_rectify = force_rectify
        CFG.rectify_padding = max(0, min(20, rectify_padding))
        CFG.rectangle_expansion_factor = max(0.5, min(2.0, rectangle_expansion_factor))
        
        # Display Mode - COPY FROM SEGMENT_GRADIO.py
        CFG.display_mode = mode.lower().replace(" ", "_")
        CFG.min_component_area = max(0, min(10000, min_component_area))
        CFG.show_green_frame = show_green_frame
        
        # Update old config attributes for compatibility
        CFG.min_comp_area = CFG.min_component_area
        CFG.rect_pad = CFG.rectify_padding
        CFG.use_convex_hull = CFG.convex_hull
        
        # GPU Settings
        CFG.video_use_gpu = use_gpu
        EDGE.set_gpu_mode(use_gpu)
        
        # Lock Size Settings
        CFG.video_lock_enable = lock_size_enable
        CFG.video_lock_n_warmup = 50  # Default warmup frames
        CFG.video_lock_trim = 0.03    # Default trim factor
        CFG.video_lock_pad = max(0, min(50, lock_size_pad))
        
        # Legacy background removal params (only if white-ring is disabled)
        if not use_white_ring:
            CFG.bg_removal_model = bg_model
            CFG.feather_px = max(0, min(20, feather))
        
        status_msg = f"✅ Config Updated:\n"
        status_msg += f"Prompt: {CFG.current_prompt}\n"
        status_msg += f"Box Threshold: {CFG.current_box_thr}\n"
        status_msg += f"Text Threshold: {CFG.current_text_thr}\n"
        
        if use_white_ring:
            status_msg += f"\n🧠 DexiNed White-ring: ENABLED ({seg_mode})\n"
            status_msg += f"   Backend: {edge_backend}\n"
            status_msg += f"   DexiNed Threshold: {dexi_thr:.2f}\n"
            status_msg += f"   Canny: {canny_lo}-{canny_hi}\n"
            status_msg += f"   GPU: {'ON' if use_gpu else 'OFF'}\n"
            status_msg += f"   Dilate: {dilate_iters}, Close: {close_kernel}\n"
            status_msg += f"   Min Area: {min_area_ratio}%, Rect Score: {rect_score_min:.2f}\n"
            status_msg += f"   Aspect Ratio: {aspect_ratio_min:.1f}-{aspect_ratio_max:.1f}\n"
            status_msg += f"   Erode Inner: {erode_inner}px\n"
            status_msg += f"   Pair Filter: {ring_pair_edge_filter} (gap {pair_min_gap}-{pair_max_gap})\n"
            status_msg += f"   Smooth: close={smooth_close}, open={smooth_open}\n"
            status_msg += f"   Convex Hull: {convex_hull}\n"
            status_msg += f"   Force Rectify: {force_rectify} (pad {rectify_padding}px)\n"
            status_msg += f"   Expansion: {rectangle_expansion_factor:.1f}\n"
            status_msg += f"   Display: {mode}\n"
            status_msg += f"   Min Component: {min_component_area}\n"
            status_msg += f"   Green Frame: {show_green_frame}\n"
            status_msg += f"   Lock Size: {'ON' if lock_size_enable else 'OFF'}"
            if lock_size_enable:
                status_msg += f" (Long: {lock_size_long}px, Short: {lock_size_short}px, Pad: {lock_size_pad}px)"
            status_msg += f"\n🎨 Legacy BG Removal: DISABLED (DexiNed active)"
        else:
            status_msg += f"\n🎨 Legacy BG Removal: {bg_model}\n"
            status_msg += f"🔲 White-ring: DISABLED"
        
        _log_success("Config Update", "All parameters updated successfully")
        return status_msg
        
    except Exception as e:
        _log_error("Config Update", e, "Failed to update configuration")
        return f"❌ Config Update Failed: {str(e)}"

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
        
        # Morphology & Filtering for Video
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
        
        # Smooth & Rectify for Video
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
        
        _log_success("Video Config Update", "All video processing parameters updated successfully")
        return "✅ Video configuration updated successfully!"
        
    except Exception as e:
        _log_error("Video Config Update", e, "Failed to update video configuration")
        return f"❌ Video configuration update failed: {str(e)}"