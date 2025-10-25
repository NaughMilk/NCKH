import cv2
import numpy as np
import os
from typing import List, Dict, Any

def extract_gallery_from_video(video_path, cfg, backend, canny_lo, canny_hi, dexi_thr,
                               dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                               ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                               use_hull, rectify_mode, rect_pad, expand_factor, mode, min_comp_area, show_green_frame,
                               frame_step, max_frames, keep_only_detected=True, use_pair_filter=True, pair_min_gap=4, pair_max_gap=18,
                               lock_enable=False, lock_n_warmup=50, lock_trim=0.1, lock_pad=0):
    """Extract frames from video - simplified without size-lock"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            return [], "❌ Không mở được video file."

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total} frames, {fps}fps, {width}x{height}")
        
        step = max(1, int(frame_step))
        images = []
        kept = 0
        processed = 0
        idx = 0
        
        # Initialize pipeline once for all frames
        from sections_g.g_sdy_core import SDYPipeline
        pipeline = SDYPipeline(cfg)
        
        # Import functions from other modules
        from sections_a.a_edges import process
        from sections_f.f_dataset_utils import cleanup_dataset_files
        
        # Process frames directly without size-lock
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
                
            # Only process frames at step intervals
            if idx % step != 0: 
                idx += 1
                continue

            try:
                # Use existing pipeline to process frame with full pipeline: QR -> GroundingDINO -> White-ring
                result = pipeline.process_frame(frame, preview_only=False, save_dataset=True, return_both_visualizations=True)
                
                if result is None or len(result) < 6:
                    print(f"Frame {idx}: skipped (pipeline failed)")
                    continue
                    
                original_img, gdino_vis, seg_vis, meta, img_path, lab_path = result
                
                processed += 1
                
                # Check if QR was detected and GroundingDINO found objects
                if meta and meta.get("qr_detected", False) and meta.get("gdino_detections", 0) > 0:
                    # Return both GroundingDINO detection and segmentation as separate images
                    images.append(gdino_vis)
                    images.append(seg_vis)
                    kept += 1
                    print(f"Frame {idx}: kept (QR detected, {meta.get('gdino_detections', 0)} objects)")
                else:
                    print(f"Frame {idx}: skipped (no QR or no detections)")
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
                
            if max_frames > 0 and kept >= max_frames: 
                print(f"Reached max_frames limit: {max_frames}")
                break
            idx += 1
        
        cap.release()
        duration = total / fps if fps > 0 else 0
        
        # Cleanup dataset sau khi xử lý hết video
        print("\n🧹 Đang cleanup dataset...")
        cleanup_result = cleanup_dataset_files()
        if cleanup_result["status"] == "clean":
            print("✅ Dataset đã sạch, không cần cleanup")
        elif cleanup_result["status"] == "cleaned":
            print(f"✅ Đã cleanup: {cleanup_result['deleted']} files thừa")
        
        # Build result message
        if processed == 0:
            return [], f"❌ Không xử lý được frame nào. Video: {total} frames, {width}x{height}"
        elif kept == 0 and keep_only_detected:
            return [], f"❌ Không có frame nào có detection. Đã xử lý {processed} frames. Thử tắt 'Chỉ giữ frame có mask'"
        else:
            msg = f"✅ {kept} ảnh từ {processed} frames | tổng {total} frames | step {step} | {duration:.1f}s @ {fps}fps"
            return images, msg
            
    except Exception as e:
        return [], f"❌ Lỗi xử lý video: {str(e)}"

def process_multiple_videos(video_paths: List[str], cfg) -> Dict[str, Any]:
    """
    Xử lý multiple videos với size-lock riêng biệt cho từng video
    Mỗi video sẽ có size-lock object riêng vì mỗi video là một hộp khác nhau
    """
    results = {}
    total_processed = 0
    total_kept = 0
    
    # Import log functions from other modules (will be available after all sections are loaded)
    from sections_a.a_config import _log_info, _log_success, _log_error
    
    _log_info("Multi-Video", f"Bắt đầu xử lý {len(video_paths)} videos...")
    
    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        _log_info("Multi-Video", f"Processing video {i+1}/{len(video_paths)}: {video_name}")
        
        try:
            # Xử lý từng video với size-lock riêng biệt
            images, msg = extract_gallery_from_video(
                video_path, cfg,
                backend=cfg.video_backend,
                canny_lo=cfg.video_canny_lo,
                canny_hi=cfg.video_canny_hi,
                dexi_thr=cfg.video_dexi_thr,
                dilate_iters=cfg.video_dilate_iters,
                close_kernel=cfg.video_close_kernel,
                min_area_ratio=cfg.video_min_area_ratio,
                rect_score_min=cfg.video_rect_score_min,
                ar_min=cfg.video_ar_min,
                ar_max=cfg.video_ar_max,
                erode_inner=cfg.video_erode_inner,
                smooth_close=cfg.video_smooth_close,
                smooth_open=cfg.video_smooth_open,
                use_hull=cfg.video_use_hull,
                rectify_mode=cfg.video_rectify_mode,
                rect_pad=cfg.video_rect_pad,
                expand_factor=cfg.video_expand_factor,
                mode=cfg.video_mode,
                min_comp_area=cfg.video_min_comp_area,
                show_green_frame=cfg.video_show_green_frame,
                frame_step=cfg.video_frame_step,
                max_frames=cfg.video_max_frames,
                keep_only_detected=cfg.video_keep_only_detected,
                use_pair_filter=cfg.video_use_pair_filter,
                pair_min_gap=cfg.video_pair_min_gap,
                pair_max_gap=cfg.video_pair_max_gap,
                lock_enable=cfg.video_lock_enable,
                lock_n_warmup=cfg.video_lock_n_warmup,
                lock_trim=cfg.video_lock_trim,
                lock_pad=cfg.video_lock_pad
            )
            
            # Lưu kết quả cho video này
            results[video_name] = {
                "video_path": video_path,
                "images": images,
                "message": msg,
                "success": len(images) > 0,
                "frame_count": len(images)
            }
            
            total_processed += 1
            total_kept += len(images)
            
            _log_success("Multi-Video", f"Video {video_name}: {len(images)} frames extracted")
            
        except Exception as e:
            error_msg = f"❌ Lỗi xử lý video {video_name}: {str(e)}"
            _log_error("Multi-Video", e, f"Video: {video_name}")
            
            results[video_name] = {
                "video_path": video_path,
                "images": [],
                "message": error_msg,
                "success": False,
                "frame_count": 0,
                "error": str(e)
            }
    
    # Tổng kết
    summary_msg = f"✅ Hoàn thành xử lý {total_processed}/{len(video_paths)} videos | Tổng {total_kept} frames"
    _log_success("Multi-Video", summary_msg)
    
    return {
        "results": results,
        "summary": {
            "total_videos": len(video_paths),
            "processed_videos": total_processed,
            "total_frames": total_kept,
            "success_rate": f"{total_processed/len(video_paths)*100:.1f}%" if video_paths else "0%"
        },
        "message": summary_msg
    }