import cv2
import numpy as np
import os
from typing import List, Dict, Any

def extract_gallery_from_video(video_path, cfg, backend, canny_lo, canny_hi, dexi_thr,
                               dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                               ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                               use_hull, rectify_mode, rect_pad, expand_factor, mode, min_comp_area, show_green_frame,
                               frame_step, max_frames, keep_only_detected=True, use_pair_filter=True, pair_min_gap=4, pair_max_gap=18,
                               lock_enable=True, lock_n_warmup=50, lock_trim=0.1, lock_pad=0):
    """Extract frames from video with size-lock pre-pass and reprocess"""
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
        
        # Size-lock variables
        size_lock = None
        samples = []
        raw_frames_warmup = []
        warmup_processed = 0
        valid_detections = 0
        
        # Import functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_geometry import fit_rect_core, robust_avg_box
            from sections.SECTION_A_CONFIG_UTILS import process
            
            # Phase 1: Pre-pass for size-lock (if enabled)
            if lock_enable:
                print(f"🔍 Size-lock pre-pass: collecting {lock_n_warmup} valid detections...")
                
                while valid_detections < lock_n_warmup:
                    ret, frame = cap.read()
                    if not ret: 
                        break
                        
                    # Only process frames at step intervals
                    if idx % step != 0: 
                        idx += 1
                        continue
                    
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Get detection parameters using fit_rect_core
                        cx, cy, w, h, ang, mask, poly_core = fit_rect_core(
                            rgb, backend, canny_lo, canny_hi, dexi_thr,
                            dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                            ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                            use_hull, use_pair_filter, pair_min_gap, pair_max_gap
                        )
                        
                        # Store raw frame for reprocessing (regardless of detection)
                        raw_frames_warmup.append((idx, rgb))
                        warmup_processed += 1
                        
                        if cx is not None and w > 5 and h > 5:  # Valid detection
                            # Normalize to (long, short)
                            long_side = max(w, h)
                            short_side = min(w, h)
                            samples.append((long_side, short_side))
                            valid_detections += 1
                            print(f"Valid detection {valid_detections}/{lock_n_warmup}: frame {idx}, size {long_side:.1f}x{short_side:.1f}")
                        else:
                            print(f"Frame {idx}: no valid detection (skipped)")
                        
                    except Exception as e:
                        print(f"Error in warmup frame {idx}: {e}")
                        
                    idx += 1
                
                # Calculate locked size
                print(f"📊 Collected {len(samples)} valid detections from {warmup_processed} processed frames")
                if len(samples) >= 5:
                    long_avg, short_avg, n_used = robust_avg_box(samples, lock_trim)
                    if long_avg and short_avg:
                        size_lock = {
                            "enabled": True,
                            "long": long_avg,
                            "short": short_avg,
                            "pad": lock_pad
                        }
                        print(f"🔒 Size locked: {long_avg:.1f}x{short_avg:.1f}px from {n_used} samples")
                    else:
                        print("⚠️ Size-lock failed: invalid averages")
                else:
                    print(f"⚠️ Size-lock disabled: insufficient valid detections ({len(samples)} < 5)")
                    print(f"   Processed {warmup_processed} frames but only {len(samples)} had valid detections")
            
            # Phase 2: Reprocess all frames (including warmup frames)
            print("🔄 Reprocessing all frames with locked size...")
            
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            
            # Process all frames
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                # Only process frames at step intervals
                if idx % step != 0: 
                    idx += 1
                    continue
                
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with size-lock if available
                    out, edges, info = process(
                        rgb, backend, canny_lo, canny_hi, dexi_thr,
                        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                        ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                        rectify_mode, rect_pad, min_comp_area, mode, show_green_frame, expand_factor,
                        use_pair_filter, pair_min_gap, pair_max_gap, size_lock
                    )
                    
                    processed += 1
                    
                    if out is not None:
                        if keep_only_detected:
                            # Check if detection was successful (has green frame)
                            if show_green_frame and "comps=" in info:
                                # Simple heuristic: if we have components info, likely detected
                                images.append(out)
                                kept += 1
                            elif not show_green_frame:
                                # If not showing green frame, keep all processed frames
                                images.append(out)
                                kept += 1
                        else:
                            # Keep all frames regardless of detection
                            images.append(out)
                            kept += 1
                    
                    # Limit max frames
                    if max_frames > 0 and kept >= max_frames:
                        print(f"Reached max frames limit: {max_frames}")
                        break
                        
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    
                idx += 1
            
            cap.release()
            
            msg = f"✅ Extracted {kept} frames from {processed} processed frames"
            if size_lock:
                msg += f" (size-locked: {size_lock['long']:.1f}x{size_lock['short']:.1f}px)"
            
            return images, msg
            
        except ImportError:
            # Fallback if other functions not available yet
            cap.release()
            return [], "❌ Video processing functions not available"
            
    except Exception as e:
        return [], f"❌ Error processing video: {str(e)}"

def process_multiple_videos(video_paths: List[str], cfg) -> Dict[str, Any]:
    """
    Xử lý multiple videos với size-lock riêng biệt cho từng video
    Mỗi video sẽ có size-lock object riêng vì mỗi video là một hộp khác nhau
    """
    results = {}
    total_processed = 0
    total_kept = 0
    
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
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
        
    except ImportError:
        # Fallback if log functions not available yet
        return {
            "results": {},
            "summary": {
                "total_videos": len(video_paths),
                "processed_videos": 0,
                "total_frames": 0,
                "success_rate": "0%"
            },
            "message": "❌ Video processing functions not available"
        }
