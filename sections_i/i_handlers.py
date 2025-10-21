# ========================= SECTION I: HANDLERS ========================= #

import os
import cv2
import numpy as np
import json
import time
import traceback
import shutil
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

# Import dependencies
from sections_a.a_config import CFG, _log_info, _log_success, _log_error, _log_warning
from sections_a.a_utils import ensure_dir
from sections_a.a_video import process_multiple_videos
from sections_e.e_qr_detection import QR
from sections_e.e_qr_utils import parse_qr_payload
from sections_e.e_qr_generation import generate_qr_with_metadata
from sections_g.g_sdy_core import SDYPipeline

def _get_path(f):
    """Safe wrapper to get file path from Gradio File component"""
    if f is None: 
        return ""
    return getattr(f, "name", f)  # f.name if object, otherwise f is already path string
from sections_h.h_warehouse_core import warehouse_check_frame
from sections_i.i_model_init import pipe

def handle_capture(cam_image, img_upload, video_path, supplier_id=None):
    """Handle image/video capture and processing with session support"""
    _log_info("Dataset", f"Starting handle_capture with supplier_id={supplier_id}")
    
    # Debug: Check pipe import
    from sections_i.i_model_init import pipe
    _log_info("Dataset", f"Imported pipe from i_model_init: pipe={pipe is not None}")
    
    if pipe is None:
        _log_error("Dataset", "Models not initialized", "Please click 'Initialize Models' first")
        return None, "[ERROR] Models not initialized. Please click 'Initialize Models' first.", None
    
    try:
        # Create session-specific pipeline
        _log_info("Dataset", "Creating SDYPipeline instance...")
        session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
        _log_success("Dataset", "SDYPipeline created successfully")
        
        previews, metas, saved = [], [], []
        
        # Process webcam
        if cam_image is not None:
            _log_info("Dataset", "Processing webcam image...")
            try:
                bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                _log_info("Dataset", "Calling session_pipe.process_frame for webcam...")
                vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
                
                # Debug: Print meta structure
                if meta:
                    _log_info("Dataset", f"Webcam meta keys: {list(meta.keys())}")
                    if meta.get('qr'):
                        _log_info("Dataset", f"Webcam QR keys: {list(meta['qr'].keys())}")
                        if meta['qr'].get('parsed'):
                            _log_info("Dataset", f"Webcam QR parsed keys: {list(meta['qr']['parsed'].keys())}")
                
                has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('fruits')
                if vis_bbox is not None and vis_seg is not None and has_qr_items:
                    # Add both visualizations with labels - ensure proper format for Gradio Gallery
                    previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                    metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                    saved.append(f"WEBCAM→ {img_path}")
                    _log_success("Dataset", f"Webcam processed successfully: {img_path}")
                elif meta and not has_qr_items:
                    # QR decode failed - skip this image
                    _log_warning("Dataset", "Skipping webcam image: QR decode failed")
                else:
                    _log_warning("Dataset", f"Webcam processing failed: vis_bbox={vis_bbox is not None}, vis_seg={vis_seg is not None}, meta={meta is not None}")
            except Exception as e:
                _log_error("Dataset", f"Error processing webcam image: {e}", traceback.format_exc())
        
        # Process single upload
        if img_upload is not None:
            _log_info("Dataset", "Processing uploaded image...")
            try:
                bgr = cv2.cvtColor(img_upload, cv2.COLOR_RGB2BGR)
                _log_info("Dataset", "Calling session_pipe.process_frame for upload...")
                vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
                
                # Debug: Print meta structure
                if meta:
                    _log_info("Dataset", f"Upload meta keys: {list(meta.keys())}")
                    if meta.get('qr'):
                        _log_info("Dataset", f"Upload QR keys: {list(meta['qr'].keys())}")
                        if meta['qr'].get('parsed'):
                            _log_info("Dataset", f"Upload QR parsed keys: {list(meta['qr']['parsed'].keys())}")
                
                has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('fruits')
                has_qr_data = meta and meta.get('qr') and meta['qr'].get('data')
                
                if vis_bbox is not None and vis_seg is not None and (has_qr_items or has_qr_data):
                    # Add both visualizations with labels - ensure proper format for Gradio Gallery
                    previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                    metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                    saved.append(f"UPLOAD→ {img_path}")
                    _log_success("Dataset", f"Upload processed successfully: {img_path}")
                elif meta and not has_qr_data:
                    # QR decode failed - skip this image
                    _log_warning("Dataset", "Skipping uploaded image: QR decode failed")
                else:
                    _log_warning("Dataset", f"Upload processing failed: vis_bbox={vis_bbox is not None}, vis_seg={vis_seg is not None}, meta={meta is not None}")
            except Exception as e:
                _log_error("Dataset", f"Error processing uploaded image: {e}", traceback.format_exc())
        
        # Process video
        if video_path:
            _log_info("Dataset", f"Processing video: {video_path}")
            try:
                video_file_path = _get_path(video_path)
                _log_info("Dataset", f"Video file path: {video_file_path}")
                
                cap = cv2.VideoCapture(video_file_path)
                if cap.isOpened():
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    _log_info("Dataset", f"Video info: {total} frames, {fps}fps, {width}x{height}")
                    
                    step = max(CFG.min_frame_step, int(total // CFG.frames_per_video) if total > 0 else 10)
                    _log_info("Dataset", f"Processing every {step} frames, max {CFG.frames_per_video} frames")
                    
                    idx, grabbed = 0, 0
                    
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            _log_info("Dataset", f"End of video reached at frame {idx}")
                            break
                        
                        # Log every frame read (not just processed ones)
                        if idx % 10 == 0:  # Log every 10th frame
                            _log_info("Dataset", f"Reading video frame {idx}...")
                        
                        if idx % step == 0:
                            _log_info("Dataset", f"Processing video frame {idx} (step={step})...")
                            try:
                                result = session_pipe.process_frame(frame, return_both_visualizations=True)
                                
                                # Handle None return (frame rejected)
                                if result is None or not isinstance(result, tuple):
                                    _log_warning("Dataset", f"Frame {idx} was rejected or returned None")
                                    continue
                                
                                vis_bbox, vis_seg, meta, img_path, lab_path = result
                                
                                # Debug: Print meta structure for video frames
                                if meta:
                                    _log_info("Dataset", f"Video frame {idx} meta keys: {list(meta.keys())}")
                                    if meta.get('qr'):
                                        _log_info("Dataset", f"Video frame {idx} QR keys: {list(meta['qr'].keys())}")
                                        if meta['qr'].get('parsed'):
                                            _log_info("Dataset", f"Video frame {idx} QR parsed keys: {list(meta['qr']['parsed'].keys())}")
                                
                                # Only process if QR decode was successful (has QR items)
                                has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('fruits')
                                _log_info("Dataset", f"Video frame {idx} QR check: has_qr_items={has_qr_items}")
                                
                                if vis_bbox is not None and vis_seg is not None and has_qr_items:
                                    # Add both visualizations with labels - ensure proper format for Gradio Gallery
                                    previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                                    metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                                    saved.append(f"VIDEO→ {img_path}")
                                    grabbed += 1
                                    _log_success("Dataset", f"Video frame {idx} processed successfully: {img_path}")
                                    if grabbed >= CFG.frames_per_video:
                                        _log_info("Dataset", f"Reached max frames limit: {CFG.frames_per_video}")
                                        break
                                elif meta and not has_qr_items:
                                    # QR decode failed - skip this frame
                                    _log_warning("Dataset", f"Skipping video frame {idx}: QR decode failed")
                                else:
                                    _log_warning("Dataset", f"Video frame {idx} processing failed: vis_bbox={vis_bbox is not None}, vis_seg={vis_seg is not None}, meta={meta is not None}")
                            except Exception as e:
                                _log_error("Dataset", f"Error processing video frame {idx}: {e}", traceback.format_exc())
                        idx += 1
                    cap.release()
                    _log_success("Dataset", f"Video processing completed: {grabbed} frames processed")
                else:
                    _log_error("Dataset", f"Failed to open video file: {video_file_path}")
            except Exception as e:
                _log_error("Dataset", f"Error processing video: {e}", traceback.format_exc())
        
        if not previews:
            _log_warning("Dataset", "No valid frames processed")
            return None, "[WARN] No valid frames processed", None
        
        _log_info("Dataset", f"Processing completed: {len(previews)} previews, {len(metas)} metadata entries")
        
        # Debug: Print metadata content
        if metas:
            _log_info("Dataset", f"First metadata entry: {metas[0][:200]}...")
        
        # Create ZIP from original dataset (contains all data + meta files)
        _log_info("Dataset", "Creating dataset ZIP export...")
        try:
            # Create temporary directory with original dataset structure
            temp_export_dir = os.path.join(CFG.project_dir, f"temp_export_{session_pipe.ds.session_id}")
            ensure_dir(temp_export_dir)
            _log_info("Dataset", f"Temp export dir: {temp_export_dir}")
            
            # Copy original dataset directory (contains images, labels, masks, meta)
            original_dataset_src = session_pipe.ds.root
            original_dataset_dst = os.path.join(temp_export_dir, "dataset")
            _log_info("Dataset", f"Copying dataset from {original_dataset_src} to {original_dataset_dst}")
            
            if os.path.exists(original_dataset_src):
                shutil.copytree(original_dataset_src, original_dataset_dst, dirs_exist_ok=True)
                _log_success("Dataset", "Dataset directory copied successfully")
            else:
                _log_warning("Dataset", f"Original dataset source not found: {original_dataset_src}")
            
            # Copy registry directory for reference
            registry_src = os.path.join(CFG.project_dir, "registry")
            registry_dst = os.path.join(temp_export_dir, "registry")
            _log_info("Dataset", f"Copying registry from {registry_src} to {registry_dst}")
            
            if os.path.exists(registry_src):
                shutil.copytree(registry_src, registry_dst, dirs_exist_ok=True)
                _log_success("Dataset", "Registry directory copied successfully")
            else:
                _log_warning("Dataset", f"Registry source not found: {registry_src}")
            
            # Create ZIP
            zip_name = f"dataset_export_{session_pipe.ds.session_id}"
            zip_path = shutil.make_archive(os.path.join(CFG.project_dir, zip_name), 'zip', temp_export_dir)
            _log_success("Dataset", f"ZIP created successfully: {zip_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_export_dir, ignore_errors=True)
            _log_info("Dataset", "Temporary directory cleaned up")
            
            _log_success("Dataset", f"Dataset processing completed successfully: {len(previews)} images processed")
            return previews, "\n\n".join(metas), zip_path
            
        except Exception as e:
            _log_error("Dataset", f"Error creating dataset ZIP: {e}", traceback.format_exc())
            return previews, "\n\n".join(metas), None
        
    except Exception as e:
        _log_error("Dataset", f"Critical error in handle_capture: {e}", traceback.format_exc())
        return None, f"[ERROR] Critical error: {e}", None

def handle_multiple_uploads(images, videos, supplier_id=None):
    """Handle multiple image/video uploads and processing with session support"""
    _log_info("Multi Upload", f"Starting handle_multiple_uploads with supplier_id={supplier_id}")
    
    # Debug: Check pipe import
    from sections_i.i_model_init import pipe
    _log_info("Multi Upload", f"Imported pipe from i_model_init: pipe={pipe is not None}")
    
    if pipe is None:
        _log_error("Multi Upload", "Models not initialized", "Please click 'Initialize Models' first")
        return None, "[ERROR] Models not initialized. Please click 'Initialize Models' first.", None
    
    try:
        # Create session-specific pipeline
        _log_info("Multi Upload", "Creating SDYPipeline instance...")
        session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
        _log_success("Multi Upload", "SDYPipeline created successfully")
        
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
                    
                    has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('fruits')
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
        
        # Debug: Print return values
        metadata_json = json.dumps(metas, ensure_ascii=False, indent=2)
        _log_info("Dataset", f"Returning: {len(previews)} previews, metadata length: {len(metadata_json)} chars")
        
        return previews, metadata_json, zip_path
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
        
        # Extract qr_id from meta_file path to get the folder and filename
        qr_folder = os.path.dirname(meta_file)
        qr_id = os.path.basename(qr_folder)  # Get QR ID from folder name
        qr_filename = f"{qr_id}.png"  # Use QR ID as filename
        qr_path = os.path.join(qr_folder, qr_filename)
        Image.fromarray(qr_image).save(qr_path)
        
        return qr_image, qr_content, qr_path, meta_file
    except Exception as e:
        return None, f"[ERROR] {e}", None, None

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
