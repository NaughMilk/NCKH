#!/usr/bin/env python3
"""
White Ring Segmentation - Gradio Demo
=====================================
Demo ứng dụng White Ring Segmentation với logging chi tiết cho báo cáo.

Mục tiêu: Tách chính xác vùng "hộp chứa" bằng pipeline dựa vào biên trắng quanh thành hộp.
"""

import gradio as gr
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhiteRingSegmenter:
    """White Ring Segmentation với logging chi tiết"""
    
    def __init__(self):
        self.edge_backend = "canny"  # canny, dexined
        self.canny_low = 50
        self.canny_high = 150
        self.dexined_threshold = 0.1
        self.min_gap = 4
        self.max_gap = 18
        self.min_area_ratio = 0.1
        self.rect_score_min = 0.7
        self.aspect_ratio_min = 0.3
        self.aspect_ratio_max = 3.0
        self.erode_inner = 3
        self.convex_hull = False
        self.rectify_mode = "robust"  # square, rectangle, robust
        self.rect_pad = 8
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Tiền xử lý ảnh BGR với chuẩn hoá độ tương phản-chi tiết"""
        start_time = time.time()
        
        # Chuẩn hoá độ tương phản
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge lại
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        processing_time = time.time() - start_time
        
        log_info = {
            "step": "Tiền xử lý ảnh",
            "input_shape": image.shape,
            "enhanced_shape": enhanced.shape,
            "processing_time": f"{processing_time:.3f}s",
            "contrast_enhancement": "CLAHE applied",
            "conversion": "BGR → LAB → Enhanced → Grayscale"
        }
        
        logger.info(f"✅ {log_info['step']}: {log_info['processing_time']}")
        return gray, log_info
    
    def detect_edges(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Phát hiện biên với backend linh hoạt (Canny/DexiNed)"""
        start_time = time.time()
        
        if self.edge_backend == "canny":
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            backend_info = f"Canny (low={self.canny_low}, high={self.canny_high})"
        else:
            # DexiNed fallback (simplified)
            edges = cv2.Canny(gray, 50, 150)
            backend_info = "DexiNed (fallback to Canny)"
        
        processing_time = time.time() - start_time
        edge_pixels = np.count_nonzero(edges)
        
        log_info = {
            "step": "Phát hiện biên",
            "backend": self.edge_backend,
            "backend_info": backend_info,
            "edge_pixels": edge_pixels,
            "edge_density": f"{edge_pixels / (edges.shape[0] * edges.shape[1]) * 100:.2f}%",
            "processing_time": f"{processing_time:.3f}s",
            "edge_shape": edges.shape
        }
        
        logger.info(f"✅ {log_info['step']}: {edge_pixels} pixels, {log_info['edge_density']}")
        return edges, log_info
    
    def filter_paired_edges(self, edges: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Lọc cặp biên và dựng mặt nạ vòng"""
        start_time = time.time()
        
        # Dilation để nối các cạnh gần nhau
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Morphological operations để làm mịn
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours theo diện tích
        image_area = edges.shape[0] * edges.shape[1]
        min_area = image_area * self.min_area_ratio
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Kiểm tra aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                if self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max:
                    # Tính rect score
                    rect_area = w * h
                    rect_score = area / rect_area if rect_area > 0 else 0
                    
                    if rect_score >= self.rect_score_min:
                        filtered_contours.append(contour)
        
        processing_time = time.time() - start_time
        
        log_info = {
            "step": "Lọc cặp biên",
            "total_contours": len(contours),
            "filtered_contours": len(filtered_contours),
            "min_area_ratio": self.min_area_ratio,
            "min_area_pixels": int(min_area),
            "aspect_ratio_range": f"{self.aspect_ratio_min}-{self.aspect_ratio_max}",
            "rect_score_min": self.rect_score_min,
            "processing_time": f"{processing_time:.3f}s"
        }
        
        logger.info(f"✅ {log_info['step']}: {len(filtered_contours)}/{len(contours)} contours passed")
        return filtered_contours, log_info
    
    def create_mask(self, contours: list, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Tạo mặt nạ từ contours và giữ thành phần lớn nhất"""
        start_time = time.time()
        
        if not contours:
            mask = np.zeros(image_shape, dtype=np.uint8)
            log_info = {
                "step": "Tạo mặt nạ",
                "mask_created": False,
                "reason": "No valid contours",
                "processing_time": f"{time.time() - start_time:.3f}s"
            }
            logger.warning(f"⚠️ {log_info['step']}: {log_info['reason']}")
            return mask, log_info
        
        # Tìm contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # Tạo mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Erode inner nếu cần
        if self.erode_inner > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erode_inner, self.erode_inner))
            mask = cv2.erode(mask, kernel, iterations=1)
        
        # Convex hull nếu cần
        if self.convex_hull:
            hull = cv2.convexHull(largest_contour)
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [hull], 255)
        
        # Kiểm tra mask size
        mask_area = np.count_nonzero(mask)
        image_area = image_shape[0] * image_shape[1]
        mask_ratio = mask_area / image_area
        
        processing_time = time.time() - start_time
        
        log_info = {
            "step": "Tạo mặt nạ",
            "largest_contour_area": int(largest_area),
            "mask_area": mask_area,
            "mask_ratio": f"{mask_ratio:.2%}",
            "erode_inner": self.erode_inner,
            "convex_hull": self.convex_hull,
            "mask_valid": mask_ratio < 0.8,  # Tránh mask quá lớn
            "processing_time": f"{processing_time:.3f}s"
        }
        
        logger.info(f"✅ {log_info['step']}: {mask_area} pixels, {log_info['mask_ratio']}")
        return mask, log_info
    
    def rectify_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ép hình chữ nhật và kiểm soát sai lệch"""
        start_time = time.time()
        
        # Tìm contours trong mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            log_info = {
                "step": "Ép hình chữ nhật",
                "rectify_mode": self.rectify_mode,
                "rectified": False,
                "reason": "No contours in mask",
                "processing_time": f"{time.time() - start_time:.3f}s"
            }
            logger.warning(f"⚠️ {log_info['step']}: {log_info['reason']}")
            return mask, log_info
        
        # Lấy contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        if self.rectify_mode == "robust":
            # Robust mode: erode-fit-pad
            # Erode để giảm ảnh hưởng ngoại lai
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(mask, kernel, iterations=1)
            
            # Tìm minAreaRect
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            # Tạo mask từ rectangle
            rect_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(rect_mask, [box], 255)
            
            # Thêm padding
            if self.rect_pad > 0:
                kernel_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                      (self.rect_pad*2+1, self.rect_pad*2+1))
                rect_mask = cv2.dilate(rect_mask, kernel_pad, iterations=1)
            
            final_mask = rect_mask
            
        elif self.rectify_mode == "square":
            # Square mode
            x, y, w, h = cv2.boundingRect(largest_contour)
            size = max(w, h)
            center_x, center_y = x + w//2, y + h//2
            x1 = max(0, center_x - size//2)
            y1 = max(0, center_y - size//2)
            x2 = min(mask.shape[1], x1 + size)
            y2 = min(mask.shape[0], y1 + size)
            
            final_mask = np.zeros(mask.shape, dtype=np.uint8)
            final_mask[y1:y2, x1:x2] = 255
            
        else:  # rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            final_mask = np.zeros(mask.shape, dtype=np.uint8)
            final_mask[y:y+h, x:x+w] = 255
        
        processing_time = time.time() - start_time
        
        log_info = {
            "step": "Ép hình chữ nhật",
            "rectify_mode": self.rectify_mode,
            "rect_pad": self.rect_pad,
            "original_area": np.count_nonzero(mask),
            "rectified_area": np.count_nonzero(final_mask),
            "area_change": f"{np.count_nonzero(final_mask) / np.count_nonzero(mask) * 100:.1f}%" if np.count_nonzero(mask) > 0 else "0%",
            "processing_time": f"{processing_time:.3f}s"
        }
        
        logger.info(f"✅ {log_info['step']}: {log_info['rectify_mode']} mode, {log_info['area_change']}")
        return final_mask, log_info
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """Pipeline chính: xử lý ảnh và tạo mask"""
        total_start = time.time()
        
        # Step 1: Tiền xử lý
        gray, preprocess_log = self.preprocess_image(image)
        
        # Step 2: Phát hiện biên
        edges, edge_log = self.detect_edges(gray)
        
        # Step 3: Lọc cặp biên
        contours, filter_log = self.filter_paired_edges(edges)
        
        # Step 4: Tạo mask
        mask, mask_log = self.create_mask(contours, gray.shape)
        
        # Step 5: Rectify mask
        final_mask, rectify_log = self.rectify_mask(mask)
        
        total_time = time.time() - total_start
        
        # Tạo log chi tiết với ghi chú từng ảnh
        detailed_log = f"""
🔍 WHITE RING SEGMENTATION PIPELINE - BÁO CÁO CHI TIẾT
{'='*80}

📊 TỔNG QUAN HỆ THỐNG:
• Thời gian xử lý tổng: {total_time:.3f}s
• Kích thước ảnh đầu vào: {image.shape[1]}x{image.shape[0]} pixels
• Backend phát hiện biên: {self.edge_backend.upper()}
• Chế độ ép hình: {self.rectify_mode.upper()}

🔄 PIPELINE XỬ LÝ - ẢNH ĐẦU VÀO → BIÊN → LỌC CẶP BIÊN → MASK LỚN NHẤT → ÉP HÌNH

1️⃣ ẢNH ĐẦU VÀO (INPUT IMAGE):
   📸 Thông tin ảnh gốc:
   • Định dạng: BGR Color Image
   • Kích thước: {image.shape[1]}x{image.shape[0]} pixels
   • Số kênh màu: {image.shape[2]}
   • Kiểu dữ liệu: {image.dtype}
   • Giá trị min-max: {image.min()}-{image.max()}

2️⃣ TIỀN XỬ LÝ ẢNH (PREPROCESSING):
   🎨 Chuẩn hoá độ tương phản-chi tiết:
   • Phương pháp: CLAHE (Contrast Limited Adaptive Histogram Equalization)
   • Chuyển đổi không gian màu: BGR → LAB → Enhanced → Grayscale
   • Kích thước tile grid: 8x8 pixels
   • Clip limit: 2.0
   • Thời gian xử lý: {preprocess_log['processing_time']}
   • Kích thước ảnh sau xử lý: {preprocess_log['enhanced_shape'][1]}x{preprocess_log['enhanced_shape'][0]}

3️⃣ PHÁT HIỆN BIÊN (EDGE DETECTION):
   🔍 Backend: {edge_log['backend_info']}
   • Số pixel biên được phát hiện: {edge_log['edge_pixels']:,} pixels
   • Mật độ biên trong ảnh: {edge_log['edge_density']}
   • Tỷ lệ biên/ảnh: {edge_log['edge_pixels']/(edge_log['edge_shape'][0]*edge_log['edge_shape'][1])*100:.2f}%
   • Kích thước ảnh biên: {edge_log['edge_shape'][1]}x{edge_log['edge_shape'][0]}
   • Thời gian xử lý: {edge_log['processing_time']}

4️⃣ LỌC CẶP BIÊN (PAIRED EDGES FILTERING):
   🔗 Morphological operations + Contour filtering:
   • Tổng số contours phát hiện: {filter_log['total_contours']}
   • Contours vượt qua bộ lọc: {filter_log['filtered_contours']}
   • Tỷ lệ lọc thành công: {filter_log['filtered_contours']/filter_log['total_contours']*100:.1f}%
   • Diện tích tối thiểu yêu cầu: {filter_log['min_area_ratio']:.1%} của ảnh
   • Số pixel diện tích tối thiểu: {filter_log['min_area_pixels']:,} pixels
   • Khoảng tỷ lệ khung hình cho phép: {filter_log['aspect_ratio_range']}
   • Điểm số hình chữ nhật tối thiểu: {filter_log['rect_score_min']:.1f}
   • Thời gian xử lý: {filter_log['processing_time']}

5️⃣ TẠO MẶT NĂNG (MASK CREATION):
   🎭 Giữ contour lớn nhất và tạo mask:
   • Diện tích contour lớn nhất: {mask_log['largest_contour_area']:,} pixels
   • Diện tích mask cuối cùng: {mask_log['mask_area']:,} pixels
   • Tỷ lệ mask so với ảnh: {mask_log['mask_ratio']}
   • Số pixel mask: {mask_log['mask_area']:,} / {image.shape[0]*image.shape[1]:,} pixels
   • Ăn mòn phía trong: {mask_log['erode_inner']} pixels
   • Sử dụng Convex Hull: {'Có' if mask_log['convex_hull'] else 'Không'}
   • Mask hợp lệ: {'✅ Có' if mask_log['mask_valid'] else '❌ Không (quá lớn)'}
   • Thời gian xử lý: {mask_log['processing_time']}

6️⃣ ÉP HÌNH CHỮ NHẬT (RECTIFICATION):
   📐 Chế độ {rectify_log['rectify_mode'].upper()}:
   • Padding thêm: {rectify_log['rect_pad']} pixels
   • Diện tích mask gốc: {rectify_log['original_area']:,} pixels
   • Diện tích mask sau ép: {rectify_log['rectified_area']:,} pixels
   • Thay đổi diện tích: {rectify_log['area_change']}
   • Số pixel thay đổi: {rectify_log['rectified_area'] - rectify_log['original_area']:,} pixels
   • Thời gian xử lý: {rectify_log['processing_time']}

📈 KẾT QUẢ CUỐI CÙNG:
• ✅ Pipeline hoàn thành thành công
• 🎯 Tỷ lệ contours hợp lệ: {filter_log['filtered_contours']}/{filter_log['total_contours']} ({filter_log['filtered_contours']/filter_log['total_contours']*100:.1f}%)
• ⚡ Hiệu suất xử lý: {total_time:.3f}s cho ảnh {image.shape[1]}x{image.shape[0]}
• 🔄 Tốc độ xử lý: {image.shape[0]*image.shape[1]/total_time/1000:.1f} MPix/s
• 📊 Tỷ lệ mask cuối cùng: {np.count_nonzero(final_mask)/(image.shape[0]*image.shape[1])*100:.2f}% của ảnh

🎨 GHI CHÚ ẢNH TRONG PIPELINE:
• Ảnh 1: INPUT - Ảnh BGR gốc từ camera/upload
• Ảnh 2: PREPROCESSED - Ảnh grayscale sau CLAHE enhancement  
• Ảnh 3: EDGES - Ảnh biên trắng-đen từ Canny/DexiNed
• Ảnh 4: FILTERED - Contours sau morphological operations
• Ảnh 5: MASK - Vùng trắng là hộp chứa được segment
• Ảnh 6: RECTIFIED - Mask đã ép thành hình chữ nhật chuẩn
        """
        
        return final_mask, edges, gray, mask, final_mask, detailed_log

def create_gradio_interface():
    """Tạo giao diện Gradio"""
    
    segmenter = WhiteRingSegmenter()
    
    def process_uploaded_image(image):
        """Xử lý ảnh được upload"""
        if image is None:
            return None, None, None, None, None, "❌ Vui lòng upload ảnh!"
        
        try:
            # Xử lý ảnh
            final_mask, edges, gray, mask, rectified_mask, log = segmenter.process_image(image)
            
            # Tạo ảnh kết quả với overlay
            result_image = image.copy()
            result_image[final_mask > 0] = [0, 255, 0]  # Màu xanh cho vùng được segment
            
            # Overlay mask
            overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            
            return overlay, edges, gray, mask, rectified_mask, log
            
        except Exception as e:
            error_msg = f"❌ Lỗi xử lý: {str(e)}"
            logger.error(error_msg)
            return None, None, None, None, None, error_msg
    
    # Tạo giao diện
    with gr.Blocks(title="White Ring Segmentation Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 White Ring Segmentation Demo
        
        **Mục tiêu**: Tách chính xác vùng "hộp chứa" bằng pipeline dựa vào biên trắng quanh thành hộp.
        
        **Pipeline**: Tiền xử lý → Phát hiện biên → Lọc cặp biên → Tạo mask → Ép hình chữ nhật
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="📸 Upload ảnh", type="numpy")
                process_btn = gr.Button("🚀 Xử lý ảnh", variant="primary")
                
                # Tham số điều chỉnh
                gr.Markdown("### ⚙️ Tham số điều chỉnh")
                with gr.Row():
                    canny_low = gr.Slider(10, 100, 50, step=5, label="Canny Low")
                    canny_high = gr.Slider(100, 300, 150, step=10, label="Canny High")
                
                with gr.Row():
                    min_area_ratio = gr.Slider(0.01, 0.5, 0.1, step=0.01, label="Min Area Ratio")
                    rect_score_min = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Rect Score Min")
                
                with gr.Row():
                    rectify_mode = gr.Dropdown(["robust", "square", "rectangle"], value="robust", label="Rectify Mode")
                    rect_pad = gr.Slider(0, 20, 8, step=1, label="Rect Padding")
            
            with gr.Column():
                gr.Markdown("### 🔄 Pipeline Images")
                with gr.Row():
                    output_image = gr.Image(label="🎯 Kết quả cuối", type="numpy")
                    edges_image = gr.Image(label="🔍 Ảnh biên", type="numpy")
                
                with gr.Row():
                    gray_image = gr.Image(label="🎨 Ảnh grayscale", type="numpy")
                    mask_image = gr.Image(label="🎭 Mask gốc", type="numpy")
                
                rectified_image = gr.Image(label="📐 Mask đã ép", type="numpy")
                detailed_log = gr.Textbox(label="📋 Log chi tiết", lines=15, max_lines=20)
        
        # Event handlers
        def update_params(low, high, area_ratio, rect_score, mode, pad):
            segmenter.canny_low = int(low)
            segmenter.canny_high = int(high)
            segmenter.min_area_ratio = area_ratio
            segmenter.rect_score_min = rect_score
            segmenter.rectify_mode = mode
            segmenter.rect_pad = int(pad)
        
        def process_with_params(image, low, high, area_ratio, rect_score, mode, pad):
            if image is None:
                return None, None, None, None, None, "❌ Vui lòng upload ảnh!"
            
            # Cập nhật tham số
            update_params(low, high, area_ratio, rect_score, mode, pad)
            
            try:
                final_mask, edges, gray, mask, rectified_mask, log = segmenter.process_image(image)
                
                # Tạo ảnh kết quả với overlay
                result_image = image.copy()
                result_image[final_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                
                return overlay, edges, gray, mask, rectified_mask, log
                
            except Exception as e:
                error_msg = f"❌ Lỗi xử lý: {str(e)}"
                return None, None, None, None, None, error_msg
        
        # Bind events
        process_btn.click(
            fn=process_with_params,
            inputs=[input_image, canny_low, canny_high, min_area_ratio, rect_score_min, rectify_mode, rect_pad],
            outputs=[output_image, edges_image, gray_image, mask_image, rectified_image, detailed_log]
        )
        
        # Auto process when image changes
        input_image.change(
            fn=process_with_params,
            inputs=[input_image, canny_low, canny_high, min_area_ratio, rect_score_min, rectify_mode, rect_pad],
            outputs=[output_image, edges_image, gray_image, mask_image, rectified_image, detailed_log]
        )
    
    return demo

if __name__ == "__main__":
    # Tạo và chạy demo
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )
