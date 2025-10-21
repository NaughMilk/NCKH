# Image Alignment by Box ID

## Tính năng mới: Căn chỉnh ảnh theo Box ID

Tính năng này cho phép bạn:
1. Upload weights đã train (YOLO và U2-Net) - tùy chọn
2. Chọn ngẫu nhiên N ảnh từ mỗi Box ID trong dataset
3. Căn chỉnh các ảnh sao cho QR code nằm ở cùng một vị trí góc
4. Xuất ra folder có cấu trúc: `output/[box_id]/image1.jpg, image2.jpg, image3.jpg`

## Cách sử dụng

### 1. Sử dụng qua UI (Gradio)

1. Chạy pipeline:
   ```bash
   python run.py
   ```
   hoặc
   ```bash
   START.bat
   ```

2. Mở tab **"Image Alignment"** (Tab thứ 5)

3. (Tùy chọn) Upload trained weights:
   - YOLO weight (.pt file)
   - U2-Net weight (.pth file)

4. Cấu hình:
   - **Dataset Path**: Đường dẫn đến dataset (mặc định: `sdy_project/dataset_sdy_box`)
   - **Output Path**: Nơi lưu ảnh đã căn chỉnh (mặc định: `sdy_project/aligned_images`)
   - **Box ID**: Nhập ID cụ thể hoặc `ALL` để xử lý tất cả
   - **Number of Images per ID**: Số lượng ảnh chọn ngẫu nhiên (mặc định: 3)
   - **Target QR Corner**: Vị trí góc QR mong muốn (TL/TR/BR/BL, mặc định: BL)
   - **Force Square Output**: Bắt buộc output hình vuông
   - **Mask Mode**: square hoặc polygon
   - **Final Resize**: Resize cuối cùng (0 = không resize)

5. Click **"Scan Dataset"** để xem danh sách Box IDs có trong dataset

6. Click **"Align Images"** để chạy căn chỉnh

7. Kết quả:
   - **Aligned Images Preview**: Xem trước các ảnh đã căn chỉnh
   - **Alignment Log**: Log chi tiết quá trình xử lý
   - **Output Folder**: Đường dẫn thư mục chứa kết quả

### 2. Sử dụng qua Command Line

```bash
python sections_k\k_image_aligner.py --dataset sdy_project\dataset_sdy_box --output sdy_project\aligned_images --box_id ALL --num_images 3 --target_corner BL
```

Hoặc chạy batch file test:
```bash
test_aligner.bat
```

### 3. Sử dụng như Python Module

```python
from sections_k.k_image_aligner import align_images_by_id

# Căn chỉnh ảnh cho một Box ID cụ thể
aligned_images, output_folder, log = align_images_by_id(
    dataset_root="sdy_project/dataset_sdy_box",
    output_root="sdy_project/aligned_images",
    box_id="623333",  # hoặc "ALL" để xử lý tất cả
    num_images=3,
    target_qr_corner="BL",
    mask_mode="square",
    force_square=True,
    final_size=0
)

print(f"Aligned {len(aligned_images)} images")
print(f"Saved to: {output_folder}")
print(f"Log:\n{log}")
```

## Logic hoạt động

1. **Scan Dataset**: 
   - Quét tất cả file JSON trong thư mục `meta/`
   - Nhóm ảnh theo `id_qr` hoặc `box_id`
   - Đếm số lượng ảnh cho mỗi ID

2. **Select Images**:
   - Chọn ngẫu nhiên N ảnh từ mỗi Box ID
   - Nếu Box ID có ít hơn N ảnh, sử dụng tất cả ảnh có sẵn

3. **Alignment Process** (cho mỗi ảnh):
   - Load ảnh và metadata (JSON)
   - Mask vùng ngoài ROI (segment_square_corners hoặc segment_corners)
   - Warp perspective để biến đổi thành hình chữ nhật/vuông
   - Xác định vị trí QR code (TL/TR/BR/BL)
   - Xoay ảnh (0°/90°/180°/270°) để QR code về vị trí mong muốn
   - (Tùy chọn) Resize về kích thước cuối cùng

4. **Export**:
   - Lưu vào: `output_root/[box_id]/image_1.jpg, image_2.jpg, ...`
   - Mỗi Box ID có một thư mục riêng

## Ví dụ Output Structure

```
sdy_project/aligned_images/
├── 623333/
│   ├── IMG_001_1.jpg
│   ├── IMG_005_2.jpg
│   └── IMG_012_3.jpg
├── 695991/
│   ├── IMG_003_1.jpg
│   ├── IMG_007_2.jpg
│   └── IMG_015_3.jpg
└── 861326/
    ├── IMG_002_1.jpg
    ├── IMG_009_2.jpg
    └── IMG_018_3.jpg
```

## Yêu cầu Dataset

Dataset phải có cấu trúc:
```
dataset_root/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── meta/
    ├── img1.json
    ├── img2.json
    ├── img3.json
    └── img4.json
```

Mỗi file JSON phải có:
- `id_qr` hoặc `box_id`: ID của hộp
- `segment_square_corners`: 4 điểm góc của vùng vuông (required)
- `segment_corners`: polygon (optional, dùng khi mask_mode="polygon")
- `qr_corners`: 4 điểm góc của QR code (dùng để xác định hướng)

## Ghi chú

- Tính năng này sử dụng logic từ `dataset_normalizer.py`
- QR corner detection: TL (Top-Left), TR (Top-Right), BR (Bottom-Right), BL (Bottom-Left)
- Rotation chỉ sử dụng các góc 90°, 180°, 270° để tránh flip/mirror
- Default target corner là BL (Bottom-Left)

## Troubleshooting

**Lỗi: "segment_square_corners missing"**
- Đảm bảo file JSON có field `segment_square_corners` với 4 điểm

**Lỗi: "Box ID not found"**
- Check lại tên Box ID (case-sensitive)
- Dùng "Scan Dataset" để xem danh sách IDs

**Ảnh không xoay đúng**
- Check field `qr_corners` trong JSON
- Thử thay đổi `target_corner` (TL/TR/BR/BL)

**Dataset path not found**
- Dùng đường dẫn tương đối từ project root
- Hoặc dùng đường dẫn tuyệt đối

## Author

SDY Pipeline Team - 2025

