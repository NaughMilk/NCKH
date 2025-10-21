# Dataset Normalizer - Strict Square Mode

## 📋 Mô tả

Tool chuẩn hóa dataset với **perspective transform** từ 4 góc hộp vuông (`segment_square_corners`).

## 🎯 Tính năng

1. **Perspective Warp**: Sử dụng `cv2.getPerspectiveTransform` để warp chính xác từ 4 góc
2. **Square Masking**: Mask background, chỉ giữ ROI theo hình vuông
3. **QR Orientation**: Tự động detect và xoay để QR ở góc bottom-left
4. **Flexible Output**: Hỗ trợ nhiều modes (square/rectangle, resize/no-resize, rotation modes)

## 📦 Requirements

Dataset JSON phải có:
- ✅ `segment_square_corners`: **4 điểm góc hộp vuông** (bắt buộc)
- ⚠️ `qr_corners`: 4 điểm góc QR (optional, để detect orientation)
- ℹ️ `segment_corners`: Polygon segment (optional, chỉ dùng cho mask mode=polygon)

## 🚀 Usage

### Mode 1: Dataset batch processing

```bash
python dataset_normalizer.py dataset \
  --root "path/to/dataset_sdy_box" \
  --out_root "path/to/output" \
  --mask_mode square \
  --force_square true \
  --rot_mode only180 \
  --final_size 1024
```

### Mode 2: Single image

```bash
python dataset_normalizer.py single \
  --image "path/to/image.jpg" \
  --json "path/to/metadata.json" \
  --out "path/to/output.jpg" \
  --out_meta "path/to/output_meta.json" \
  --mask_mode square \
  --force_square true \
  --rot_mode only180 \
  --final_size 1024
```

## ⚙️ Parameters

### `--mask_mode`
- `square` (default): Mask theo 4 góc `segment_square_corners`
- `polygon`: Mask theo polygon `segment_corners`

### `--force_square`
- `true` (default): Output là hình vuông với `side = max(width, height)`
- `false`: Giữ tỷ lệ chữ nhật từ geometry của box

### `--rot_mode`
- `only180` (default): Chỉ xoay 180° nếu QR ở TR (top-right)
- `any90`: Xoay 0/90/180/270° để luôn đưa QR về BL

### `--final_size`
- `1024` (default): Resize về 1024x1024
- `0`: Không resize, giữ kích thước tự nhiên sau warp
- `512`, `2048`, etc: Resize về kích thước tùy chọn

## 📊 Batch Scripts

### Standard (1024x1024, QR at BL/TR)
```bash
normalize_dataset.bat
```

### Any 90° rotation (QR luôn ở BL)
```bash
normalize_dataset_any90.bat
```

### Natural size (không resize)
```bash
normalize_dataset_noResize.bat
```

## 📄 Output Metadata

Metadata được augment với trường `normalization`:

```json
{
  "normalization": {
    "mask_mode": "square",
    "force_square": true,
    "rot_mode": "only180",
    "extra_rotation_deg": 180,
    "qr_corner_after": "BL",
    "original_shape": [1080, 1920],
    "normalized_shape": [1024, 1024],
    "timestamp": "2025-10-21T20:52:16.634499"
  }
}
```

## 🎨 QR Corner Detection

Sau khi warp, QR được detect ở 4 góc:
- **BL (Bottom-Left)**: ✅ Ideal position
- **TR (Top-Right)**: ✅ Auto-rotated 180° → BL
- **TL (Top-Left)**: ⚠️ Warning (không xoay với `only180` mode)
- **BR (Bottom-Right)**: ⚠️ Warning (không xoay với `only180` mode)

Nếu muốn **tất cả QR đều ở BL**, dùng `--rot_mode any90`.

## 📈 Success Rate

Với dataset hiện tại: **214/214 (100%)**

## 🔧 Troubleshooting

### Error: "segment_square_corners missing"
→ JSON chưa có field này. Cần re-run pipeline để generate.

### QR ở TL/BR không được xoay
→ Dùng `--rot_mode any90` thay vì `only180`

### Ảnh bị resize nhỏ
→ Dùng `--final_size 0` để giữ kích thước tự nhiên

### Background không đen hoàn toàn
→ Kiểm tra `segment_square_corners` có đúng không

