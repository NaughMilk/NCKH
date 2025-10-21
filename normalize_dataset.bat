@echo off
REM ========================================
REM Dataset Normalizer - Strict Square Mode
REM ========================================
REM
REM Chuẩn hóa dataset với segment_square_corners
REM - Perspective warp từ 4 góc hộp vuông
REM - Mask background (chỉ giữ ROI)
REM - Xoay 180° nếu QR ở góc phải-trên
REM - Output 1024x1024 square
REM

echo.
echo ========================================
echo Dataset Normalizer - Running...
echo ========================================
echo.

python dataset_normalizer.py dataset ^
  --root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_sdy_box" ^
  --out_root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_normalized" ^
  --mask_mode square ^
  --force_square false ^
  --rot_mode any90 ^
  --final_size 0

echo.
echo ========================================
echo Done! Check output at:
echo D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_normalized
echo ========================================
echo.

pause
