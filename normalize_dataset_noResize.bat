@echo off
REM ========================================
REM Dataset Normalizer - No Resize (Natural Size)
REM ========================================
REM
REM Giữ kích thước tự nhiên sau warp (không resize về 1024)
REM Mỗi ảnh có thể có size khác nhau tùy vào segment
REM

echo.
echo ========================================
echo Dataset Normalizer - Natural Size Mode
echo ========================================
echo.

python dataset_normalizer.py dataset ^
  --root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_sdy_box" ^
  --out_root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_normalized_noResize" ^
  --mask_mode square ^
  --force_square true ^
  --rot_mode only180 ^
  --final_size 0

echo.
echo ========================================
echo Done! Natural size output (no resize)
echo ========================================
echo.

pause

