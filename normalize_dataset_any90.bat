@echo off
REM ========================================
REM Dataset Normalizer - Any 90° Rotation
REM ========================================
REM
REM Mode này sẽ xoay 0/90/180/270° để luôn đưa QR về BL
REM Phù hợp khi QR có thể ở bất kỳ góc nào
REM

echo.
echo ========================================
echo Dataset Normalizer - Any 90deg Mode
echo ========================================
echo.

python dataset_normalizer.py dataset ^
  --root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_sdy_box" ^
  --out_root "D:\NCKH CODE\PIPELINE ORIGINAL\sdy_project\dataset_normalized_any90" ^
  --mask_mode square ^
  --force_square true ^
  --rot_mode any90 ^
  --final_size 1024

echo.
echo ========================================
echo Done! QR should be at BL for all images
echo ========================================
echo.

pause

