@echo off
REM Test Image Aligner Module
echo ================================================
echo Testing Image Alignment by Box ID
echo ================================================

python sections_k\k_image_aligner.py --dataset sdy_project\dataset_sdy_box --output sdy_project\aligned_images --box_id ALL --num_images 3 --target_corner BL

echo.
echo ================================================
echo Test Complete!
echo Check output in: sdy_project\aligned_images
echo ================================================
pause

