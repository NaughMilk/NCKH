import cv2
import numpy as np

def preprocess_cpu(bgr):
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    enh = cv2.cvtColor(cv2.merge([Lc, A, B]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    norm = cv2.addWeighted(tophat, 0.3, gray, 0.7, 0)
    return norm

def preprocess_gpu(bgr, use_gpu=True):
    try:
        from sections_a.a_config import CUDA_AVAILABLE
    except Exception:
        CUDA_AVAILABLE = False
    if not use_gpu or not CUDA_AVAILABLE:
        return preprocess_cpu(bgr)
    try:
        gpu_bgr = cv2.cuda_GpuMat(); gpu_bgr.upload(bgr)
        gpu_den = cv2.cuda.bilateralFilter(gpu_bgr, 7, 50, 50)
        gpu_lab = cv2.cuda.cvtColor(gpu_den, cv2.COLOR_BGR2LAB)
        gpu_l, gpu_a, gpu_b = cv2.cuda.split(gpu_lab)
        l_cpu, a_cpu, b_cpu = gpu_l.download(), gpu_a.download(), gpu_b.download()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lc_cpu = clahe.apply(l_cpu)
        gpu_lc = cv2.cuda_GpuMat(); gpu_lc.upload(lc_cpu)
        gpu_a.upload(a_cpu); gpu_b.upload(b_cpu)
        gpu_enh = cv2.cuda.merge([gpu_lc, gpu_a, gpu_b])
        gpu_enh = cv2.cuda.cvtColor(gpu_enh, cv2.COLOR_LAB2BGR)
        gpu_gray = cv2.cuda.cvtColor(gpu_enh, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        gpu_tophat = cv2.cuda.morphologyEx(gpu_gray, cv2.MORPH_TOPHAT, se)
        gpu_norm = cv2.cuda.addWeighted(gpu_tophat, 0.3, gpu_gray, 0.7, 0)
        return gpu_norm.download()
    except Exception:
        return preprocess_cpu(bgr)

def preprocess(bgr):
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    labc = cv2.merge([Lc, A, B])
    enh = cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.addWeighted(gray, 1.6, bg, -0.6, 0)
    return norm



