"""
Phase 3 image-parsing subpackage.

Stages:
  1. layout   — boundary/container detection (OpenCV, classical CV)
  2. hash     — icon matching via perceptual hashing against AWS icon pack
  3. yolo     — YOLOv8 fallback for icons Stage 2 misses
  4. ocr      — text/label extraction (runs unconditionally alongside 2+3)
"""
