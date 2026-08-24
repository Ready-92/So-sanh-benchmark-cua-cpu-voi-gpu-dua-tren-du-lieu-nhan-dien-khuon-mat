# face_utils.py
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, cascade=face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return [(x, y, w, h) for (x, y, w, h) in faces]

def preprocess_face(face_img, size=64):
    face_resized = cv2.resize(face_img, (size, size))
    face_norm = face_resized.astype('float32') / 255.0
    face_norm = face_norm[np.newaxis, :, :]
    return face_norm

def augment_face(face_img):
    """Sinh thêm biến thể từ 1 ảnh gốc để tăng đa dạng dữ liệu (lỗi #2)."""
    variants = [face_img]
    variants.append(cv2.flip(face_img, 1))  # lật ngang

    h, w = face_img.shape
    center = (w // 2, h // 2)
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, M, (w, h))
        variants.append(rotated)

    bright = cv2.convertScaleAbs(face_img, alpha=1.0, beta=25)
    dark = cv2.convertScaleAbs(face_img, alpha=1.0, beta=-25)
    variants.extend([bright, dark])

    return variants