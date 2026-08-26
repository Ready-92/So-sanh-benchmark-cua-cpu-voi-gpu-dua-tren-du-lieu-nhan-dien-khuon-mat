# face_utils.py
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, cascade=face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return [(x, y, w, h) for (x, y, w, h) in faces]


def normalize_crop(frame, box, margin=0.25):
    """Cat khuon mat thanh crop vuong, doi xung ca khi mat sat bien khung hinh (ngoi gan camera)."""
    x, y, w, h = box
    mw, mh = int(round(w * margin)), int(round(h * margin))
    H, W = frame.shape[:2]
    x0, y0 = x - mw, y - mh
    x1, y1 = x + w + mw, y + h + mh
    # Neu le tran ra ngoai khung hinh: pad bang BORDER_REPLICATE thay vi cat xot,
    # de crop luon doi xung quanh khuon mat o moi khoang cach.
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = frame[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch == cw:
        return crop
    if ch > cw:
        pad = ch - cw
        return cv2.copyMakeBorder(crop, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_REPLICATE)
    pad = cw - ch
    return cv2.copyMakeBorder(crop, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_REPLICATE)


MULTISCALE_FACTORS = (0.8, 1.0, 1.25)


def multiscale_tensors(gray_crop, size=64, factors=MULTISCALE_FACTORS):
    """Tao nhieu phien ban zoom cua crop de vote da ti le."""
    h, w = gray_crop.shape
    tensors = []
    for factor in factors:
        nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
        resized = cv2.resize(gray_crop, (nw, nh))
        if factor >= 1:
            y, x = (nh - h) // 2, (nw - w) // 2
            variant = resized[y:y + h, x:x + w]
        else:
            top, left = (h - nh) // 2, (w - nw) // 2
            variant = cv2.copyMakeBorder(resized, top, h - nh - top, left, w - nw - left, cv2.BORDER_REPLICATE)
        tensors.append(preprocess_face(variant, size=size))
    # np.stack: moi tensor (1,64,64) -> (3,1,64,64) de model nhan dung batch 4D
    return np.stack(tensors, axis=0)


def preprocess_face(face_img, size=64):
    face_resized = cv2.resize(face_img, (size, size))
    # Chuan hoa anh sang: CLAHE giu tuong phan cuc bo, chong lech sang giua webcam va anh train
    face_resized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(face_resized)
    face_norm = face_resized.astype('float32') / 255.0
    # z-score: bo phu thuoc vao do sang/do tuong phan toan cuc (den khac, phong khac, camera khac)
    face_norm = (face_norm - face_norm.mean()) / (face_norm.std() + 1e-5)
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

    # Them bien the anh sang manh de model bat bien voi den/phong/camera
    for beta in (-50, 50):
        variants.append(cv2.convertScaleAbs(face_img, alpha=1.0, beta=beta))
    for alpha in (0.75, 1.3):
        variants.append(cv2.convertScaleAbs(face_img, alpha=alpha, beta=0))
    for gamma in (0.6, 1.5):
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        variants.append(cv2.LUT(face_img, lut))

    # Bien the zoom: chong lai viec model hoc nham khoang cach ngoi = identity
    # Dai zoom rong (0.65 - 1.45) de model ben ca khi ngoi rat gan hoac rat xa
    for factor in (0.65, 0.8, 1.15, 1.3, 1.45):
        variants.append(scale_variant(face_img, factor))

    return variants


def scale_variant(face_img, factor):
    """Zoom ra (factor < 1) hoac zoom vao (factor > 1) nhung giu nguyen kich thuoc anh."""
    h, w = face_img.shape
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    resized = cv2.resize(face_img, (nw, nh))
    if factor >= 1:
        y, x = (nh - h) // 2, (nw - w) // 2
        return resized[y:y + h, x:x + w]
    top, left = (h - nh) // 2, (w - nw) // 2
    return cv2.copyMakeBorder(resized, top, h - nh - top, left, w - nw - left, cv2.BORDER_REPLICATE)