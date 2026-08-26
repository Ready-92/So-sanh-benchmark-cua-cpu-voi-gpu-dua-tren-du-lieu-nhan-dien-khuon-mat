# test_pipeline_distance.py
# Mo phong day du pipeline cua step5_demo_video.py:
# dat khuon mat o cac kich thuoc khac nhau (gan = to, xa = nho) tren nen,
# chay detect_faces + normalize_crop + multiscale_tensors + mean softmax.
import glob
import cv2
import numpy as np
import torch

from face_utils import detect_faces, normalize_crop, multiscale_tensors
from step3_train import FaceCNN

FACE_SIZES = (80, 100, 140, 180, 240, 320, 400)  # kich thuoc khuon mat tren khung 640x480

labels = [line.strip() for line in open("label_names.txt", encoding="utf-8") if line.strip()]
model = FaceCNN(num_classes=len(labels))
model.load_state_dict(torch.load("face_cnn.pth", map_location="cpu", weights_only=True))
model.eval()

CANVAS_W, CANVAS_H = 640, 480

def make_frame(face_img, size):
    small = cv2.resize(face_img, (size, size))
    canvas = np.full((CANVAS_H, CANVAS_W), 128, dtype=np.uint8)
    x0 = (CANVAS_W - size) // 2
    y0 = (CANVAS_H - size) // 2
    canvas[y0:y0 + size, x0:x0 + size] = small
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

def pipeline_predict(face_img, size):
    frame = make_frame(face_img, size)
    faces = detect_faces(frame)
    if not faces:
        return None
    x, y, w, h = faces[0]
    crop = normalize_crop(frame, (x, y, w, h))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    t = multiscale_tensors(crop)  # (3,1,64,64)
    with torch.inference_mode():
        prob = torch.softmax(model(torch.from_numpy(t)), dim=1).mean(dim=0)
    return prob.numpy(), (w, h)

for student in labels:
    paths = sorted(glob.glob(f"data/{student}/*.jpg"))[:30]
    print(f"\n=== {student} — du doan qua pipeline that theo kich thuoc khuon mat ===")
    for size in FACE_SIZES:
        correct = 0
        detected = 0
        wrong_to = {}
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = pipeline_predict(img, size)
            if res is None:
                continue
            prob, box = res
            detected += 1
            pred = labels[int(prob.argmax())]
            if pred == student:
                correct += 1
            else:
                wrong_to[pred] = wrong_to.get(pred, 0) + 1
        if detected:
            print(f"  face={size:4d}px (box ~{int(size*1.5)}px): {correct}/{detected} dung, sai sang: {wrong_to or 'khong'}")
        else:
            print(f"  face={size:4d}px: KHONG detect duoc mat")
