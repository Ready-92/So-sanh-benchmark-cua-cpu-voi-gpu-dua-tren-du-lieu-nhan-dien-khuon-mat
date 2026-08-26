# test_scale_robustness.py
# Kiem tra model hien tai (face_cnn.pth) co bi nham danh tinh theo khoang cach khong.
# Mo phong: ngoi gan = khuon mat lon (zoom out trong crop), ngoi xa = khuon mat nho.
import glob
import cv2
import numpy as np
import torch

from face_utils import preprocess_face, scale_variant, multiscale_tensors
from step3_train import FaceCNN

SCALES = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)

labels = [line.strip() for line in open("label_names.txt", encoding="utf-8") if line.strip()]
model = FaceCNN(num_classes=len(labels))
model.load_state_dict(torch.load("face_cnn.pth", map_location="cpu", weights_only=True))
model.eval()

def predict(gray_crop):
    t = multiscale_tensors(gray_crop)  # (3,1,64,64)
    with torch.inference_mode():
        prob = torch.softmax(model(torch.from_numpy(t)), dim=1).mean(dim=0)
    return prob.numpy()

print("Label thu tu:", labels)
for student in labels:
    paths = sorted(glob.glob(f"data/{student}/*.jpg"))[:30]
    print(f"\n=== {student} ({len(paths)} anh) — ti le du doan dung theo scale ===")
    stats = {s: 0 for s in SCALES}
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        base = cv2.resize(img, (64, 64))
        for s in SCALES:
            variant = scale_variant(base, s) if s != 1.0 else base
            prob = predict(variant)
            pred = labels[int(prob.argmax())]
            if pred == student:
                stats[s] += 1
    total = len(paths)
    for s in SCALES:
        print(f"  scale {s:.2f}: {stats[s]}/{total} dung")
