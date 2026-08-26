# capture_me.py
# Bat khuon mat tu webcam ~10s, so sanh voi toan bo anh train de tim danh tinh that.
import glob
import os
import time

import cv2
import numpy as np
import torch

from face_utils import detect_faces, normalize_crop, preprocess_face
from step3_train import FaceCNN

CAPTURE_SECONDS = 10
MIN_CROPS = 10

labels = [line.strip() for line in open("label_names.txt", encoding="utf-8") if line.strip()]
model = FaceCNN(num_classes=len(labels))
model.load_state_dict(torch.load("face_cnn.pth", map_location="cpu", weights_only=True))
model.eval()


def embed_face(gray_face):
    t = torch.from_numpy(preprocess_face(gray_face, 64)).unsqueeze(0)
    with torch.inference_mode():
        f = model.features(t).flatten(1)
    return f / f.norm()


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise RuntimeError("Khong mo duoc webcam index 0")

print(f"Hay nhin thang vao camera trong {CAPTURE_SECONDS}s ...")
start = time.time()
crops = []
while time.time() - start < CAPTURE_SECONDS:
    ok, frame = capture.read()
    if not ok:
        continue
    for x, y, w, h in detect_faces(frame):
        crop = normalize_crop(frame, (x, y, w, h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crops.append(crop)
    if len(crops) >= 60:  # da du, dung som
        break
capture.release()

print(f"Da bat duoc {len(crops)} khuon mat")
if len(crops) < MIN_CROPS:
    print("Khong du khuon mat de so sanh. Thu lai voi anh sang tot hon.")
    raise SystemExit

live_feats = torch.cat([embed_face(c) for c in crops])
live_centroid = live_feats.mean(dim=0)
live_centroid = live_centroid / live_centroid.norm()

# Load toan bo anh train + embedding
train_paths = sorted(glob.glob("data/student_2/*.jpg")) + sorted(glob.glob("data/student_3/*.jpg"))
train_feats = []
valid_paths = []
for p in train_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    train_feats.append(embed_face(img))
    valid_paths.append(p)
T = torch.cat(train_feats)

sims = (T @ live_centroid).numpy()
order = np.argsort(-sims)

print("\n=== Top 10 anh train giong mat ban nhat ===")
for idx in order[:10]:
    folder = "student_2" if "student_2" in valid_paths[idx] else "student_3"
    print(f"  {folder:10s} {valid_paths[idx]:28s} sim={sims[idx]:.4f}")

for folder in labels:
    mask = np.array([("student_2" in p) == (folder == "student_2") for p in valid_paths])
    print(f"\nDo tuong dong mat ban vs TOAN BO {folder}: {sims[mask].mean():.4f} (max {sims[mask].max():.4f})")

# Luu lai crop de user kiem tra
os.makedirs("capture_me", exist_ok=True)
for i, c in enumerate(crops[::max(1, len(crops) // 10)][:10]):
    cv2.imwrite(f"capture_me/live_{i}.jpg", c)
print("\nDa luu 10 crop vao capture_me/ de ban tu kiem tra.")
