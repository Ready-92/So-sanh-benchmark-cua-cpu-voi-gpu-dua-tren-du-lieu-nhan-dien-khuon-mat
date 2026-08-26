# capture_multi.py
# Bat khuon mat tu webcam ~10s, PHAN RIENG TUNG NGUOI roi so khớp với 2 lớp.
import glob
import os
import time
from collections import Counter

import cv2
import numpy as np
import torch

from face_utils import detect_faces, normalize_crop, preprocess_face
from step3_train import FaceCNN

CAPTURE_SECONDS = 10

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

print(f"Hay de 1 hoac 2 nguoi dung truoc camera trong {CAPTURE_SECONDS}s ...")
start = time.time()
all_crops = []
while time.time() - start < CAPTURE_SECONDS:
    ok, frame = capture.read()
    if not ok:
        continue
    for x, y, w, h in detect_faces(frame):
        crop = normalize_crop(frame, (x, y, w, h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        all_crops.append((crop, (x + w // 2, y + h // 2)))
capture.release()

print(f"Da bat duoc {len(all_crops)} khuon mat")
if len(all_crops) < 10:
    print("Khong du khuon mat. Thu lai voi anh sang tot hon.")
    raise SystemExit

crops = [c for c, _ in all_crops]
centers = [p for _, p in all_crops]
F = torch.cat([embed_face(c) for c in crops])
# Du doan lop tu anh goc (khong phai tu vector dac trung)
img_batch = torch.stack([torch.from_numpy(preprocess_face(c, 64)) for c in crops])
with torch.inference_mode():
    probs_all = torch.softmax(model(img_batch), dim=1)

# Gom nhom thanh cac nguoi khac nhau: neu 2 crop giong nhau (sim cao) -> cung nguoi
people = []
used = [False] * len(crops)
for i in range(len(crops)):
    if used[i]:
        continue
    group = [i]
    used[i] = True
    for j in range(i + 1, len(crops)):
        if not used[j] and (F[i] @ F[j]).item() > 0.85:
            group.append(j)
            used[j] = True
    people.append(group)

# Gop nhom nho cung nguoi (nguong thap hon, dung centroid)
changed = True
while changed and len(people) > 1:
    changed = False
    for a in range(len(people)):
        for b in range(a + 1, len(people)):
            ca = F[people[a]].mean(dim=0); ca = ca / ca.norm()
            cb = F[people[b]].mean(dim=0); cb = cb / cb.norm()
            if (ca @ cb).item() > 0.75:
                people[a].extend(people[b])
                people.pop(b)
                changed = True
                break
        if changed:
            break

# Load anh train
train_paths = sorted(glob.glob("data/student_2/*.jpg")) + sorted(glob.glob("data/student_3/*.jpg"))
T = []
valid_paths = []
for p in train_paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    T.append(embed_face(img))
    valid_paths.append(p)
T = torch.cat(T)

os.makedirs("capture_multi", exist_ok=True)
for pi, group in enumerate(people):
    g = sorted(group, key=lambda i: -len([1 for j in group if (F[i] @ F[j]).item() > 0.8]))
    centroid = F[group].mean(dim=0); centroid = centroid / centroid.norm()
    sims = (T @ centroid).numpy()
    order = np.argsort(-sims)
    pred = labels[int(probs_all[group].mean(dim=0).argmax())]
    xs = [centers[i][0] for i in group]
    print(f"\n=== NGUOI {pi + 1} ({len(group)} crop, vi tri x ~ {int(np.mean(xs))}) ===")
    print(f"  Model du doan: {pred}")
    for idx in order[:5]:
        folder = "student_2" if "student_2" in valid_paths[idx] else "student_3"
        print(f"    giong nhat: {folder:10s} {valid_paths[idx]:28s} sim={sims[idx]:.4f}")
    s2 = sims[np.array(["student_2" in p for p in valid_paths])]
    s3 = sims[np.array(["student_3" in p for p in valid_paths])]
    print(f"  mean sim vs student_2: {s2.mean():.4f} | vs student_3: {s3.mean():.4f}")
    for k, j in enumerate(group[:5]):
        cv2.imwrite(f"capture_multi/p{pi + 1}_{k}.jpg", crops[j])
print("\nDa luu crop tung nguoi vao capture_multi/ de ban kiem tra.")
