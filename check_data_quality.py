# check_data_quality.py
# Kiem tra chat luong du lieu: tim anh co the bi dat nham thu muc.
# Dung dac trung cua model (features truoc classifier) de so sanh khuon mat.
import glob
import os

import cv2
import numpy as np
import torch

from face_utils import preprocess_face, face_cascade
from step3_train import FaceCNN


def load_features():
    labels = [line.strip() for line in open("label_names.txt", encoding="utf-8") if line.strip()]
    model = FaceCNN(num_classes=len(labels))
    model.load_state_dict(torch.load("face_cnn.pth", map_location="cpu", weights_only=True))
    model.eval()

    class_paths = []
    class_names = []
    for name in labels:
        paths = sorted(glob.glob(f"data/{name}/*.jpg"))
        if paths:
            class_paths.append(paths)
            class_names.append(name)

    features_by_class = []
    with torch.inference_mode():
        for paths in class_paths:
            feats = []
            valid = []
            for p in paths:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if max(h, w) > 600:
                    faces = face_cascade.detectMultiScale(img, 1.1, 5)
                    if len(faces) == 0:
                        continue
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    img = img[y:y + fh, x:x + fw]
                tensor = torch.from_numpy(preprocess_face(img, size=64)).unsqueeze(0)
                feat = model.features(tensor).flatten(1)
                feat = feat / feat.norm(dim=1, keepdim=True)
                feats.append(feat)
                valid.append(p)
            features_by_class.append((valid, torch.cat(feats, dim=0)))

    return class_names, features_by_class


def main():
    class_names, features_by_class = load_features()

    centroids = []
    for paths, feats in features_by_class:
        centroids.append(feats.mean(dim=0, keepdim=True))
        centroids[-1] = centroids[-1] / centroids[-1].norm()

    print("=== Anh nghi bi dat nham thu muc (gan tam lop khac hon tam lop minh) ===")
    for i, (paths, feats) in enumerate(features_by_class):
        for path, feat in zip(paths, feats):
            sims = [(class_names[j], (feat * centroids[j]).sum().item()) for j in range(len(centroids))]
            sims.sort(key=lambda t: -t[1])
            best = sims[0][0]
            if best != class_names[i]:
                own = next(s for c, s in sims if c == class_names[i])
                print(f"{path:40s} trong {class_names[i]:10s} -> giong {best:10s} ({sims[0][1]:.3f}) | tam {class_names[i]}: {own:.3f}")

    print()
    print("=== Top anh lech nhat trong tung lop (do tuong dong voi tam lop minh) ===")
    for i, (paths, feats) in enumerate(features_by_class):
        sims = sorted(
            [(p, (f * centroids[i]).sum().item()) for p, f in zip(paths, feats)],
            key=lambda t: t[1],
        )
        for p, s in sims[:5]:
            print(f"{class_names[i]:10s} {p:40s} sim={s:.3f}")


if __name__ == "__main__":
    main()
