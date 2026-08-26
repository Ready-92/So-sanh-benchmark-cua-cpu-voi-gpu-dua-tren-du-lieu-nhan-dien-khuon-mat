import argparse
import csv
import glob
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import torch

from face_utils import (
    detect_faces, normalize_crop, multiscale_tensors, MULTISCALE_FACTORS, preprocess_face,
)
from step3_train import FaceCNN

SIM_TEMP = 25.0  # nhiet do cho softmax tren cosine similarity


def load_model(path, num_classes, device):
    model = FaceCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def build_centroids(model, label_names, device):
    """Tinh tam cum (centroid) da chuan hoa cua tung lop tu anh train.
    Nhan dien bang cosine similarity toi centroid thay vi softmax classifier
    -> ben vung hon voi anh webcam khac domain (classifier hay overfit)."""
    centroids = []
    with torch.inference_mode():
        for name in label_names:
            feats = []
            for path in sorted(glob.glob(f"data/{name}/*.jpg")):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                t = torch.from_numpy(preprocess_face(img, 64)).unsqueeze(0).to(device)
                f = model.features(t).flatten(1)
                feats.append(f / f.norm(dim=1, keepdim=True))
            centroid = torch.cat(feats).mean(dim=0)
            centroids.append(centroid / centroid.norm())
    return torch.stack(centroids)  # (num_classes, feat_dim)


def run_demo(confidence_threshold=0.55, log_path="recognition_log.csv", log_interval=1.0,
             stats_path="recognition_stats.csv", stats_interval=1.0,
             history_frames=15, vote_ratio=0.5):
    with open("label_names.txt", encoding="utf-8") as file:
        label_names = [line.strip() for line in file if line.strip()]

    model_cpu = load_model("face_cnn.pth", len(label_names), torch.device("cpu"))
    model_gpu = None
    if torch.cuda.is_available():
        model_gpu = load_model("face_cnn.pth", len(label_names), torch.device("cuda"))

    centroids = build_centroids(model_cpu, label_names, torch.device("cpu"))

    log_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "timestamp", "frame", "student", "confidence", "x", "y", "width", "height"
        ])
        log_file.flush()

    stats_exists = os.path.exists(stats_path) and os.path.getsize(stats_path) > 0
    stats_file = open(stats_path, "a", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_file)
    if not stats_exists:
        stats_writer.writerow([
            "timestamp", "frame", "faces", "cpu_ms", "gpu_ms", "cpu_fps", "gpu_fps"
        ])
        stats_file.flush()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        log_file.close()
        raise RuntimeError("Khong mo duoc webcam index 0")

    frame_index = 0
    last_logged = {}
    last_stats = -float("inf")
    cpu_fps_display = 0.0
    gpu_fps_display = 0.0
    tracks = {}
    next_track_id = 0

    try:
        while True:
            captured, frame = capture.read()
            if not captured:
                break
            frame_index += 1
            faces = detect_faces(frame)
            face_variant_tensors = []
            for x, y, width, height in faces:
                crop = normalize_crop(frame, (x, y, width, height))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                face_variant_tensors.append(multiscale_tensors(crop))

            cpu_conf = torch.empty(0)
            cpu_pred = torch.empty(0, dtype=torch.long)
            gpu_conf = torch.empty(0)
            gpu_pred = torch.empty(0, dtype=torch.long)
            cpu_elapsed = None
            gpu_elapsed = None
            if face_variant_tensors:
                variants = np.concatenate(face_variant_tensors, axis=0)
                face_batch_cpu = torch.from_numpy(np.asarray(variants, dtype=np.float32))
                cpu_start = time.perf_counter()
                with torch.inference_mode():
                    cpu_feats = model_cpu.features(face_batch_cpu)
                cpu_feats = cpu_feats.view(len(faces), len(MULTISCALE_FACTORS), -1).mean(dim=1)
                cpu_feats = cpu_feats / cpu_feats.norm(dim=1, keepdim=True)
                cpu_probability = torch.softmax(cpu_feats @ centroids.T * SIM_TEMP, dim=1)
                cpu_conf, cpu_pred = cpu_probability.max(dim=1)
                cpu_elapsed = time.perf_counter() - cpu_start
                cpu_fps_display = 0.9 * cpu_fps_display + 0.1 * (
                    len(faces) / cpu_elapsed if cpu_elapsed else 0.0
                )

                if model_gpu is not None:
                    face_batch_gpu = face_batch_cpu.to("cuda")
                    torch.cuda.synchronize()
                    gpu_start = time.perf_counter()
                    with torch.inference_mode():
                        gpu_feats = model_gpu.features(face_batch_gpu)
                    gpu_feats = gpu_feats.view(len(faces), len(MULTISCALE_FACTORS), -1).mean(dim=1).cpu()
                    gpu_feats = gpu_feats / gpu_feats.norm(dim=1, keepdim=True)
                    gpu_probability = torch.softmax(gpu_feats @ centroids.T * SIM_TEMP, dim=1)
                    gpu_conf, gpu_pred = gpu_probability.max(dim=1)
                    torch.cuda.synchronize()
                    gpu_elapsed = time.perf_counter() - gpu_start
                    gpu_fps_display = 0.9 * gpu_fps_display + 0.1 * (
                        len(faces) / gpu_elapsed if gpu_elapsed else 0.0
                    )

            now = time.perf_counter()
            matched = set()
            for index, (x, y, width, height) in enumerate(faces):
                center = (x + width / 2, y + height / 2)
                track_id = None
                for tid, track in tracks.items():
                    tx, ty = track["center"]
                    if abs(center[0] - tx) < 60 and abs(center[1] - ty) < 60:
                        track_id = tid
                        break
                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {
                        "center": center,
                        "history": deque(maxlen=history_frames),
                        "age": 0,
                    }
                track = tracks[track_id]
                track["center"] = center
                track["age"] = 0
                matched.add(track_id)

                confidence = cpu_conf[index].item()
                track["history"].append((cpu_pred[index].item(), confidence))

                votes = Counter(p for p, _ in track["history"])
                winner, count = votes.most_common(1)[0]
                mean_conf = sum(c for p, c in track["history"] if p == winner) / count
                if count / len(track["history"]) >= vote_ratio and mean_conf >= confidence_threshold:
                    name = label_names[winner]
                    confidence = mean_conf
                else:
                    name = "Unknown"

                gpu_name = "Unknown"
                if model_gpu is not None and gpu_conf[index].item() > confidence_threshold:
                    gpu_name = label_names[gpu_pred[index].item()]

                if name != "Unknown" and now - last_logged.get(name, -float("inf")) >= log_interval:
                    log_writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        frame_index,
                        name,
                        f"{confidence:.4f}",
                        x,
                        y,
                        width,
                        height,
                    ])
                    log_file.flush()
                    last_logged[name] = now

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"CPU: {name} ({confidence:.2f})", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"GPU: {gpu_name}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            for tid in list(tracks):
                if tid not in matched:
                    tracks[tid]["age"] += 1
                    if tracks[tid]["age"] > 2 * history_frames:
                        del tracks[tid]

            if now - last_stats >= stats_interval:
                stats_writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    frame_index,
                    len(faces),
                    f"{cpu_elapsed * 1000:.4f}" if cpu_elapsed is not None else "",
                    f"{gpu_elapsed * 1000:.4f}" if gpu_elapsed is not None else "",
                    f"{cpu_fps_display:.2f}",
                    f"{gpu_fps_display:.2f}" if model_gpu else "",
                ])
                stats_file.flush()
                last_stats = now

            cv2.putText(frame, f"CPU CNN FPS: {cpu_fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            gpu_text = f"GPU CNN FPS: {gpu_fps_display:.1f}" if model_gpu else "GPU: unavailable"
            cv2.putText(frame, gpu_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow("Nhan dien - Nhan q de thoat", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        log_file.close()
        stats_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--log", default="recognition_log.csv")
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument("--stats", default="recognition_stats.csv")
    parser.add_argument("--stats-interval", type=float, default=1.0)
    parser.add_argument("--history", type=int, default=15, help="So frame dung de bo phieu (lam min ket qua)")
    parser.add_argument("--vote-ratio", type=float, default=0.5, help="Ti le dong y toi thieu de nhan ten")
    args = parser.parse_args()
    run_demo(
        confidence_threshold=args.threshold,
        log_path=args.log,
        log_interval=args.log_interval,
        stats_path=args.stats,
        stats_interval=args.stats_interval,
        history_frames=args.history,
        vote_ratio=args.vote_ratio,
    )
