import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch

from face_utils import detect_faces, preprocess_face
from step3_train import FaceCNN


def load_model(path, num_classes, device):
    model = FaceCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def run_demo(confidence_threshold=0.6, log_path="recognition_log.csv", log_interval=1.0):
    with open("label_names.txt", encoding="utf-8") as file:
        label_names = [line.strip() for line in file if line.strip()]

    model_cpu = load_model("face_cnn.pth", len(label_names), torch.device("cpu"))
    model_gpu = None
    if torch.cuda.is_available():
        model_gpu = load_model("face_cnn.pth", len(label_names), torch.device("cuda"))

    log_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "timestamp", "frame", "student", "confidence", "x", "y", "width", "height"
        ])
        log_file.flush()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        log_file.close()
        raise RuntimeError("Khong mo duoc webcam index 0")

    frame_index = 0
    last_logged = {}
    cpu_fps_display = 0.0
    gpu_fps_display = 0.0

    try:
        while True:
            captured, frame = capture.read()
            if not captured:
                break
            frame_index += 1
            faces = detect_faces(frame)
            face_tensors = []
            for x, y, width, height in faces:
                crop = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
                face_tensors.append(preprocess_face(crop, size=64))

            cpu_conf = torch.empty(0)
            cpu_pred = torch.empty(0, dtype=torch.long)
            gpu_conf = torch.empty(0)
            gpu_pred = torch.empty(0, dtype=torch.long)
            if face_tensors:
                face_batch_cpu = torch.from_numpy(np.asarray(face_tensors, dtype=np.float32))
                cpu_start = time.perf_counter()
                with torch.inference_mode():
                    cpu_output = model_cpu(face_batch_cpu)
                    cpu_probability = torch.softmax(cpu_output, dim=1)
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
                        gpu_output = model_gpu(face_batch_gpu)
                        gpu_probability = torch.softmax(gpu_output, dim=1)
                        gpu_conf, gpu_pred = gpu_probability.max(dim=1)
                    torch.cuda.synchronize()
                    gpu_elapsed = time.perf_counter() - gpu_start
                    gpu_fps_display = 0.9 * gpu_fps_display + 0.1 * (
                        len(faces) / gpu_elapsed if gpu_elapsed else 0.0
                    )

            now = time.perf_counter()
            for index, (x, y, width, height) in enumerate(faces):
                confidence = cpu_conf[index].item()
                name = label_names[cpu_pred[index].item()] if confidence > confidence_threshold else "Unknown"
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--log", default="recognition_log.csv")
    parser.add_argument("--log-interval", type=float, default=1.0)
    args = parser.parse_args()
    run_demo(
        confidence_threshold=args.threshold,
        log_path=args.log,
        log_interval=args.log_interval,
    )
