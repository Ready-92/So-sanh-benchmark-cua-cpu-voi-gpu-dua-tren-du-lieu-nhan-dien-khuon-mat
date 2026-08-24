import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from face_utils import detect_faces, preprocess_face
from step3_train import FaceCNN


def percentile(values, value):
    return float(np.percentile(np.asarray(values), value))


def benchmark_pipeline(model, device, camera_index, seconds, warmup_frames, display):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc webcam index {camera_index}")

    model = model.to(device)
    model.eval()
    total_times = []
    detection_times = []
    preprocess_times = []
    inference_times = []
    face_count = 0
    frame_count = 0
    measured_count = 0
    start_run = time.perf_counter()

    try:
        while time.perf_counter() - start_run < seconds:
            captured, frame = capture.read()
            if not captured:
                break
            frame_start = time.perf_counter()

            detection_start = time.perf_counter()
            faces = detect_faces(frame)
            detection_times.append((time.perf_counter() - detection_start) * 1000)

            preprocess_start = time.perf_counter()
            face_inputs = []
            for x, y, width, height in faces:
                crop = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
                face_inputs.append(preprocess_face(crop, size=64))
            preprocess_times.append((time.perf_counter() - preprocess_start) * 1000)

            inference_start = time.perf_counter()
            if face_inputs:
                batch = torch.from_numpy(np.asarray(face_inputs, dtype=np.float32)).to(device)
                with torch.inference_mode():
                    model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            inference_times.append((time.perf_counter() - inference_start) * 1000)

            if display:
                for x, y, width, height in faces:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.imshow("Pipeline benchmark - press q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            face_count += len(faces)
            if frame_count > warmup_frames:
                total_times.append((time.perf_counter() - frame_start) * 1000)
                measured_count += 1
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()

    if not total_times:
        raise RuntimeError("Khong co frame hop le de do webcam")

    elapsed_seconds = time.perf_counter() - start_run
    return {
        "device": device.type,
        "frames": measured_count,
        "total_frames": frame_count,
        "faces": face_count,
        "elapsed_seconds": elapsed_seconds,
        "median_ms": percentile(total_times, 50),
        "p95_ms": percentile(total_times, 95),
        "detection_median_ms": percentile(detection_times[warmup_frames:], 50),
        "preprocess_median_ms": percentile(preprocess_times[warmup_frames:], 50),
        "inference_median_ms": percentile(inference_times[warmup_frames:], 50),
        "pipeline_fps": measured_count / elapsed_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark toan bo pipeline webcam L3.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model", default="face_cnn.pth")
    parser.add_argument("--labels", default="label_names.txt")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--csv", default="benchmark_pipeline_results.csv")
    args = parser.parse_args()

    if args.seconds <= 0 or args.warmup_frames < 0:
        parser.error("seconds phai > 0 va warmup-frames phai >= 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA khong kha dung trong moi truong hien tai")
    if not os.path.exists(args.model) or not os.path.exists(args.labels):
        raise FileNotFoundError("Thieu model hoac label file")

    with open(args.labels, encoding="utf-8") as file:
        label_names = [line.strip() for line in file if line.strip()]
    model = FaceCNN(num_classes=len(label_names))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    result = benchmark_pipeline(
        model,
        torch.device(args.device),
        args.camera,
        args.seconds,
        args.warmup_frames,
        args.display,
    )

    fields = list(result.keys())
    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)

    print(
        f"L3 {result['device']} | frames={result['frames']} | faces={result['faces']} | "
        f"median={result['median_ms']:.3f} ms | p95={result['p95_ms']:.3f} ms | "
        f"FPS={result['pipeline_fps']:.3f}"
    )
    print(
        f"components median | detection={result['detection_median_ms']:.3f} ms | "
        f"preprocess={result['preprocess_median_ms']:.3f} ms | "
        f"inference={result['inference_median_ms']:.3f} ms"
    )
    print(f"Da luu ket qua vao {args.csv}")


if __name__ == "__main__":
    main()
