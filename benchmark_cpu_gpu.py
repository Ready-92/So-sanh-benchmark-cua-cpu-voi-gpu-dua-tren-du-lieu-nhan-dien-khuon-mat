import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from face_utils import preprocess_face
from step3_train import FaceCNN


def load_input(path, size=64):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")

    source_mode = "grayscale" if image.ndim == 2 else "color"
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Giữ nguyên pipeline của model: mọi ảnh đều trở thành 1x64x64.
    tensor = torch.from_numpy(preprocess_face(image, size=size)).unsqueeze(0)
    return tensor, source_mode


def percentile(values, value):
    return float(np.percentile(np.asarray(values), value))


def benchmark(model, sample, device, warmup, iterations):
    sample = sample.to(device)
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_ms = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": percentile(elapsed_ms, 50),
        "p95_ms": percentile(elapsed_ms, 95),
        "throughput_fps": 1000 / (sum(elapsed_ms) / len(elapsed_ms)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="So sanh benchmark CNN cho anh grayscale va anh mau tren CPU/GPU."
    )
    parser.add_argument("--gray-image", default="data/student_01/0.jpg")
    parser.add_argument(
        "--color-image",
        default=(
            "data/student_2/"
            "1787208972081_1400025893745852164_g6068070637985194847_"
            "887552051926f2595ed7d6380c2d295e.jpg"
        ),
    )
    parser.add_argument("--model", default="face_cnn.pth")
    parser.add_argument("--labels", default="label_names.txt")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--csv", default="benchmark_results.csv")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup phai >= 0 va --iterations phai >= 1")
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Chua co {args.model}. Hay chay step2_preprocess.py va step3_train.py truoc."
        )
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Chua co file nhan: {args.labels}")

    with open(args.labels, encoding="utf-8") as file:
        label_names = [line.strip() for line in file if line.strip()]
    model = FaceCNN(num_classes=len(label_names))
    model.load_state_dict(
        torch.load(args.model, map_location="cpu", weights_only=True)
    )

    inputs = [
        ("grayscale", args.gray_image),
        ("color", args.color_image),
    ]
    samples = []
    for expected_mode, path in inputs:
        sample, actual_mode = load_input(path)
        if expected_mode == "color" and actual_mode != "color":
            print(f"[Canh bao] {path} khong phai anh mau; anh van duoc doc nhu grayscale.")
        samples.append((expected_mode, path, sample))

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("[Thong tin] CUDA khong kha dung; chi benchmark CPU.")

    rows = []
    for mode, path, sample in samples:
        for device in devices:
            metrics = benchmark(
                model, sample, device, args.warmup, args.iterations
            )
            row = {
                "input": mode,
                "path": path,
                "device": device.type,
                "median_ms": f"{metrics['median_ms']:.3f}",
                "p95_ms": f"{metrics['p95_ms']:.3f}",
                "throughput_fps": f"{metrics['throughput_fps']:.3f}",
            }
            rows.append(row)
            print(
                f"{mode:9} {device.type:4} | median={row['median_ms']} ms | "
                f"p95={row['p95_ms']} ms | FPS={row['throughput_fps']}"
            )

    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Da luu ket qua vao {args.csv}")


if __name__ == "__main__":
    main()