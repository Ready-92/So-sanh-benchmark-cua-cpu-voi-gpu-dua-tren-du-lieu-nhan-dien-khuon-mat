import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from benchmark_cpu_gpu import load_input, percentile
from step3_train import FaceCNN


def benchmark_batch(model, sample, device, batch_size, warmup, iterations):
    batch = sample.repeat(batch_size, 1, 1, 1).to(device)
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_ms = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000)

    total_fps = batch_size * 1000 / (sum(elapsed_ms) / len(elapsed_ms))
    return {
        "median_ms": percentile(elapsed_ms, 50),
        "p95_ms": percentile(elapsed_ms, 95),
        "batch_fps": total_fps,
        "per_image_ms": percentile(elapsed_ms, 50) / batch_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CNN theo batch size.")
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
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--csv", default="benchmark_results_batch.csv")
    args = parser.parse_args()

    if any(size < 1 for size in args.batch_sizes):
        parser.error("--batch-sizes phai >= 1")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup phai >= 0 va iterations phai >= 1")
    if not os.path.exists(args.model) or not os.path.exists(args.labels):
        raise FileNotFoundError("Thieu model hoac label file.")

    with open(args.labels, encoding="utf-8") as file:
        label_names = [line.strip() for line in file if line.strip()]
    model = FaceCNN(num_classes=len(label_names))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))

    samples = []
    for mode, path in (("grayscale", args.gray_image), ("color", args.color_image)):
        sample, actual_mode = load_input(path)
        if mode == "color" and actual_mode != "color":
            raise ValueError(f"Anh mau khong hop le: {path}")
        samples.append((mode, path, sample))

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for mode, path, sample in samples:
        for batch_size in args.batch_sizes:
            for device in devices:
                metrics = benchmark_batch(
                    model, sample, device, batch_size, args.warmup, args.iterations
                )
                row = {
                    "input": mode,
                    "path": path,
                    "batch_size": batch_size,
                    "device": device.type,
                    "median_batch_ms": f"{metrics['median_ms']:.3f}",
                    "p95_batch_ms": f"{metrics['p95_ms']:.3f}",
                    "per_image_ms": f"{metrics['per_image_ms']:.3f}",
                    "batch_fps": f"{metrics['batch_fps']:.3f}",
                }
                rows.append(row)
                print(
                    f"{mode:9} batch={batch_size:3} {device.type:4} | "
                    f"median={row['median_batch_ms']} ms | "
                    f"per-image={row['per_image_ms']} ms | FPS={row['batch_fps']}"
                )

    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Da luu ket qua vao {args.csv}")


if __name__ == "__main__":
    main()
