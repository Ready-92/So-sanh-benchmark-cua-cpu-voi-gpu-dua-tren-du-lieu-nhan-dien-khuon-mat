# step5_demo_video.py
import cv2
import torch
import numpy as np
import argparse
import time
from face_utils import detect_faces, preprocess_face
from step3_train import FaceCNN

def run_demo(confidence_threshold=0.6):
    with open('label_names.txt') as f:
        label_names = [line.strip() for line in f]

    model_cpu = FaceCNN(num_classes=len(label_names))
    model_cpu.load_state_dict(torch.load('face_cnn.pth', map_location='cpu', weights_only=True))
    model_cpu.eval()

    model_gpu = None
    if torch.cuda.is_available():
        model_gpu = FaceCNN(num_classes=len(label_names)).cuda()
        model_gpu.load_state_dict(torch.load('face_cnn.pth', map_location='cuda', weights_only=True))
        model_gpu.eval()

    cpu_fps = 0.0
    gpu_fps = 0.0

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        cpu_start = time.perf_counter()
        gpu_elapsed = 0.0
        for (x, y, w, h) in faces:
            face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face_input = preprocess_face(face_gray, size=64)
            face_t_cpu = torch.tensor(face_input[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                cpu_out = model_cpu(face_t_cpu)
                cpu_prob = torch.softmax(cpu_out, dim=1)
                cpu_conf, cpu_pred = cpu_prob.max(dim=1)

            if model_gpu is not None:
                face_t_gpu = face_t_cpu.cuda(non_blocking=True)
                torch.cuda.synchronize()
                gpu_start = time.perf_counter()
                with torch.no_grad():
                    gpu_out = model_gpu(face_t_gpu)
                    gpu_prob = torch.softmax(gpu_out, dim=1)
                    gpu_conf, gpu_pred = gpu_prob.max(dim=1)
                torch.cuda.synchronize()
                gpu_elapsed += time.perf_counter() - gpu_start

        cpu_elapsed = time.perf_counter() - cpu_start
        if faces:
            cpu_fps = len(faces) / cpu_elapsed if cpu_elapsed > 0 else 0.0
        if model_gpu is not None:
            if faces:
                gpu_fps = len(faces) / gpu_elapsed if gpu_elapsed > 0 else 0.0

        for (x, y, w, h) in faces:
            name = label_names[cpu_pred.item()] if cpu_conf.item() > confidence_threshold else "Unknown"
            gpu_name = label_names[gpu_pred.item()] if model_gpu is not None and gpu_conf.item() > confidence_threshold else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"CPU: {name} ({cpu_conf.item():.2f})", (x, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"GPU: {gpu_name}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"CPU CNN FPS: {cpu_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        gpu_text = f"GPU CNN FPS: {gpu_fps:.1f}" if model_gpu is not None else "GPU: unavailable"
        cv2.putText(frame, gpu_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.imshow('Nhan dien - Nhan q de thoat', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Không hard-code 0.6 nữa, có thể chỉnh khi chạy: python step5_demo_video.py --threshold 0.7
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()
    run_demo(confidence_threshold=args.threshold)
    