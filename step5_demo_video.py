# step5_demo_video.py
import cv2
import torch
import numpy as np
import argparse
from face_utils import detect_faces, preprocess_face
from step3_train import FaceCNN

def run_demo(confidence_threshold=0.6):
    with open('label_names.txt') as f:
        label_names = [line.strip() for line in f]

    model = FaceCNN(num_classes=len(label_names))
    model.load_state_dict(torch.load('face_cnn.pth', map_location='cpu', weights_only=True))
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face_input = preprocess_face(face_gray, size=64)
            face_t = torch.tensor(face_input[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                out = model(face_t)
                prob = torch.softmax(out, dim=1)
                conf, pred = prob.max(dim=1)
            name = label_names[pred.item()] if conf.item() > confidence_threshold else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf.item():.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
    