# step1_collect_data.py
import argparse
import glob
import cv2
import os
import time
from face_utils import face_cascade, normalize_crop

POSE_HINTS = [
    "Nhin thang vao camera",
    "Quay dau sang trai nhe",
    "Quay dau sang phai nhe",
    "Ngua dau len nhe",
    "Cui dau xuong nhe",
]

def collect_faces(student_id, num_images=30, save_dir='data', capture_interval=0.3):
    os.makedirs(f'{save_dir}/{student_id}', exist_ok=True)
    start_index = len(glob.glob(f'{save_dir}/{student_id}/*.jpg'))
    cap = cv2.VideoCapture(0)
    count = 0
    last_capture = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        distance_phase = count // max(1, num_images // 3)
        distance_hints = ["Ngoi GAN camera", "Ngoi VUA (binh thuong)", "Ngoi XA camera"]
        distance_hint = distance_hints[min(distance_phase, 2)]
        pose_hint = POSE_HINTS[count % len(POSE_HINTS)]
        cv2.putText(frame, f"{distance_hint} | {pose_hint}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Da chup: {count}/{num_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        if len(faces) > 0:
            # Chi chup khuon mat LON NHAT (tranh chup nham nguoi dung sau), bo qua mat qua nho
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Chỉ chụp nếu đã cách lần chụp trước ít nhất capture_interval giây
            # -> tránh 30 frame liên tiếp giống hệt nhau (lỗi #2)
            if (now - last_capture >= capture_interval and w >= 100 and h >= 100):
                face_img = normalize_crop(gray, (x, y, w, h))
                cv2.imwrite(f'{save_dir}/{student_id}/{start_index + count}.jpg', face_img)
                count += 1
                last_capture = now

        cv2.imshow('Collecting - Nhan q de dung', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Da thu thap {count} anh cho {student_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Thu thap anh khuon mat tu webcam.")
    parser.add_argument('--student', default=None, help="Ma hoc sinh, vd: student_01")
    parser.add_argument('--num', type=int, default=30, help="So anh can thu thap")
    parser.add_argument('--save-dir', default='data', help="Thu muc luu anh")
    parser.add_argument('--interval', type=float, default=0.3, help="Khoang cach giua 2 lan chup (giay)")
    args = parser.parse_args()
    student_id = args.student or input("Nhap ma hoc sinh (vd: student_01): ")
    collect_faces(student_id, num_images=args.num, save_dir=args.save_dir, capture_interval=args.interval)