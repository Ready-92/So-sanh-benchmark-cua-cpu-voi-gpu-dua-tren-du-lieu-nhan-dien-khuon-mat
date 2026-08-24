# step1_collect_data.py
import cv2
import os
import time
from face_utils import face_cascade

POSE_HINTS = [
    "Nhin thang vao camera",
    "Quay dau sang trai nhe",
    "Quay dau sang phai nhe",
    "Ngua dau len nhe",
    "Cui dau xuong nhe",
]

def collect_faces(student_id, num_images=30, save_dir='dataset', capture_interval=0.3):
    os.makedirs(f'{save_dir}/{student_id}', exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    last_capture = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        hint = POSE_HINTS[count // 6 % len(POSE_HINTS)]
        cv2.putText(frame, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Da chup: {count}/{num_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Chỉ chụp nếu đã cách lần chụp trước ít nhất capture_interval giây
            # -> tránh 30 frame liên tiếp giống hệt nhau (lỗi #2)
            if now - last_capture >= capture_interval:
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f'{save_dir}/{student_id}/{count}.jpg', face_img)
                count += 1
                last_capture = now

        cv2.imshow('Collecting - Nhan q de dung', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Da thu thap {count} anh cho {student_id}")

if __name__ == '__main__':
    student_id = input("Nhap ma hoc sinh (vd: student_01): ")
    collect_faces(student_id)