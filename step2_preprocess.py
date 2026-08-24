# step2_preprocess.py
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from face_utils import preprocess_face, augment_face

def load_dataset(dataset_dir='dataset', size=64, use_augmentation=True):
    images, labels, label_names = [], [], []

    # Lỗi #3: chỉ lấy các mục là THƯ MỤC, bỏ qua file lẻ nằm lẫn trong dataset/
    student_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not student_dirs:
        raise RuntimeError(f"Khong tim thay thu muc hoc sinh nao trong '{dataset_dir}'")

    skipped = 0
    for idx, student in enumerate(student_dirs):
        img_paths = glob.glob(f'{dataset_dir}/{student}/*.jpg')
        if not img_paths:
            print(f"[Canh bao] {student} khong co anh nao, se bi bo qua khi train.")
            continue

        label_names.append(student)
        label = len(label_names) - 1

        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Lỗi #3: kiểm tra ảnh đọc được hay không trước khi xử lý tiếp
            if img is None:
                print(f"[Canh bao] Anh loi, bo qua: {img_path}")
                skipped += 1
                continue

            if use_augmentation:
                for variant in augment_face(img):
                    images.append(preprocess_face(variant, size))
                    labels.append(label)
            else:
                images.append(preprocess_face(img, size))
                labels.append(label)

    if skipped:
        print(f"Tong so anh bi bo qua vi loi: {skipped}")

    return np.array(images), np.array(labels), label_names

if __name__ == '__main__':
    X, y, label_names = load_dataset(use_augmentation=True)
    print("Tong so anh (sau augmentation):", X.shape)
    print("So hoc sinh:", len(label_names))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train:", X_train.shape, "Val:", X_val.shape)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    with open('label_names.txt', 'w') as f:
        for name in label_names:
            f.write(name + '\n')
    print("Da luu xong file .npy va label_names.txt")