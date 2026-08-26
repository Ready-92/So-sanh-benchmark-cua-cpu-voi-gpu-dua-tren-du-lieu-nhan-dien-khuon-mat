# step2_preprocess.py
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from face_utils import preprocess_face, augment_face, face_cascade

MAX_IMAGE_SIDE = 600


def crop_face_if_large(image):
    """Anh lon (anh canh goc) -> cat vung khuon mat; anh nho (da crop san) -> giu nguyen."""
    h, w = image.shape[:2]
    if max(h, w) <= MAX_IMAGE_SIDE:
        return image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    return image[y:y + fh, x:x + fw]

def load_dataset(dataset_dir='data', size=64, use_augmentation=True):
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

            img = crop_face_if_large(img)
            if img is None:
                print(f"[Canh bao] Anh lon khong tim thay khuon mat, bo qua: {img_path}")
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


def load_image_paths(dataset_dir='data'):
    image_paths, labels, label_names = [], [], []
    student_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    for student in student_dirs:
        paths = glob.glob(f'{dataset_dir}/{student}/*.jpg')
        if not paths:
            continue
        label_names.append(student)
        label = len(label_names) - 1
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    if not image_paths:
        raise RuntimeError(f"Khong tim thay anh nao trong '{dataset_dir}'")
    return image_paths, np.array(labels), label_names


def preprocess_split(image_paths, labels, size=64, use_augmentation=False):
    images, output_labels = [], []
    for image_path, label in zip(image_paths, labels):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = crop_face_if_large(image)
        if image is None:
            continue
        variants = augment_face(image) if use_augmentation else [image]
        for variant in variants:
            images.append(preprocess_face(variant, size))
            output_labels.append(label)
    return np.array(images), np.array(output_labels)

if __name__ == '__main__':
    image_paths, y, label_names = load_image_paths()
    train_paths, val_paths, y_train_source, y_val_source = train_test_split(
        image_paths, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, y_train = preprocess_split(
        train_paths, y_train_source, use_augmentation=True
    )
    X_val, y_val = preprocess_split(val_paths, y_val_source)
    print("Anh goc:", len(image_paths), "Train sau augmentation:", X_train.shape)
    print("Validation khong augmentation:", X_val.shape)
    print("So hoc sinh:", len(label_names))
    print("Train:", X_train.shape, "Val:", X_val.shape)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    with open('label_names.txt', 'w') as f:
        for name in label_names:
            f.write(name + '\n')
    print("Da luu xong file .npy va label_names.txt")