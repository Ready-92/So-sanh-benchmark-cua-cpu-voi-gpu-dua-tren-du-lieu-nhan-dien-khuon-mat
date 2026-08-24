# step4_export_cuda.py
import numpy as np
import torch
from step3_train import FaceCNN
import os

os.makedirs('output', exist_ok=True)

if __name__ == '__main__':
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    with open('label_names.txt') as f:
        label_names = [line.strip() for line in f]

    X_train.astype('float32').tofile('output/train_images.bin')
    y_train.astype('int32').tofile('output/train_labels.bin')
    X_val.astype('float32').tofile('output/val_images.bin')
    y_val.astype('int32').tofile('output/val_labels.bin')

    with open('output/label_names.txt', 'w') as f:
        for name in label_names:
            f.write(name + '\n')

    with open('output/meta.txt', 'w') as f:
        f.write(f"N_train={X_train.shape[0]}\n")
        f.write(f"N_val={X_val.shape[0]}\n")
        f.write("channels=1\nheight=64\nwidth=64\n")
        f.write(f"num_classes={len(label_names)}\n")

    # Lỗi #5: khai báo rõ map_location='cpu' và weights_only=True
    model = FaceCNN(num_classes=len(label_names))
    model.load_state_dict(
        torch.load('face_cnn.pth', map_location='cpu', weights_only=True)
    )

    # Ghi kèm shape của từng layer để bên CUDA biết cách reshape khi đọc .bin
    with open('output/weights_shapes.txt', 'w') as f:
        for name, param in model.state_dict().items():
            arr = param.numpy().astype('float32')
            arr.tofile(f'output/weights_{name}.bin')
            f.write(f"{name} {list(arr.shape)}\n")

    print("Da xuat xong toan bo du lieu + trong so vao thu muc output/")
    print("Xem output/weights_shapes.txt de biet shape tung layer khi nap ben CUDA")