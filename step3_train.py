# step3_train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

class FaceCNN(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def to_loader(X, y, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    # Lỗi #4: KHÔNG suy ra num_classes từ y_train nữa, đọc thẳng từ label_names.txt
    with open('label_names.txt') as f:
        label_names = [line.strip() for line in f]
    num_classes = len(label_names)
    print(f"So lop (num_classes) = {num_classes} (lay tu label_names.txt)")

    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Dang chay tren:", device)

    model = FaceCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience, patience_counter = 5, 0  # early stopping đơn giản

    for epoch in range(30):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b).argmax(dim=1)
                correct += (pred == y_b).sum().item()
                total += y_b.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.4f}")

        # Lưu model tốt nhất theo val accuracy, thay vì luôn lưu epoch cuối
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'face_cnn.pth')
            patience_counter = 0
            print(f"  -> Model moi tot hon, da luu (acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Dung som (early stopping) o epoch {epoch+1}, best acc = {best_val_acc:.4f}")
                break

    print(f"\nBest Val Accuracy = {best_val_acc:.4f}, model da luu vao face_cnn.pth")

    # Benchmark tốc độ CPU
    model_cpu = FaceCNN(num_classes=num_classes)
    model_cpu.load_state_dict(torch.load('face_cnn.pth', map_location='cpu', weights_only=True))
    model_cpu.eval()
    X_sample = torch.tensor(X_val[:1], dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model_cpu(X_sample)
    elapsed = time.time() - start
    print(f"CPU inference: {elapsed/100*1000:.2f} ms/anh")