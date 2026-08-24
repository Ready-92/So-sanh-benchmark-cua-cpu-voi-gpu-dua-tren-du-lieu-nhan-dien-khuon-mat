import re
import subprocess

import numpy as np
import torch

from step3_train import FaceCNN


result = subprocess.run(
    [".\\inference_cuda.exe"],
    check=True,
    capture_output=True,
    text=True,
)
match = re.search(r"Logits:([^\n]+)", result.stdout)
if not match:
    raise RuntimeError(f"Khong tim thay logits trong output CUDA:\n{result.stdout}")
cuda_logits = np.fromstring(match.group(1), sep=" ", dtype=np.float32)

with open("label_names.txt", encoding="utf-8") as file:
    label_names = [line.strip() for line in file if line.strip()]
model = FaceCNN(num_classes=len(label_names))
model.load_state_dict(torch.load("face_cnn.pth", map_location="cpu", weights_only=True))
model.eval()

images = np.fromfile("output/val_images.bin", dtype=np.float32).reshape(-1, 1, 64, 64)
with torch.inference_mode():
    pytorch_logits = model(torch.from_numpy(images[:1])).numpy()[0]

if cuda_logits.shape != pytorch_logits.shape:
    raise RuntimeError(
        f"So logits khong khop: CUDA={cuda_logits.size}, PyTorch={pytorch_logits.size}"
    )

max_abs_error = np.max(np.abs(cuda_logits - pytorch_logits))
cuda_prediction = int(np.argmax(cuda_logits))
pytorch_prediction = int(np.argmax(pytorch_logits))
print(f"CUDA logits:    {cuda_logits[0]:.8f} {cuda_logits[1]:.8f}")
print(f"PyTorch logits: {pytorch_logits[0]:.8f} {pytorch_logits[1]:.8f}")
print(f"Max abs error: {max_abs_error:.8e}")
print(f"CUDA prediction: {cuda_prediction} ({label_names[cuda_prediction]})")
print(f"PyTorch prediction: {pytorch_prediction} ({label_names[pytorch_prediction]})")
if max_abs_error > 1e-3 or cuda_prediction != pytorch_prediction:
    raise SystemExit("FAIL: CUDA va PyTorch khong khop trong nguong.")
print("PASS: CUDA va PyTorch khop trong nguong 1e-3.")
