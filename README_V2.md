# README V2 — Tối ưu thuật toán nhận diện khuôn mặt với tính toán song song

> Dành cho pipeline nhận diện khuôn mặt trên máy **i5-14400F + RTX 3060**.
> Trả lời câu hỏi: *"Muốn ưu tiên thuật toán chạy nhanh hơn, liên quan đến tính toán song song, nên dùng thuật toán nào tối ưu nhất?"*

## 1. Xếp hạng phương án theo tốc độ GPU (RTX 3060)

| Hạng | Phương án | Ghi chú |
|------|-----------|---------|
| 1 | **TensorRT (FP16/INT8)** | Nhanh nhất trên GPU NVIDIA: fusion kernel, tensor core, bỏ hết overhead framework. Chuyển từ ONNX → TensorRT engine |
| 2 | **ONNX Runtime (CUDA EP / TensorRT EP)** | Gần bằng TensorRT, dễ dùng hơn nhiều, có cả Execution Provider CPU/GPU |
| 3 | **cuDNN/cuBLAS qua PyTorch CUDA** | Model nhỏ thì launch overhead + Python chiếm nhiều, nhưng batch lớn vẫn rất nhanh |
| 4 | **CUDA viết tay (kiểu `inference_template.cu`)** | Chỉ nên dùng để **học CUDA** — thực tế luôn chậm hơn cuDNN trừ khi tối ưu cực kỳ kỹ (shared memory, tensor core, warp shuffle) |
| 5 | **CPU thuần** | Model nhỏ + ảnh lẻ thì đôi khi nhanh hơn GPU vì không tốn launch overhead |

## 2. Nguyên tắc quan trọng cho tính song song

1. **Batch càng lớn càng lợi** — xử lý N khuôn mặt trong 1 lần gọi (batch) thay vì N lần gọi riêng. Đây là yếu tố quyết định, hơn cả việc chọn framework.
2. **FP16 (half precision)** — RTX 3060 có tensor core, FP16 nhanh gấp ~2x FP32.
3. **CUDA Streams** — chạy nhiều tác vụ song song (detect + recognize) trên các stream khác nhau để overlap.
4. **Đừng viết kernel tay** — trừ khi mục tiêu là học; cuDNN/TensorRT đã tối ưu sẵn theo kiến trúc GPU.

## 3. Khuyến nghị cụ thể cho pipeline hiện tại

Pipeline hiện tại dùng **YuNet (detect) + SFace (recognize)** qua OpenCV DNN. Bật GPU chỉ cần 2 dòng:

```python
net = cv2.dnn.readNetFromONNX("model.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

Muốn nhanh hơn nữa: convert ONNX → **TensorRT engine (FP16)** và chạy qua ONNX Runtime.

## 4. Benchmark công bằng ("thuật toán tối ưu" thay vì code tay vs PyTorch)

- **CPU**: ONNX Runtime (oneDNN, AVX2, multi-thread) là đại diện tối ưu nhất.
- **GPU**: TensorRT FP16 là đại diện tối ưu nhất.
- So sánh 2 thứ đó với **cùng batch size**, ghi rõ **warm-up** và latency **P50/P99**.

## 5. File liên quan trong project

| File | Mục đích |
|------|----------|
| `benchmark_cpu_gpu.py` | So sánh CPU vs GPU |
| `benchmark_pipeline.py` | Benchmark pipeline end-to-end |
| `step4_export_cuda.py`, `inference_template.cu` | Xuất model sang CUDA (phục vụ học tập) |
| `models/` | Model ONNX: `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx` |
| `step5_demo_video.py` | Demo nhận diện webcam (YuNet + SFace + centroid) |
