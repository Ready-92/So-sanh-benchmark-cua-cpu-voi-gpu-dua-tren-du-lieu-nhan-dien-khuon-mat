# README V3 — Tài liệu thuyết trình môn Tính toán song song (bản mở rộng)

> Dựa trên số liệu đo thật của project (24/08/2026, file `BENCHMARK_RESULTS_FULL.md`, `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv`).
> Tài liệu này trả lời đầy đủ: hệ thống cần show gì, giá trị thực tế, lợi ích thuật toán tối ưu, phần nào tuần tự / song song, ý nghĩa so sánh CPU vs GPU, dàn 10 slide, và câu hỏi dự kiến khi bảo vệ.

---

## 1. Kiến trúc hệ thống — chi tiết từng bước

**Pipeline nhận diện khuôn mặt = chuỗi 5 bước.** Mỗi bước chạy ở đâu và bản chất song song khác nhau:

| # | Bước | Code | Chạy trên | Tuần tự / Song song | Chi phí |
|---|------|------|-----------|---------------------|---------|
| 1 | Thu ảnh | `cv2.VideoCapture.read()` | CPU (driver webcam) | **Tuần tự** — I/O theo chu kỳ khung hình, không tách được | ~5–10 ms/frame |
| 2 | Detect mặt | Haar cascade / YuNet | CPU | Song song mức SIMD, nhưng **quét tuần tự** qua image pyramid (từng scale một) | ~10–20 ms |
| 3 | Tiền xử lý | grayscale → crop → resize 64×64 | CPU | **Tuần tự** từng ảnh | ~1 ms |
| 4 | **CNN inference** | conv 32→64→128 + FC | **CPU (OpenMP) / GPU (CUDA+cuDNN)** | **SONG SONG chính** — trọng tâm toàn bộ đề tài | 0.4–0.7 ms |
| 5 | Hiển thị | `imshow`, vẽ box | CPU | Tuần tự (đồng bộ với chu kỳ màn hình) | ~5–15 ms |

> **Điểm mấu chốt khi trình bày**: chỉ bước 4 là tính toán song song thật sự; bước 1–3–5 là phần tuần tự (sequential fraction) — chính là giới hạn theo **Định luật Amdahl**.
>
> **Câu nói "ăn điểm"**: *"Song song hóa không làm nhanh mọi thứ — nó chỉ làm nhanh phần tính toán. Phần còn lại của pipeline quyết định tốc độ thực tế của hệ thống."*

---

## 2. Vì sao CNN inference là "điểm nóng" — lượng tính toán thật

Model `FaceCNN` (ảnh grayscale 1×64×64):

| Lớp | Kích thước đầu ra | MACs (phép nhân-tích lũy) |
|---|---:|---:|
| Conv1: 3×3, 1→32 kênh | 64×64×32 | 64·64·32·9 = 1.18 M |
| Conv2: 3×3, 32→64 kênh | 32×32×64 | 32·32·64·32·9 = 18.87 M |
| Conv3: 3×3, 64→128 kênh | 16×16×128 | 16·16·128·64·9 = 18.87 M |
| FC1: 8192→256 | — | 2.10 M |
| FC2: 256→2 | — | 0.5 K |

**Tổng ≈ 41 triệu MACs ≈ 82 MFLOPs mỗi ảnh** (~0.08 GFLOPs).

Điểm quan trọng cho môn song song: **41 triệu phép tính này độc lập với nhau theo pixel và theo kênh** → chia được cho hàng nghìn nhân. Đây là cơ sở lý thuyết của việc chọn GPU.

---

## 3. Nền tảng song song sử dụng

### CPU — i5-14400F (10 nhân / 16 luồng)
- PyTorch dùng **OpenMP + MKL-DNN**: chia phép toán thành các task chạy trên nhiều luồng.
- Mỗi nhân còn có **SIMD (AVX2)** — 1 lệnh xử lý nhiều dữ liệu cùng lúc.

### GPU — RTX 3060 (3584 CUDA core)
- Mô hình **SIMT**: hàng nghìn thread chạy cùng một lệnh trên dữ liệu khác nhau.
- Tổ chức: thread → **warp (32 thread)** → block → grid.
- **cuDNN**: thư viện tối ưu sẵn kernel convolution, tự chọn kernel nhanh nhất (autotuning).

### So sánh bản chất
| | CPU | GPU |
|---|---|---|
| Đơn vị | 10 nhân mạnh, tần số cao (4.7 GHz) | 3584 core nhỏ, tần số thấp |
| Chiến lược | Vài luồng lớn, ưu tiên latency | Hàng nghìn luồng, ưu tiên throughput |
| Phù hợp | Phần tuần tự + điều khiển | Phần tính toán lặp lại hàng loạt |

---

## 4. Phương pháp đo benchmark — vì sao số liệu tin được

- **Warm-up 20 lần** trước khi đo (loại bỏ khởi tạo kernel, cache lạnh).
- **Đo 100 lần**, báo **median + P95** thay vì trung bình (chống nhiễu do hệ điều hành).
- `torch.cuda.synchronize()` trước khi bấm giờ — vì GPU chạy **bất đồng bộ** (không đồng bộ thì đo ra số 0 giả).
- **Lặp lại 3 lượt độc lập** → hệ số biến thiên CV 0.4–4.1% (bảng mục 11 trong `BENCHMARK_RESULTS_FULL.md`).
- Chỉ đo bước CNN inference, không tính webcam/Haar/hiển thị → so sánh công bằng đúng phần song song.

> **Điểm học thuật**: đo đạc đúng là một nửa của môn song song. Kernel không đồng bộ là lỗi đo kinh điển mà ai cũng gặp.

---

## 5. Kết quả đo thật — batch size 1

| Thiết bị | Median (ms) | P95 (ms) | FPS |
|---|---:|---:|---:|
| i5-14400F (CPU) | 0.725 | 0.913 | ~1330 |
| RTX 3060 (GPU) | 0.363 | 0.483 | ~2600 |

→ **GPU nhanh hơn ~2.0x** ở batch 1, model nhỏ.

### So sánh nâng cấp phần cứng (scaling phần cứng)

| | Cũ | Mới | Cải thiện |
|---|---:|---:|---:|
| GPU | GTX 1650: 0.616 ms | RTX 3060: 0.363 ms | **1.70x** |
| CPU | i5-10300F: 1.551 ms | i5-14400F: 0.725 ms | **2.14x** |

---

## 6. Kết quả đo thật — theo batch size (dữ liệu `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv`)

| Batch | CPU ms/ảnh | GPU ms/ảnh | **Speedup GPU/CPU** |
|---:|---:|---:|---:|
| 1 | 0.712 | 0.361 | **2.0x** |
| 4 | 0.477 | 0.128 | **3.7x** |
| 8 | 0.468 | 0.052 | **9.0x** |
| 16 | 0.433 | 0.038 | **11.4x** |
| 32 | 0.464 | 0.034 | **13.6x** |
| 128 | 0.477 | 0.028 | **17.0x** |

Hai điều quan trọng rút ra:

1. **GPU**: ms/ảnh giảm 0.361 → 0.028 (12.9x) khi tăng batch → overhead được **chia đều (amortize)** cho nhiều ảnh.
2. **CPU**: ms/ảnh gần như không đổi (~0.46) → CPU đã song song từ đầu, batching không giúp thêm.

→ Đây chính là minh họa sống của **Định luật Gustafson**.

---

## 7. Giải thích bằng Định luật Amdahl

Câu hỏi thầy cô chắc chắn hỏi: *"GPU có 3584 nhân, CPU có 10 nhân, sao GPU chỉ nhanh hơn 2 lần?"*

Công thức Amdahl (bài toán có kích thước cố định):

$$
S = \frac{1}{(1-p) + p/N} \qquad \lim_{N \to \infty} S = \frac{1}{1-p}
$$

Tính ngược từ số đo: batch 1 có $S = 2.0$ → $\frac{1}{1-p} \approx 2.0$ → **$p \approx 0.5$**.

**Nghĩa là:** trong 0.361 ms của GPU, chỉ ~50% là tính toán thật, ~50% còn lại là phần tuần tự không parallel hóa được:
- Launch kernel (khởi động lưới thread trên GPU)
- Copy dữ liệu CPU↔GPU (H2D/D2H)
- Overhead Python/PyTorch framework

→ Dù $N = 3584$ nhân, speedup vẫn bị chặn ở ~2x. **Đây là bài học trung tâm của toàn môn học.**

---

## 8. Giải thích bằng Định luật Gustafson

Gustafson trả lời: *"nếu tăng kích thước bài toán, phần song song chiếm ưu thế → speedup tăng."*

Công thức: $S = N - \alpha(N-1)$ (với $\alpha$ = phần tuần tự trong workload mở rộng).

Bằng chứng từ bảng mục 6: batch 1 → 128, speedup **2.0x → 17.0x**, vì overhead tuần tự (launch, copy) được amortize trên 128 ảnh — phần song song (41 triệu MACs × 128) chiếm gần như toàn bộ.

---

## 9. Đo "hiệu suất khai thác phần cứng" — con số ấn tượng cho slide

RTX 3060 đỉnh lý thuyết FP32 ≈ **12.74 TFLOPS**. Tính throughput thực tế từ số đo:

| Batch | FPS đo được | TFLOPs thực tế (FPS × 0.082 GFLOPs) | % đỉnh GPU |
|---:|---:|---:|---:|
| 1 | ~2600 | 0.21 | **1.6%** |
| 128 | ~33000 | 2.71 | **21%** |

→ Batch 1 chỉ dùng 1.6% sức mạnh GPU (nghẽn overhead) → batch 128 dùng 21% (nghẽn dần chuyển sang tính toán). Đây là minh chứng định lượng đẹp cho Amdahl + Gustafson, và cho thấy **vì sao phải batch**.

---

## 10. Các thuật toán tối ưu — chi tiết và ích lợi

| Thuật toán | Cách hoạt động | Ích lợi đo được / dự kiến |
|---|---|---|
| **cuBLAS GEMM** | Biến conv thành nhân ma trận (im2col), tận dụng nhân ma trận song song tối ưu | Nền tảng tốc độ GPU hiện tại |
| **cuDNN autotuning** | Tự thử nhiều kernel và chọn kernel nhanh nhất cho từng kích thước tensor | Tránh phải tối ưu tay |
| **Batch processing** | N ảnh trong 1 lần gọi = thêm chiều song song dữ liệu | **12.9x** ms/ảnh (đo thật, batch 1→128) |
| **CUDA Streams** | Nhiều luồng lệnh chạy đồng thời, overlap tính toán với copy dữ liệu | Che giấu phần tuần tự |
| **TensorRT + FP16** | Fusion kernel + dùng tensor core nửa chính xác | Dự kiến nhanh thêm ~2x |
| **OpenCV DNN CUDA** | Bật GPU cho YuNet+SFace bằng 2 dòng `setPreferableBackend` | Pipeline mới chạy trên GPU |

> **Điểm học thuật**: không viết kernel CUDA tay (`inference_template.cu`) để chạy production — chỉ để học. Thư viện tối ưu sẵn luôn tốt hơn kernel tay chưa được tinh chỉnh sâu.

---

## 11. Dàn 10 slide + nội dung nói từng slide

**Slide 1 — Giới thiệu đề tài**
- Nhận diện khuôn mặt realtime bằng CNN; câu hỏi nghiên cứu: tối ưu bằng tính toán song song CPU/GPU.
- Phạm vi: tập trung vào bước CNN inference — phần duy nhất song song hóa được.

**Slide 2 — Kiến trúc pipeline 5 bước**
- Vẽ sơ đồ 5 bước, tô đỏ phần tuần tự (thu ảnh, tiền xử lý, hiển thị), tô xanh phần song song (CNN).
- Nói: "chỉ 1/5 pipeline là song song — phần còn lại quyết định trần tốc độ".

**Slide 3 — Vì sao CNN là điểm nóng**
- Bảng 41 triệu MACs/ảnh; các phép tính độc lập → chia được cho nghìn nhân.

**Slide 4 — Nền tảng song song**
- CPU OpenMP + AVX2 vs GPU CUDA SIMT; bảng so sánh 10 nhân mạnh vs 3584 core nhỏ.

**Slide 5 — Phương pháp đo**
- Warm-up, median/P95, synchronize, 3 lượt lặp CV < 5%; vì sao phải synchronize (GPU bất đồng bộ).

**Slide 6 — Kết quả batch 1**
- CPU 0.725ms vs GPU 0.363ms → 2.0x; kèm bảng nâng cấp phần cứng 1.7x/2.1x.

**Slide 7 — Amdahl**
- Công thức + tính ngược $p \approx 0.5$ từ số đo; kể tên overhead tuần tự.

**Slide 8 — Gustafson + batch**
- Bảng batch 1→128: speedup 2.0x → 17.0x; GPU ms/ảnh giảm 12.9x, CPU không đổi.

**Slide 9 — Thuật toán tối ưu**
- cuDNN/GEMM, batch, streams, TensorRT/FP16; kèm con số 1.6% → 21% peak GPU.

**Slide 10 — Giá trị mang lại + kết luận**
- Kỹ thuật: realtime 0.4ms/ảnh, 2600 FPS.
- Học thuật: đo đúng, hiểu Amdahl/Gustafson, chọn đúng công cụ.
- Hướng phát triển: TensorRT, model lớn, nhiều người cùng lúc.

---

## 12. Câu hỏi dự kiến khi bảo vệ + cách trả lời

**Q1. Sao không thấy speedup 300x dù GPU nhiều nhân?**
→ Amdahl: batch 1 có ~50% overhead tuần tự → speedup chặn ở ~2x (mục 7).

**Q2. Vậy GPU có đáng dùng không?**
→ Có, khi workload lớn: batch 128 cho 17x, và GPU là con đường duy nhất khi model lớn hơn (Gustafson, mục 8).

**Q3. CPU của bạn cũng 1330 FPS — cần gì GPU?**
→ Ở model nhỏ batch 1 thì CPU đủ; nhưng pipeline thực tế có thêm nhiều người (batch tăng), model lớn hơn → GPU phát huy. Nêu số 1.6% vs 21% peak (mục 9).

**Q4. Vì sao dùng median mà không dùng trung bình?**
→ Chống nhiễu hệ điều hành; P95 cho thấy đuôi phân phối — chuẩn benchmark công nghiệp.

**Q5. Phần tuần tự trong pipeline có song song hóa được không?**
→ Được một phần: CUDA Streams overlap copy+compute; detect có thể đưa lên GPU (YuNet qua OpenCV DNN CUDA) — đang là hướng phát triển.

**Q6. TensorRT khác gì cuDNN?**
→ cuDNN chỉ là thư viện kernel; TensorRT tối ưu toàn mô hình (fusion lớp, chọn precision FP16/INT8, build engine riêng cho phần cứng).

---

## 13. Checklist trước buổi thuyết trình

- [ ] Chạy lại `torch.cuda.is_available()` và xác nhận môi trường (version OpenCV/PyTorch khớp số liệu)
- [ ] Chạy `benchmark_batch.py` một lần để có số demo tươi (nếu máy đổi trạng thái)
- [ ] Mở sẵn `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv` trên màn hình phụ
- [ ] Demo live: `python step5_demo_video.py` (nhấn `q` thoát) — thử trước với người thật
- [ ] Chuẩn bị 2 công thức Amdahl + Gustafson trên slide (không học thuộc chữ)

---

## 14. File dữ liệu tham chiếu

| File | Nội dung |
|------|----------|
| `BENCHMARK_RESULTS_FULL.md` | Báo cáo benchmark đầy đủ (24/08/2026) |
| `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv` | Số liệu batch 1→128 (CPU vs GPU) |
| `benchmark_cpu_gpu.py` | So sánh CPU vs GPU batch 1 |
| `benchmark_batch.py` | Script đo theo batch size |
| `step4_export_cuda.py`, `inference_template.cu` | Kernel CUDA viết tay (phục vụ học tập) |
| `models/` | ONNX YuNet + SFace cho pipeline mới |
