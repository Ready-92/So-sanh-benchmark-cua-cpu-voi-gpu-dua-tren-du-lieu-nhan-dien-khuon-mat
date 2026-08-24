# Nhận diện khuôn mặt và benchmark CPU/GPU

Dự án nhận diện học sinh từ webcam/ảnh, sau đó đo và so sánh tốc độ suy luận trên CPU và GPU. Pipeline hiện tại dùng Haar Cascade để phát hiện khuôn mặt và CNN PyTorch để phân loại học sinh.

## 1. Mục tiêu

- Nhận diện khuôn mặt học sinh từ webcam/video.
- Trả về mã học sinh hoặc `Unknown` khi confidence thấp.
- So sánh hiệu năng CPU và GPU ở bước CNN inference.
- Đối chiếu ảnh grayscale và ảnh màu.
- Làm nền tảng để mở rộng sang CUDA inference đầy đủ và benchmark theo batch.

Theo dàn ý, mục tiêu cuối là đánh giá khi số khuôn mặt hoặc batch thay đổi: GPU có thể có lợi rõ hơn khi lượng dữ liệu song song đủ lớn, trong khi toàn bộ pipeline video vẫn chịu ảnh hưởng của webcam, Haar Cascade và hiển thị.

## 1.1. Vì sao đề tài phù hợp với Tính toán song song?

CNN có nhiều phép tính độc lập và lặp lại:

- Các vị trí `(x, y)` trong convolution có thể xử lý song song.
- Nhiều output channel của convolution có thể xử lý song song.
- Nhiều ảnh hoặc nhiều khuôn mặt trong cùng batch có thể xử lý song song.
- Các phép nhân-cộng trong fully connected layer có thể ánh xạ lên nhiều thread.

Vì vậy đề tài cho phép minh họa data parallelism trên CPU đa luồng và GPU CUDA. Câu hỏi nghiên cứu chính là:

> Hiệu năng CNN thay đổi như thế nào trên CPU và GPU khi thay đổi batch size, số lượng khuôn mặt, chi phí truyền dữ liệu và phạm vi pipeline được đo?

Các đại lượng cần báo cáo:

```text
Speedup   = T_CPU / T_GPU
Efficiency = Speedup / số đơn vị xử lý
Throughput = số ảnh xử lý / giây
```

## 1.2. Thiết kế thí nghiệm đề xuất

### So sánh thiết bị

Giữ nguyên model, trọng số, dữ liệu, batch size, warm-up và số lần đo khi so sánh:

- CPU Intel Core i5-14400F.
- GPU NVIDIA GeForce GTX 1650.
- GPU NVIDIA GeForce RTX 3060.

### So sánh theo batch size

Nên thử batch size `1, 4, 8, 16, 32, 64, 128`.

- Batch nhỏ thường bị ảnh hưởng bởi kernel launch và thời gian copy.
- Batch lớn cung cấp nhiều data parallelism hơn cho GPU.
- Kết quả cần có median, p95, FPS và speedup.

Ví dụ bảng báo cáo:

| Batch | CPU ms | GPU ms | CPU FPS | GPU FPS | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... | ... |
| 32 | ... | ... | ... | ... | ... |
| 128 | ... | ... | ... | ... | ... |

### Các mức đo pipeline

- `L0`: chỉ đo kernel CNN trên GPU.
- `L1`: model và input đã nằm trên GPU.
- `L2`: gồm truyền H2D, inference và D2H.
- `L3`: webcam, Haar Cascade, crop, preprocess, inference và display.

GPU có thể nhanh hơn nhiều ở `L0` nhưng chỉ cải thiện ít ở `L3`, vì webcam, Haar Cascade và hiển thị vẫn là phần chạy trên CPU. Đây là trường hợp phù hợp để phân tích Amdahl's Law:

$$
S_{total} = \frac{1}{(1-p)+p/S}
$$

Trong đó `p` là tỷ lệ công việc có thể song song hóa và `S` là speedup của phần được tăng tốc.

### CPU đa luồng

Trên i5-14400F, có thể thử `1, 2, 4, 8, 16` threads bằng `torch.set_num_threads(...)`. Thời gian không giảm tuyến tính cũng là một kết quả quan trọng, có thể do overhead quản lý thread, giới hạn bộ nhớ hoặc phần tuần tự.

## 2. Cấu trúc dự án

```text
.
├── data/
│   ├── student_01/              # 90 ảnh grayscale
│   ├── student_2/                # 11 ảnh màu gốc
│   └── benchmark/
│       ├── gray/                 # bản sao ảnh grayscale dùng benchmark
│       └── color/                # bản sao ảnh màu dùng benchmark
├── face_utils.py                 # phát hiện và tiền xử lý khuôn mặt
├── step1_collect_data.py         # thu thập ảnh từ webcam
├── step2_preprocess.py           # resize, normalize, augmentation, split
├── step3_train.py                # định nghĩa và huấn luyện CNN
├── step4_export_cuda.py          # xuất dữ liệu/trọng số sang .bin
├── step5_demo_video.py           # demo nhận diện webcam
├── benchmark_cpu_gpu.py          # benchmark ảnh grayscale/color trên CPU/GPU
├── inference_template.cu         # khung CUDA hiện tại
├── face_cnn.pth                  # model đã huấn luyện
├── label_names.txt               # danh sách lớp của model
├── X_train.npy, y_train.npy      # tập train đã preprocess
├── X_val.npy, y_val.npy          # tập validation đã preprocess
├── benchmark_results.csv         # kết quả CPU trước đó
└── benchmark_results_cuda.csv    # kết quả CPU/GPU mới nhất
```

## 3. Môi trường đã sử dụng

Máy benchmark hiện tại:

- CPU: chưa ghi nhận model chính xác trong log benchmark hiện tại.
- GPU: `NVIDIA GeForce GTX 1650`
- NVIDIA driver: `592.00`
- CUDA driver capability: `13.1`
- Python: `3.13.7`
- PyTorch: `2.11.0+cu128` trong `.venv`
- OpenCV: `4.11.0`

Kích hoạt môi trường trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Hoặc gọi trực tiếp Python:

```powershell
.\.venv\Scripts\python.exe script_name.py
```

Các gói chính:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python==4.11.0.86 numpy scikit-learn
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 4. Dữ liệu hiện tại

Hai archive ban đầu đã được giải nén vào `data/`:

- `student_01`: 90 ảnh grayscale.
- `student_2`: 11 ảnh màu gốc, kích thước mẫu `2560x1924x3`.

Ảnh màu không bị chuyển đổi trong thư mục dữ liệu gốc. Khi đưa vào CNN, `benchmark_cpu_gpu.py` chuyển ảnh màu sang grayscale để phù hợp với kiến trúc model hiện tại có input `1x64x64`.

## 5. Pipeline xử lý

### Bước 1: Thu thập dữ liệu

```powershell
python step1_collect_data.py
```

Nhập mã học sinh, ví dụ `student_01`. Script mở webcam, dùng Haar Cascade phát hiện mặt, crop vùng mặt grayscale và lưu ảnh vào `dataset/<student_id>/`.

### Bước 2: Preprocess và chia dữ liệu

Do dữ liệu hiện tại nằm trong `data/`, chạy:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; from sklearn.model_selection import train_test_split; from step2_preprocess import load_dataset; X,y,n=load_dataset('data', use_augmentation=True); X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42); np.save('X_train.npy',X_train); np.save('y_train.npy',y_train); np.save('X_val.npy',X_val); np.save('y_val.npy',y_val); open('label_names.txt','w',encoding='utf-8').write(''.join(x+'\n' for x in n)); print('train=',X_train.shape,'val=',X_val.shape,'labels=',n)"
```

Các thao tác chính:

- Đọc ảnh JPG trực tiếp trong từng thư mục lớp.
- Bỏ qua thư mục không chứa JPG trực tiếp, nên `data/benchmark` không trở thành một lớp.
- Resize về `64x64`.
- Chuẩn hóa pixel về `[0, 1]`.
- Augmentation: lật ngang, xoay `-10/+10` độ, tăng/giảm sáng.
- Chia train/validation theo tỷ lệ `80/20`.

### Bước 3: Huấn luyện CNN

```powershell
.\.venv\Scripts\python.exe step3_train.py
```

Kiến trúc:

```text
Input: 1x64x64
Conv(1 -> 32) + ReLU + MaxPool
Conv(32 -> 64) + ReLU + MaxPool
Conv(64 -> 128) + ReLU + MaxPool
Flatten: 128x8x8 = 8192
Linear: 8192 -> 256 -> num_classes
```

Model đọc số lớp từ `label_names.txt`, lưu model tốt nhất vào `face_cnn.pth`, dùng early stopping sau 5 epoch không cải thiện.

Model hiện tại:

- Số lớp: 2 (`student_01`, `student_2`)
- Train samples sau augmentation: 484
- Validation samples: 122
- Validation accuracy trong lần train hiện tại: `1.0000`

Accuracy cao cần được diễn giải thận trọng vì dữ liệu ít và các ảnh/biến thể có thể khá giống nhau.

### Bước 4: Xuất dữ liệu cho CUDA

```powershell
.\.venv\Scripts\python.exe step4_export_cuda.py
```

Script tạo thư mục `output/` với:

- Dữ liệu ảnh và nhãn dạng `.bin`.
- `label_names.txt`.
- `meta.txt`.
- File trọng số từng layer dạng `.bin`.
- `weights_shapes.txt` để biết shape khi đọc từ CUDA.

### Bước 5: Demo webcam

```powershell
.\.venv\Scripts\python.exe step5_demo_video.py --threshold 0.6
```

CNN chạy trên từng khuôn mặt được Haar Cascade phát hiện. Nếu confidence thấp hơn threshold, kết quả hiển thị là `Unknown`.

Demo cung cap them log nhan dien vao `recognition_log.csv`. Moi student duoc ghi toi da mot lan trong moi khoang thoi gian mac dinh 1 giay, gom timestamp, frame, confidence va toa do khuon mat. Co the doi file log va tan suat ghi:

```powershell
.\.venv\Scripts\python.exe step5_demo_video.py --threshold 0.6 --log recognition_log.csv --log-interval 1
```

## 6. Benchmark CPU/GPU

Script benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark_cpu_gpu.py `
  --gray-image data/benchmark/gray/0.jpg `
  --color-image data/benchmark/color/1787208972081_1400025893745852164_g6068070637985194847_887552051926f2595ed7d6380c2d295e.jpg `
  --warmup 20 `
  --iterations 100 `
  --csv benchmark_results_cuda.csv
```

Script thực hiện:

1. Đọc ảnh grayscale hoặc ảnh màu.
2. Với ảnh màu, chuyển BGR sang grayscale.
3. Resize và normalize thành tensor `1x1x64x64`.
4. Chạy warm-up 20 lần.
5. Đo 100 lần inference trên CPU và GPU nếu CUDA khả dụng.
6. Đồng bộ CUDA trước khi ghi thời gian GPU.
7. Tính `median`, `p95` và throughput FPS.
8. Lưu kết quả dạng CSV.

## 7. Kết quả benchmark hiện tại

Kết quả dưới đây đo trên GTX 1650, model nhỏ 2 lớp, batch size 1:

| Input | Device | Median | P95 | Throughput |
|---|---|---:|---:|---:|
| Grayscale | CPU | 1.551 ms | 1.868 ms | 677.391 FPS |
| Grayscale | CUDA | 0.616 ms | 0.685 ms | 1589.830 FPS |
| Color -> grayscale | CPU | 1.676 ms | 2.045 ms | 588.407 FPS |
| Color -> grayscale | CUDA | 0.718 ms | 1.266 ms | 1246.629 FPS |

Speedup theo median:

- Grayscale: `2.52x` trên GPU so với CPU.
- Color: `2.33x` trên GPU so với CPU.

Kết quả gốc nằm trong `benchmark_results_cuda.csv`.

Đây là benchmark phần CNN inference, chưa phải tốc độ toàn bộ webcam. Vì batch hiện tại là 1 và model tương đối nhỏ, overhead GPU vẫn có ảnh hưởng đáng kể.

## 8. CUDA hiện tại

`inference_template.cu` hiện đã có bản triển khai inference một ảnh:

- Đọc toàn bộ trọng số từ thư mục `output/`.
- Kernel Conv2D padding 1 kèm ReLU.
- Ba tầng convolution nối tiếp.
- MaxPool 2x2 sau mỗi tầng convolution.
- FC1, FC2 và argmax.
- In logits và class index của ảnh đầu tiên trong `val_images.bin`.

Để biên dịch và chạy cần cài CUDA Toolkit để có `nvcc`. Máy hiện đã biên dịch và chạy được bằng CUDA 13.3 trên RTX 3060.

Đã đối chiếu ảnh validation đầu tiên giữa CUDA tự viết và PyTorch:

- CUDA logits: `4.33733654, -3.35250473`
- PyTorch logits: `4.33733416, -3.35250378`
- Sai số tuyệt đối lớn nhất: `2.38e-06`
- Dự đoán cả hai: class `0` (`student_01`)
- Kết quả: `PASS` với ngưỡng sai số `1e-3`

## 9. Hướng phát triển tiếp theo

### 9.1 Benchmark theo batch

Mở rộng benchmark với batch size `1, 4, 8, 16, 32, 64, 128` để so sánh rõ hơn giữa i5-14400F, GTX 1650 và RTX 3060. GPU thường có lợi thế lớn hơn khi batch hoặc số khuôn mặt trong frame tăng.

### 9.2 Benchmark nhiều mức pipeline

- `L0`: kernel-only.
- `L1`: model và input đã ở GPU.
- `L2`: bao gồm H2D và D2H.
- `L3`: webcam + Haar Cascade + preprocess + CNN + display.

Cần báo cáo median, p95, throughput, speedup, memory và độ chính xác.

### 9.3 Cải thiện độ tin cậy dữ liệu

- Chia train/validation trước khi augmentation để tránh data leakage.
- Thu thập số lượng ảnh cân bằng giữa các học sinh.
- Tách ảnh theo phiên thu thập, không chỉ chia ngẫu nhiên các biến thể gần giống nhau.
- Hiệu chuẩn threshold `Unknown` bằng tập kiểm tra riêng.
- Bổ sung đủ dữ liệu cho mục tiêu 45 học sinh.

### 9.4 Tối ưu pipeline video

- Gom nhiều khuôn mặt trong frame thành một batch.
- Tách capture, Haar detection, preprocessing và inference thành các stage.
- Đo riêng phần CPU và GPU để phân tích Amdahl's law.
- Chỉ dùng GPU cho phần có đủ độ song song; Haar Cascade và điều khiển webcam có thể vẫn chạy trên CPU.

## 10. Lưu ý khi diễn giải kết quả

- Ảnh màu hiện chỉ được dùng để đo thêm đường chuyển BGR sang grayscale; CNN vẫn là model một kênh.
- Không thể kết luận model nhận diện tốt cho 45 học sinh từ dữ liệu hiện tại chỉ có 2 lớp.
- Validation accuracy `1.0000` không đồng nghĩa khả năng tổng quát ngoài dữ liệu thu thập.
- Benchmark phải giữ nguyên model, input, số lần warm-up và số lần đo khi so sánh hai thiết bị.
- GPU nhanh ở CNN không đảm bảo toàn bộ video nhanh tương ứng vì webcam, Haar Cascade, chuyển dữ liệu và hiển thị có thể chi phối thời gian tổng.

## 11. Phân biệt thí nghiệm ảnh grayscale và ảnh màu

Model hiện tại nhận một kênh `1x64x64`, vì vậy benchmark đang chuyển ảnh màu BGR thành grayscale trước khi đưa vào CNN. Thí nghiệm này phù hợp để đo thêm chi phí preprocessing, nhưng không phải so sánh hai CNN có số kênh khác nhau.

Nếu muốn đo CNN màu thực sự, cần tạo model riêng với lớp đầu tiên là `Conv2d(3, 32, ...)`, preprocess thành `3x64x64` và train lại. Khi đó khối lượng tính toán của convolution đầu tiên tăng, nên phải ghi rõ đây là so sánh giữa hai cấu hình model khác nhau.

## 12. Tiêu chí kết luận

Một kết luận tốt cần trả lời các câu hỏi sau:

1. GPU có nhanh hơn CPU ở batch size nào?
2. RTX 3060 cải thiện bao nhiêu so với GTX 1650?
3. CPU đa luồng mở rộng hiệu quả đến bao nhiêu threads?
4. Chi phí H2D/D2H ảnh hưởng speedup như thế nào?
5. CNN nhanh hơn có làm toàn bộ video nhanh hơn không?
6. Kết quả CPU, PyTorch GPU và CUDA tự viết có cùng nhãn không?

Không được đánh giá speedup nếu logits hoặc nhãn dự đoán giữa các phiên bản không khớp trong sai số cho phép.

## 13. Cap nhat tren may i5-14400F va RTX 3060

Da sua preprocessing de chia anh goc truoc khi augmentation, tranh de cac bien the cua cung anh roi vao ca train va validation. Sau khi tao lai split:

- Anh goc: 101
- Train sau augmentation: 480
- Validation doc lap, khong augmentation: 21
- So lop: 2
- Validation accuracy: 1.0000

Accuracy nay van can dien giai than trong vi validation chi co 21 anh va du lieu moi co 2 hoc sinh.

Benchmark model moi tren RTX 3060, 20 warm-up va 100 iterations:

| Input | Device | Median | P95 | Throughput |
|---|---|---:|---:|---:|
| Grayscale | CPU i5-14400F | 0.740 ms | 0.971 ms | 1296.143 FPS |
| Grayscale | RTX 3060 | 0.371 ms | 0.791 ms | 2220.964 FPS |
| Color -> grayscale | CPU i5-14400F | 0.876 ms | 1.304 ms | 1089.841 FPS |
| Color -> grayscale | RTX 3060 | 0.439 ms | 0.672 ms | 2152.908 FPS |

Benchmark theo batch size `1, 4, 8, 16, 32, 64, 128` duoc luu trong `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv`.

CUDA tu viet da co ban inference mot anh va da duoc doi chieu logits thanh cong voi PyTorch.

### 13.1 Benchmark cuoi voi CUDA 13.3

Sau khi cap nhat CUDA Toolkit 13.3 va NVIDIA driver 610.88, benchmark lai voi 20 warm-up va 100 iterations:

| Input | Device | Median | P95 | Throughput |
|---|---|---:|---:|---:|
| Grayscale | CPU i5-14400F | 0.850 ms | 1.084 ms | 1156.037 FPS |
| Grayscale | RTX 3060 | 0.365 ms | 0.738 ms | 2340.397 FPS |
| Color -> grayscale | CPU i5-14400F | 0.928 ms | 1.315 ms | 1058.537 FPS |
| Color -> grayscale | RTX 3060 | 0.791 ms | 0.992 ms | 1400.811 FPS |

Ket qua luu trong `benchmark_results_cuda133_driver610_final.csv`. CUDA inference va PyTorch cung cho ket qua class `0` voi sai so logits toi da `2.38e-06`.

### 13.2 Benchmark CUDA tu viet theo L0, L1, L2

Da chay 20 warm-up va 100 iterations tren RTX 3060. Ket qua luu trong `benchmark_cuda_levels.csv`:

| Muc do | Pham vi do | Median | P95 | Throughput |
|---|---|---:|---:|---:|
| L0 | Kernel-only, do tung kernel | 1.3461 ms | 2.1807 ms | 742.889 FPS |
| L1 | Toan bo inference khi input/weights da o GPU | 1.2411 ms | 2.0931 ms | 805.745 FPS |
| L2 | H2D + inference + D2H | 1.2878 ms | 2.3067 ms | 776.533 FPS |

L0 co them overhead event va synchronize giua tung kernel de tach rieng thoi gian cac kernel; L1 phu hop hon de dai dien thoi gian inference tren GPU. L2 cho thay chi phi truyen input va logits trong bai do nay.

### 13.3 Benchmark toan bo pipeline webcam L3

Script `benchmark_pipeline.py` do capture, Haar Cascade, crop/preprocess, inference va tong thoi gian moi frame. Chay tren GPU:

```powershell
.\.venv\Scripts\python.exe benchmark_pipeline.py --device cuda --seconds 10
```

Chay tren CPU:

```powershell
.\.venv\Scripts\python.exe benchmark_pipeline.py --device cpu --seconds 10
```

Ket qua duoc luu vao `benchmark_pipeline_results.csv` hoac ten file truyen qua `--csv`. Lan chay trong moi truong phat trien khong co webcam index 0 nen chua thu duoc so do L3 thuc te.
