# Bao cao benchmark CPU/GPU

Ngay lap bao cao: 2026-08-24

## 1. Muc tieu

So sanh toc do suy luan CNN nhan dien khuon mat tren:

- CPU cu: Intel Core i5-10300F
- GPU cu: NVIDIA GeForce GTX 1650
- CPU moi: Intel Core i5-14400F
- GPU moi: NVIDIA GeForce RTX 3060 12 GB

## 2. Cau hinh va phuong phap

- Model: `face_cnn.pth`
- Dau vao model: anh grayscale 1 x 64 x 64
- Anh mau duoc chuyen sang grayscale truoc khi dua vao model
- Warm-up: 20 lan
- So lan do: 100 lan
- Chi so: median latency, p95 latency va throughput FPS
- CUDA duoc dong bo truoc khi ghi thoi gian
- Benchmark nay do rieng buoc CNN inference, khong bao gom webcam, Haar Cascade, crop va hien thi
- Anh grayscale: `data/student_01/0.jpg`
- Anh mau: `data/student_2/1787208972081_1400025893745852164_g6068070637985194847_887552051926f2595ed7d6380c2d295e.jpg`

## 3. Xac nhan moi truong moi

- CPU he dieu hanh: Intel Core i5-14400F
- GPU he dieu hanh: NVIDIA GeForce RTX 3060
- NVIDIA driver: 576.80
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- `torch.cuda.is_available()`: `True`
- OpenCV: `4.11.0.86`

## 4. Ket qua moc cu: i5-10300F va GTX 1650

Du lieu goc trong `benchmark_results_cuda.csv`.

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-10300F / CPU | 1.551 | 1.868 | 677.391 |
| Grayscale | GTX 1650 / CUDA | 0.616 | 0.685 | 1589.830 |
| Color | i5-10300F / CPU | 1.676 | 2.045 | 588.407 |
| Color | GTX 1650 / CUDA | 0.718 | 1.266 | 1246.629 |

## 5. Ket qua CPU moi: i5-14400F

Du lieu trong `benchmark_results_i5_14400f_cpu.csv`.

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.669 | 0.745 | 1475.172 |
| Color | i5-14400F / CPU | 0.727 | 0.966 | 1342.273 |

## 6. Ket qua moi: i5-14400F va RTX 3060, luot dau

Du lieu trong `benchmark_results_i5_14400f_rtx3060.csv`.

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.762 | 1.176 | 1198.515 |
| Grayscale | RTX 3060 / CUDA | 0.714 | 0.914 | 1620.039 |
| Color | i5-14400F / CPU | 3.043 | 6.855 | 267.359 |
| Color | RTX 3060 / CUDA | 0.891 | 1.833 | 975.251 |

## 7. Ket qua moi chinh thuc dung de so sanh: luot lap

Luut lap co ket qua on dinh hon va duoc dung lam ket qua dai dien cho RTX 3060.

Du lieu trong `benchmark_results_i5_14400f_rtx3060_repeat.csv`.

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.725 | 0.913 | 1330.725 |
| Grayscale | RTX 3060 / CUDA | 0.363 | 0.483 | 2603.584 |
| Color | i5-14400F / CPU | 0.738 | 0.922 | 1307.149 |
| Color | RTX 3060 / CUDA | 0.422 | 0.583 | 2349.320 |

## 8. So sanh GPU: RTX 3060 voi GTX 1650

Tinh theo median latency va throughput cua luot lap RTX 3060.

| Dau vao | GTX 1650 | RTX 3060 | Giam latency | Tang FPS |
|---|---:|---:|---:|---:|
| Grayscale | 0.616 ms / 1589.830 FPS | 0.363 ms / 2603.584 FPS | 1.70x | 1.64x |
| Color | 0.718 ms / 1246.629 FPS | 0.422 ms / 2349.320 FPS | 1.70x | 1.88x |

## 9. So sanh CPU: i5-14400F voi i5-10300F

Tinh theo median latency va throughput cua luot lap RTX 3060, doi chieu voi moc CPU cu.

| Dau vao | i5-10300F | i5-14400F | Giam latency | Tang FPS |
|---|---:|---:|---:|---:|
| Grayscale | 1.551 ms / 677.391 FPS | 0.725 ms / 1330.725 FPS | 2.14x | 1.96x |
| Color | 1.676 ms / 588.407 FPS | 0.738 ms / 1307.149 FPS | 2.27x | 2.22x |

## 10. Nhan xet

- RTX 3060 nhanh hon GTX 1650 khoang 1.64x den 1.88x theo FPS trong bai do nay.
- i5-14400F nhanh hon i5-10300F khoang 1.96x den 2.22x theo FPS.
- Anh mau khong lam cham CNN dang ke sau khi da chuyen ve grayscale 64 x 64; chenh lech chu yeu den tu do dao dong do he thong va chi phi xu ly truoc do.
- Day la benchmark batch size 1 voi model nho. Overhead goi CUDA co the lam khoang cach giua GPU nho hon so voi cac tac vu batch lon.
- Ket qua khong dai dien cho toan bo pipeline video thuc te, vi Haar Cascade, webcam va hien thi van co the chay tren CPU.

## 11. Kiem tra do tin cay va do lap lai

Da chay them 3 luot benchmark doc lap, moi luot gom 20 lan warm-up va 100 lan do. Cac gia tri duoi day la trung binh cua `median_ms` giua 3 luot.

| Dau vao | Thiet bi | Median trung binh (ms) | Min-Max (ms) | CV |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.742 | 0.740-0.746 | 0.4% |
| Grayscale | RTX 3060 / CUDA | 0.359 | 0.339-0.374 | 4.1% |
| Color | i5-14400F / CPU | 0.746 | 0.724-0.765 | 2.3% |
| Color | RTX 3060 / CUDA | 0.438 | 0.429-0.448 | 1.8% |

Danh sach file kiem tra:

- `benchmark_results_i5_14400f_rtx3060_check1.csv`
- `benchmark_results_i5_14400f_rtx3060_check2.csv`
- `benchmark_results_i5_14400f_rtx3060_check3.csv`

Danh gia:

- Do lap lai cua may moi la tot: CV deu duoi 5%, khong co dau hieu ket qua bi chi phoi boi mot lan chay bat thuong.
- Ket qua RTX 3060 nhanh hon GTX 1650 la dang tin cay trong pham vi bai do batch size 1 va cung cach do.
- So sanh giua hai may van chi nen xem la ket qua thuc nghiem tham khao, vi CSV cu khong luu so thread CPU, muc tai he thong va nhiet do/tan so tai thoi diem do.
- De ket luan chat che hon ve CPU, can chay lai ca i5-10300F va i5-14400F voi cung so thread, cung phien ban PyTorch, cung driver/phuong phap va nhieu luot lap.

## 12. Cac file du lieu

- `benchmark_results.csv`: moc CPU truoc do voi duong dan benchmark cu.
- `benchmark_results_cuda.csv`: moc i5-10300F va GTX 1650.
- `benchmark_results_i5_14400f_cpu.csv`: mot luot do CPU tren i5-14400F.
- `benchmark_results_i5_14400f_rtx3060.csv`: luot do dau tien tren RTX 3060.
- `benchmark_results_i5_14400f_rtx3060_repeat.csv`: luot lap dung lam ket qua chinh.
- `benchmark_results_i5_14400f_rtx3060_check1.csv` den `check3.csv`: ba luot kiem tra do lap lai.

## 13. Cap nhat sau khi sua data leakage

Da sua `step2_preprocess.py` de chia anh goc truoc khi tao augmentation. Ket qua moi:

- 101 anh goc
- 480 anh train sau augmentation
- 21 anh validation doc lap, khong augmentation
- 2 lop
- Validation accuracy: 1.0000

Model moi da duoc huan luyen lai bang `step3_train.py`, export lai bang `step4_export_cuda.py`, sau do benchmark lai.

Benchmark model moi tren i5-14400F va RTX 3060:

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.740 | 0.971 | 1296.143 |
| Grayscale | RTX 3060 / CUDA | 0.371 | 0.791 | 2220.964 |
| Color | i5-14400F / CPU | 0.876 | 1.304 | 1089.841 |
| Color | RTX 3060 / CUDA | 0.439 | 0.672 | 2152.908 |

Benchmark batch voi model moi nam trong `benchmark_results_batch_clean_model_i5_14400f_rtx3060.csv`, gom batch size 1, 4, 8, 16, 32, 64 va 128.

`inference_template.cu` da duoc trien khai Conv2D, ReLU, MaxPool, FC va argmax cho mot anh. CUDA 13.3 da bien dich va chay thanh cong tren RTX 3060.

Da doi chieu tu dong voi PyTorch tren anh validation dau tien:

- CUDA logits: `4.33733654, -3.35250473`
- PyTorch logits: `4.33733416, -3.35250378`
- Sai so tuyet doi lon nhat: `2.38e-06`
- CUDA va PyTorch cung du doan class `0` (`student_01`)
- Ket qua: `PASS` voi nguong sai so `1e-3`

## 14. Benchmark cuoi voi CUDA 13.3 va driver 610.88

Ket qua duoc do lai sau khi cap nhat CUDA Toolkit 13.3 va NVIDIA driver 610.88, voi 20 warm-up va 100 iterations. Day la moc hieu nang moi nhat.

| Dau vao | Thiet bi | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| Grayscale | i5-14400F / CPU | 0.850 | 1.084 | 1156.037 |
| Grayscale | RTX 3060 / CUDA | 0.365 | 0.738 | 2340.397 |
| Color | i5-14400F / CPU | 0.928 | 1.315 | 1058.537 |
| Color | RTX 3060 / CUDA | 0.791 | 0.992 | 1400.811 |

Du lieu day du nam trong `benchmark_results_cuda133_driver610_final.csv`.

## 16. Benchmark toan bo pipeline webcam L3

Da tao `benchmark_pipeline.py` de do cac thanh phan:

- Capture frame tu webcam.
- Haar Cascade face detection.
- Crop va preprocess.
- CNN inference tren CPU hoac CUDA.
- Tong thoi gian moi frame va pipeline FPS.

Lenh chay:

```powershell
.\.venv\Scripts\python.exe benchmark_pipeline.py --device cuda --seconds 10
```

Lan thu nghiem trong moi truong phat trien bao loi `Camera index out of range` voi webcam index 0, nen chua co so do L3. Can chay tren may co webcam hoat dong de ghi ket qua vao `benchmark_pipeline_results.csv`.

## 15. Benchmark CUDA tu viet theo L0, L1, L2

Da chay 20 warm-up va 100 iterations tren RTX 3060 sau khi bien dich bang CUDA 13.3. Du lieu luu trong `benchmark_cuda_levels.csv`.

| Muc do | Pham vi do | Median (ms) | P95 (ms) | FPS |
|---|---|---:|---:|---:|
| L0 | Kernel-only, do tung kernel | 1.3461 | 2.1807 | 742.889 |
| L1 | Device inference end-to-end | 1.2411 | 2.0931 | 805.745 |
| L2 | H2D + inference + D2H | 1.2878 | 2.3067 | 776.533 |

L0 tach thoi gian tung kernel nhung co overhead event va synchronize giua cac kernel. L1 la chi so device inference end-to-end phu hop hon de so sanh pipeline CUDA; L2 bo sung chi phi upload input va download logits.
