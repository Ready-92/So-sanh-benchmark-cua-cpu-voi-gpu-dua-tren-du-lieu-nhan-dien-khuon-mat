// inference_template.cu
// Khung soan san de doc du lieu/trong so tu Python (step4_export_cuda.py)
// Nguoi lam phan CUDA se dien phan convolution/kernel song song vao day.

#include <cstdio>
#include <cstdlib>
#include <vector>

float* load_binary(const char* path, long count) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Khong mo duoc file: %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);
    return data;
}

int main() {
    // Vi du: nap trong so lop conv dau tien (features.0.weight: [32,1,3,3])
    long conv1_w_count = 32 * 1 * 3 * 3;
    float* conv1_weight = load_binary("output/weights_features.0.weight.bin", conv1_w_count);

    long conv1_b_count = 32;
    float* conv1_bias = load_binary("output/weights_features.0.bias.bin", conv1_b_count);

    // Nap 1 anh mau tu val_images.bin de test (1x1x64x64 = 4096 float)
    float* sample_image = load_binary("output/val_images.bin", 64 * 64);

    printf("Da nap xong trong so va anh mau. San sang cho convolution kernel CUDA.\n");

    // TODO (nguoi lam CUDA): viet kernel convolution 2D chay tren GPU tai day
    // __global__ void conv2d_kernel(...) { ... }
    // Goi kernel, so sanh ket qua voi output cua PyTorch de kiem tra dung/sai.

    free(conv1_weight);
    free(conv1_bias);
    free(sample_image);
    return 0;
}