#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static float* load_binary(const char* path, size_t count) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Khong mo duoc file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    float* data = static_cast<float*>(malloc(count * sizeof(float)));
    if (!data || fread(data, sizeof(float), count, file) != count) {
        fprintf(stderr, "Khong doc du du lieu: %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
    return data;
}

__global__ void conv2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int in_channels, int out_channels, int height, int width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = out_channels * height * width;
    if (index >= output_size) return;

    int x = index % width;
    int y = (index / width) % height;
    int out_channel = index / (height * width);
    float value = bias[out_channel];

    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
            for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                int input_y = y + kernel_y;
                int input_x = x + kernel_x;
                if (input_y < 0 || input_y >= height || input_x < 0 || input_x >= width) {
                    continue;
                }
                int input_index = (in_channel * height + input_y) * width + input_x;
                int weight_index = ((out_channel * in_channels + in_channel) * 3
                                    + kernel_y + 1) * 3 + kernel_x + 1;
                value += input[input_index] * weight[weight_index];
            }
        }
    }
    output[index] = value > 0.0f ? value : 0.0f;
}

__global__ void maxpool2x2_kernel(
    const float* input, float* output, int channels, int input_height, int input_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int output_height = input_height / 2;
    int output_width = input_width / 2;
    int output_size = channels * output_height * output_width;
    if (index >= output_size) return;

    int x = index % output_width;
    int y = (index / output_width) % output_height;
    int channel = index / (output_height * output_width);
    int input_base = (channel * input_height + y * 2) * input_width + x * 2;
    float value = input[input_base];
    value = fmaxf(value, input[input_base + 1]);
    value = fmaxf(value, input[input_base + input_width]);
    value = fmaxf(value, input[input_base + input_width + 1]);
    output[index] = value;
}

__global__ void linear_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int input_size, int output_size, bool apply_relu
) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= output_size) return;

    float value = bias[output_index];
    for (int input_index = 0; input_index < input_size; ++input_index) {
        value += input[input_index] * weight[output_index * input_size + input_index];
    }
    output[output_index] = apply_relu && value < 0.0f ? 0.0f : value;
}

static void launch_check(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
}

static void copy_to_device(float** destination, const float* source, size_t count) {
    check_cuda(cudaMalloc(destination, count * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMemcpy(*destination, source, count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

static void conv2d(
    const float* input, const float* weight, const float* bias, float* output,
    int in_channels, int out_channels, int height, int width
) {
    int count = out_channels * height * width;
    conv2d_kernel<<<(count + 255) / 256, 256>>>(
        input, weight, bias, output, in_channels, out_channels, height, width
    );
    launch_check("conv2d_kernel");
}

static void maxpool2x2(const float* input, float* output, int channels, int height, int width) {
    int count = channels * (height / 2) * (width / 2);
    maxpool2x2_kernel<<<(count + 255) / 256, 256>>>(input, output, channels, height, width);
    launch_check("maxpool2x2_kernel");
}

static void linear(
    const float* input, const float* weight, const float* bias,
    float* output, int input_size, int output_size, bool apply_relu
) {
    linear_kernel<<<(output_size + 255) / 256, 256>>>(
        input, weight, bias, output, input_size, output_size, apply_relu
    );
    launch_check("linear_kernel");
}

int main() {
    const int image_size = 64;
    const int classes = 2;
    const char* prefix = "output/";

    float* host_input = load_binary("output/val_images.bin", image_size * image_size);
    float* host_weights[10];
    const char* weight_files[10] = {
        "output/weights_features.0.weight.bin", "output/weights_features.0.bias.bin",
        "output/weights_features.3.weight.bin", "output/weights_features.3.bias.bin",
        "output/weights_features.6.weight.bin", "output/weights_features.6.bias.bin",
        "output/weights_classifier.1.weight.bin", "output/weights_classifier.1.bias.bin",
        "output/weights_classifier.4.weight.bin", "output/weights_classifier.4.bias.bin"
    };
    const size_t weight_counts[10] = {
        32 * 1 * 3 * 3, 32, 64 * 32 * 3 * 3, 64,
        128 * 64 * 3 * 3, 128, 256 * 8192, 256, classes * 256, classes
    };

    float* device_input = nullptr;
    float* device_weights[10] = {};
    for (int index = 0; index < 10; ++index) {
        host_weights[index] = load_binary(weight_files[index], weight_counts[index]);
        copy_to_device(&device_weights[index], host_weights[index], weight_counts[index]);
        free(host_weights[index]);
    }
    copy_to_device(&device_input, host_input, image_size * image_size);
    free(host_input);

    float *conv1 = nullptr, *pool1 = nullptr, *conv2 = nullptr, *pool2 = nullptr;
    float *conv3 = nullptr, *pool3 = nullptr, *fc1 = nullptr, *logits = nullptr;
    check_cuda(cudaMalloc(&conv1, 32 * 64 * 64 * sizeof(float)), "cudaMalloc conv1");
    check_cuda(cudaMalloc(&pool1, 32 * 32 * 32 * sizeof(float)), "cudaMalloc pool1");
    check_cuda(cudaMalloc(&conv2, 64 * 32 * 32 * sizeof(float)), "cudaMalloc conv2");
    check_cuda(cudaMalloc(&pool2, 64 * 16 * 16 * sizeof(float)), "cudaMalloc pool2");
    check_cuda(cudaMalloc(&conv3, 128 * 16 * 16 * sizeof(float)), "cudaMalloc conv3");
    check_cuda(cudaMalloc(&pool3, 128 * 8 * 8 * sizeof(float)), "cudaMalloc pool3");
    check_cuda(cudaMalloc(&fc1, 256 * sizeof(float)), "cudaMalloc fc1");
    check_cuda(cudaMalloc(&logits, classes * sizeof(float)), "cudaMalloc logits");

    conv2d(device_input, device_weights[0], device_weights[1], conv1, 1, 32, 64, 64);
    maxpool2x2(conv1, pool1, 32, 64, 64);
    conv2d(pool1, device_weights[2], device_weights[3], conv2, 32, 64, 32, 32);
    maxpool2x2(conv2, pool2, 64, 32, 32);
    conv2d(pool2, device_weights[4], device_weights[5], conv3, 64, 128, 16, 16);
    maxpool2x2(conv3, pool3, 128, 16, 16);
    linear(pool3, device_weights[6], device_weights[7], fc1, 8192, 256, true);
    linear(fc1, device_weights[8], device_weights[9], logits, 256, classes, false);

    float host_logits[classes];
    check_cuda(cudaMemcpy(host_logits, logits, classes * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy logits D2H");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int prediction = host_logits[0] > host_logits[1] ? 0 : 1;
    printf("CUDA inference hoan tat. Logits: %.8f %.8f\n", host_logits[0], host_logits[1]);
    printf("Predicted class index: %d\n", prediction);

    cudaFree(device_input);
    for (float* weight : device_weights) cudaFree(weight);
    cudaFree(conv1); cudaFree(pool1); cudaFree(conv2); cudaFree(pool2);
    cudaFree(conv3); cudaFree(pool3); cudaFree(fc1); cudaFree(logits);
