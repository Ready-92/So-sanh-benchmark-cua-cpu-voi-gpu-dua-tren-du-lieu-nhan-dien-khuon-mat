#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
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

struct DeviceBuffers {
    float* input;
    float* conv1;
    float* pool1;
    float* conv2;
    float* pool2;
    float* conv3;
    float* pool3;
    float* fc1;
    float* logits;
};

static void run_inference(const DeviceBuffers& buffers, float** weights) {
    conv2d(buffers.input, weights[0], weights[1], buffers.conv1, 1, 32, 64, 64);
    maxpool2x2(buffers.conv1, buffers.pool1, 32, 64, 64);
    conv2d(buffers.pool1, weights[2], weights[3], buffers.conv2, 32, 64, 32, 32);
    maxpool2x2(buffers.conv2, buffers.pool2, 64, 32, 32);
    conv2d(buffers.pool2, weights[4], weights[5], buffers.conv3, 64, 128, 16, 16);
    maxpool2x2(buffers.conv3, buffers.pool3, 128, 16, 16);
    linear(buffers.pool3, weights[6], weights[7], buffers.fc1, 8192, 256, true);
    linear(buffers.fc1, weights[8], weights[9], buffers.logits, 256, 2, false);
}

static float percentile_ms(std::vector<float> values, float percentile) {
    std::sort(values.begin(), values.end());
    float position = (values.size() - 1) * percentile;
    size_t lower = static_cast<size_t>(position);
    size_t upper = std::min(lower + 1, values.size() - 1);
    float fraction = position - lower;
    return values[lower] + fraction * (values[upper] - values[lower]);
}

static float elapsed_event(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    return milliseconds;
}

static float run_inference_kernel_timing(const DeviceBuffers& buffers, float** weights) {
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate kernel start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate kernel stop");
    float total = 0.0f;
    auto measure = [&](auto launch) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord kernel start");
        launch();
        check_cuda(cudaEventRecord(stop), "cudaEventRecord kernel stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize kernel stop");
        total += elapsed_event(start, stop);
    };
    measure([&] { conv2d(buffers.input, weights[0], weights[1], buffers.conv1, 1, 32, 64, 64); });
    measure([&] { maxpool2x2(buffers.conv1, buffers.pool1, 32, 64, 64); });
    measure([&] { conv2d(buffers.pool1, weights[2], weights[3], buffers.conv2, 32, 64, 32, 32); });
    measure([&] { maxpool2x2(buffers.conv2, buffers.pool2, 64, 32, 32); });
    measure([&] { conv2d(buffers.pool2, weights[4], weights[5], buffers.conv3, 64, 128, 16, 16); });
    measure([&] { maxpool2x2(buffers.conv3, buffers.pool3, 128, 16, 16); });
    measure([&] { linear(buffers.pool3, weights[6], weights[7], buffers.fc1, 8192, 256, true); });
    measure([&] { linear(buffers.fc1, weights[8], weights[9], buffers.logits, 256, 2, false); });
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy kernel start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy kernel stop");
    return total;
}

static void benchmark_level(
    const char* level, const DeviceBuffers& buffers, float** weights,
    const float* host_input, float* host_logits, int warmup, int iterations
) {
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    for (int index = 0; index < warmup; ++index) {
        if (level[1] == '2') {
            check_cuda(cudaMemcpy(buffers.input, host_input, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice), "warmup H2D");
        }
        if (level[1] == '0') {
            run_inference_kernel_timing(buffers, weights);
        } else {
            run_inference(buffers, weights);
        }
        if (level[1] == '2') {
            check_cuda(cudaMemcpy(host_logits, buffers.logits, 2 * sizeof(float), cudaMemcpyDeviceToHost), "warmup D2H");
        }
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<float> timings;
    timings.reserve(iterations);
    for (int index = 0; index < iterations; ++index) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");
        if (level[1] == '2') {
            check_cuda(cudaMemcpy(buffers.input, host_input, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        }
        float kernel_time = 0.0f;
        if (level[1] == '0') {
            kernel_time = run_inference_kernel_timing(buffers, weights);
        } else {
            run_inference(buffers, weights);
        }
        if (level[1] == '2') {
            check_cuda(cudaMemcpy(host_logits, buffers.logits, 2 * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        }
        if (level[1] == '0') {
            timings.push_back(kernel_time);
        } else {
            check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
            timings.push_back(elapsed_event(start, stop));
        }
    }

    float median = percentile_ms(timings, 0.50f);
    float p95 = percentile_ms(timings, 0.95f);
    printf("%s | median=%.4f ms | p95=%.4f ms | FPS=%.3f\n", level, median, p95, 1000.0f / median);
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
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

    DeviceBuffers buffers = {
        device_input, conv1, pool1, conv2, pool2, conv3, pool3, fc1, logits
    };
    run_inference(buffers, device_weights);

    float host_logits[classes];
    check_cuda(cudaMemcpy(host_logits, logits, classes * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy logits D2H");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int prediction = host_logits[0] > host_logits[1] ? 0 : 1;
    printf("CUDA inference hoan tat. Logits: %.8f %.8f\n", host_logits[0], host_logits[1]);
    printf("Predicted class index: %d\n", prediction);

    printf("\nBenchmark CUDA (warmup=20, iterations=100):\n");
    benchmark_level("L0 kernel-only", buffers, device_weights, host_input, host_logits, 20, 100);
    benchmark_level("L1 device inference", buffers, device_weights, host_input, host_logits, 20, 100);
    benchmark_level("L2 H2D+inference+D2H", buffers, device_weights, host_input, host_logits, 20, 100);

    free(host_input);
    cudaFree(device_input);
    for (float* weight : device_weights) cudaFree(weight);
    cudaFree(conv1); cudaFree(pool1); cudaFree(conv2); cudaFree(pool2);
    cudaFree(conv3); cudaFree(pool3); cudaFree(fc1); cudaFree(logits);
    return 0;
}
