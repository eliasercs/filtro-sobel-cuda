#include "tile_kernels.hpp"
#include "image.hpp"

#include "cuda_tile.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define BLOCK_SIZE 16

namespace ct = cuda::tiles;
using namespace ct::literals;

__global__ void simtRgbToGrayKernel(
    const unsigned char* rgb,
    unsigned char* gray,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i = (y * width + x) * channels;
    float r = static_cast<float>(rgb[i]);
    float g = static_cast<float>(rgb[i + 1]);
    float b = static_cast<float>(rgb[i + 2]);

    float v = 0.299f * r + 0.587f * g + 0.114f * b;
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;

    gray[y * width + x] = static_cast<unsigned char>(v);
}

__global__ void simtGaussianBlurKernel(
    const unsigned char* in,
    unsigned char* out,
    int width,
    int height,
    const float* kernel,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int xx = x + kx;
            int yy = y + ky;
            if (xx < 0) xx = 0;
            if (xx >= width) xx = width - 1;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;
            sum += static_cast<float>(in[yy * width + xx]) *
                   kernel[(ky + radius) * kernelSize + (kx + radius)];
        }
    }

    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    out[y * width + x] = static_cast<unsigned char>(sum);
}

__global__ void simtSobelKernel(
    const unsigned char* in,
    unsigned char* out,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int sumX = 0;
    int sumY = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int xx = x + kx;
            int yy = y + ky;
            if (xx < 0) xx = 0;
            if (xx >= width) xx = width - 1;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;

            int v = static_cast<int>(in[yy * width + xx]);
            sumX += v * gx[ky + 1][kx + 1];
            sumY += v * gy[ky + 1][kx + 1];
        }
    }

    int mag = static_cast<int>(sqrtf(
        static_cast<float>(sumX * sumX + sumY * sumY)
    ));
    if (mag > 255) mag = 255;

    out[y * width + x] = static_cast<unsigned char>(mag);
}

__global__ void simtBilinearResizeKernel(
    const unsigned char* in,
    unsigned char* out,
    int inWidth,
    int inHeight,
    int outWidth,
    int outHeight
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outWidth || oy >= outHeight) return;

    float srcX = static_cast<float>(ox) *
        static_cast<float>(inWidth) / static_cast<float>(outWidth);
    float srcY = static_cast<float>(oy) *
        static_cast<float>(inHeight) / static_cast<float>(outHeight);

    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));

    if (x0 < 0) x0 = 0;
    if (x0 > inWidth - 1) x0 = inWidth - 1;
    if (x0 > inWidth - 2) x0 = inWidth - 2;
    if (y0 < 0) y0 = 0;
    if (y0 > inHeight - 1) y0 = inHeight - 1;
    if (y0 > inHeight - 2) y0 = inHeight - 2;

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 > inWidth - 1) x1 = inWidth - 1;
    if (y1 > inHeight - 1) y1 = inHeight - 1;

    float dx = srcX - static_cast<float>(x0);
    float dy = srcY - static_cast<float>(y0);

    float a = static_cast<float>(in[y0 * inWidth + x0]);
    float b = static_cast<float>(in[y0 * inWidth + x1]);
    float c = static_cast<float>(in[y1 * inWidth + x0]);
    float d = static_cast<float>(in[y1 * inWidth + x1]);

    float value = a * (1.0f - dx) * (1.0f - dy) +
                  b * dx * (1.0f - dy) +
                  c * (1.0f - dx) * dy +
                  d * dx * dy;

    if (value < 0.0f) value = 0.0f;
    if (value > 255.0f) value = 255.0f;

    out[oy * outWidth + ox] = static_cast<unsigned char>(value);
}

__tile_global__ void tileIdentityKernel(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    int height,
    int width
) {
    in  = ct::assume_aligned(in,  16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inSpan  = ct::tensor_span{in,  ct::extents{height, width}};
    auto inView  = ct::partition_view{inSpan,  ct::shape{16_ic, 16_ic}};

    auto outSpan = ct::tensor_span{out, ct::extents{height, width}};
    auto outView = ct::partition_view{outSpan, ct::shape{16_ic, 16_ic}};

    int bx = ct::bid().x;
    int by = ct::bid().y;
    auto t = inView.load_masked(bx, by);
    outView.store(t, bx, by);
}

unsigned char* tileRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels,
    TileStageTimings* timings
) {
    if (timings) {
        timings->hToD_ms = 0.0f;
        timings->kernel_ms = 0.0f;
        timings->dToH_ms = 0.0f;
        timings->total_ms = 0.0f;
    }

    size_t rgbBytes  = static_cast<size_t>(width) * height * channels;
    size_t grayBytes = static_cast<size_t>(width) * height;

    cudaEvent_t startTotal = nullptr, stopTotal = nullptr;
    cudaEvent_t startHtoD = nullptr, stopHtoD = nullptr;
    cudaEvent_t startKernel = nullptr, stopKernel = nullptr;
    cudaEvent_t startDtoH = nullptr, stopDtoH = nullptr;

    if (timings) {
        cudaEventCreate(&startTotal);  cudaEventCreate(&stopTotal);
        cudaEventCreate(&startHtoD);  cudaEventCreate(&stopHtoD);
        cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
        cudaEventCreate(&startDtoH);  cudaEventCreate(&stopDtoH);
        cudaEventRecord(startTotal);
    }

    unsigned char* d_rgb  = nullptr;
    unsigned char* d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb,  rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));

    if (timings) cudaEventRecord(startHtoD);
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgbBytes, cudaMemcpyHostToDevice));
    if (timings) cudaEventRecord(stopHtoD);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    if (timings) cudaEventRecord(startKernel);
    simtRgbToGrayKernel<<<grid, block>>>(d_rgb, d_gray, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    if (timings) cudaEventRecord(stopKernel);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_gray = new unsigned char[grayBytes];
    if (timings) cudaEventRecord(startDtoH);
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, grayBytes, cudaMemcpyDeviceToHost));
    if (timings) cudaEventRecord(stopDtoH);

    cudaFree(d_rgb);
    cudaFree(d_gray);

    if (timings) {
        cudaEventRecord(stopTotal);
        cudaEventSynchronize(stopTotal);
        cudaEventElapsedTime(&timings->hToD_ms, startHtoD, stopHtoD);
        cudaEventElapsedTime(&timings->kernel_ms, startKernel, stopKernel);
        cudaEventElapsedTime(&timings->dToH_ms, startDtoH, stopDtoH);
        cudaEventElapsedTime(&timings->total_ms, startTotal, stopTotal);
        cudaEventDestroy(startTotal);  cudaEventDestroy(stopTotal);
        cudaEventDestroy(startHtoD);  cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);  cudaEventDestroy(stopDtoH);
    }

    return h_gray;
}

unsigned char* tileGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma,
    TileStageTimings* timings
) {
    if (timings) {
        timings->hToD_ms = 0.0f;
        timings->kernel_ms = 0.0f;
        timings->dToH_ms = 0.0f;
        timings->total_ms = 0.0f;
    }

    float* h_kernel = createFlatGaussKernel(kernelSize, sigma);

    size_t bytes  = static_cast<size_t>(width) * height;
    size_t kbytes = static_cast<size_t>(kernelSize) * kernelSize * sizeof(float);

    cudaEvent_t startTotal = nullptr, stopTotal = nullptr;
    cudaEvent_t startHtoD = nullptr, stopHtoD = nullptr;
    cudaEvent_t startKernel = nullptr, stopKernel = nullptr;
    cudaEvent_t startDtoH = nullptr, stopDtoH = nullptr;

    if (timings) {
        cudaEventCreate(&startTotal);  cudaEventCreate(&stopTotal);
        cudaEventCreate(&startHtoD);  cudaEventCreate(&stopHtoD);
        cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
        cudaEventCreate(&startDtoH);  cudaEventCreate(&stopDtoH);
        cudaEventRecord(startTotal);
    }

    unsigned char* d_in     = nullptr;
    unsigned char* d_out    = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kbytes));

    if (timings) cudaEventRecord(startHtoD);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kbytes, cudaMemcpyHostToDevice));
    if (timings) cudaEventRecord(stopHtoD);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    if (timings) cudaEventRecord(startKernel);
    simtGaussianBlurKernel<<<grid, block>>>(
        d_in, d_out, width, height, d_kernel, kernelSize
    );
    CUDA_CHECK(cudaGetLastError());
    if (timings) cudaEventRecord(stopKernel);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[bytes];
    if (timings) cudaEventRecord(startDtoH);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (timings) cudaEventRecord(stopDtoH);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    delete[] h_kernel;

    if (timings) {
        cudaEventRecord(stopTotal);
        cudaEventSynchronize(stopTotal);
        cudaEventElapsedTime(&timings->hToD_ms, startHtoD, stopHtoD);
        cudaEventElapsedTime(&timings->kernel_ms, startKernel, stopKernel);
        cudaEventElapsedTime(&timings->dToH_ms, startDtoH, stopDtoH);
        cudaEventElapsedTime(&timings->total_ms, startTotal, stopTotal);
        cudaEventDestroy(startTotal);  cudaEventDestroy(stopTotal);
        cudaEventDestroy(startHtoD);  cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);  cudaEventDestroy(stopDtoH);
    }

    return h_out;
}

unsigned char* tileSobel(
    const unsigned char* h_in,
    int width,
    int height,
    TileStageTimings* timings
) {
    if (timings) {
        timings->hToD_ms = 0.0f;
        timings->kernel_ms = 0.0f;
        timings->dToH_ms = 0.0f;
        timings->total_ms = 0.0f;
    }

    size_t bytes = static_cast<size_t>(width) * height;

    cudaEvent_t startTotal = nullptr, stopTotal = nullptr;
    cudaEvent_t startHtoD = nullptr, stopHtoD = nullptr;
    cudaEvent_t startKernel = nullptr, stopKernel = nullptr;
    cudaEvent_t startDtoH = nullptr, stopDtoH = nullptr;

    if (timings) {
        cudaEventCreate(&startTotal);  cudaEventCreate(&stopTotal);
        cudaEventCreate(&startHtoD);  cudaEventCreate(&stopHtoD);
        cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
        cudaEventCreate(&startDtoH);  cudaEventCreate(&stopDtoH);
        cudaEventRecord(startTotal);
    }

    unsigned char* d_in  = nullptr;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    if (timings) cudaEventRecord(startHtoD);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    if (timings) cudaEventRecord(stopHtoD);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    if (timings) cudaEventRecord(startKernel);
    simtSobelKernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    if (timings) cudaEventRecord(stopKernel);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[bytes];
    if (timings) cudaEventRecord(startDtoH);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (timings) cudaEventRecord(stopDtoH);

    cudaFree(d_in);
    cudaFree(d_out);

    if (timings) {
        cudaEventRecord(stopTotal);
        cudaEventSynchronize(stopTotal);
        cudaEventElapsedTime(&timings->hToD_ms, startHtoD, stopHtoD);
        cudaEventElapsedTime(&timings->kernel_ms, startKernel, stopKernel);
        cudaEventElapsedTime(&timings->dToH_ms, startDtoH, stopDtoH);
        cudaEventElapsedTime(&timings->total_ms, startTotal, stopTotal);
        cudaEventDestroy(startTotal);  cudaEventDestroy(stopTotal);
        cudaEventDestroy(startHtoD);  cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);  cudaEventDestroy(stopDtoH);
    }

    return h_out;
}

unsigned char* tileBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight,
    TileStageTimings* timings
) {
    if (scale <= 0.0f) {
        *outWidth = 0;
        *outHeight = 0;
        if (timings) {
            timings->hToD_ms = 0.0f;
            timings->kernel_ms = 0.0f;
            timings->dToH_ms = 0.0f;
            timings->total_ms = 0.0f;
        }
        return nullptr;
    }

    int newWidth = static_cast<int>(roundf(static_cast<float>(width) * scale));
    int newHeight = static_cast<int>(roundf(static_cast<float>(height) * scale));
    if (newWidth < 1) newWidth = 1;
    if (newHeight < 1) newHeight = 1;

    *outWidth = newWidth;
    *outHeight = newHeight;

    if (timings) {
        timings->hToD_ms = 0.0f;
        timings->kernel_ms = 0.0f;
        timings->dToH_ms = 0.0f;
        timings->total_ms = 0.0f;
    }

    size_t inBytes = static_cast<size_t>(width) * height;
    size_t outBytes = static_cast<size_t>(newWidth) * newHeight;

    cudaEvent_t startTotal = nullptr, stopTotal = nullptr;
    cudaEvent_t startHtoD = nullptr, stopHtoD = nullptr;
    cudaEvent_t startKernel = nullptr, stopKernel = nullptr;
    cudaEvent_t startDtoH = nullptr, stopDtoH = nullptr;

    if (timings) {
        cudaEventCreate(&startTotal);  cudaEventCreate(&stopTotal);
        cudaEventCreate(&startHtoD);  cudaEventCreate(&stopHtoD);
        cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
        cudaEventCreate(&startDtoH);  cudaEventCreate(&stopDtoH);
        cudaEventRecord(startTotal);
    }

    unsigned char* d_in  = nullptr;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    if (timings) cudaEventRecord(startHtoD);
    CUDA_CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));
    if (timings) cudaEventRecord(stopHtoD);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (newWidth  + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (newHeight + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    if (timings) cudaEventRecord(startKernel);
    simtBilinearResizeKernel<<<grid, block>>>(
        d_in, d_out, width, height, newWidth, newHeight
    );
    CUDA_CHECK(cudaGetLastError());
    if (timings) cudaEventRecord(stopKernel);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[outBytes];
    if (timings) cudaEventRecord(startDtoH);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));
    if (timings) cudaEventRecord(stopDtoH);

    cudaFree(d_in);
    cudaFree(d_out);

    if (timings) {
        cudaEventRecord(stopTotal);
        cudaEventSynchronize(stopTotal);
        cudaEventElapsedTime(&timings->hToD_ms, startHtoD, stopHtoD);
        cudaEventElapsedTime(&timings->kernel_ms, startKernel, stopKernel);
        cudaEventElapsedTime(&timings->dToH_ms, startDtoH, stopDtoH);
        cudaEventElapsedTime(&timings->total_ms, startTotal, stopTotal);
        cudaEventDestroy(startTotal);  cudaEventDestroy(stopTotal);
        cudaEventDestroy(startHtoD);  cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);  cudaEventDestroy(stopDtoH);
    }

    return h_out;
}
