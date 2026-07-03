#include "cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- FUNCIÓN DE OPTIMIZACIÓN DINÁMICA (API DE OCCUPANCY) ---
template<class T>
void getOptimalBlockGrid2D(T kernel, int width, int height, dim3& grid, dim3& block) {
    int minGridSize;
    int blockSize; // Cantidad total de hilos recomendada por bloque
    
    // La API de CUDA analiza el kernel en tiempo de ejecución y devuelve el blockSize óptimo
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    
    // Como procesamos imágenes 2D, calculamos un bloque cuadrado (ej. si recomienda 1024, saca 32x32)
    int dim = static_cast<int>(std::sqrt(static_cast<float>(blockSize)));
    block = dim3(dim, dim);
    grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}
// -----------------------------------------------------------

__global__ void rgbToGrayKernel(
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

__global__ void gaussianBlurKernel(
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

            float w = kernel[(ky + radius) * kernelSize + (kx + radius)];
            sum += static_cast<float>(in[yy * width + xx]) * w;
        }
    }

    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    out[y * width + x] = static_cast<unsigned char>(sum);
}

__global__ void sobelKernel(
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

__global__ void bilinearResizeKernel(
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

static float* createFlatGaussKernel(int size, float sigma) {
    int radius = size / 2;
    float* kernel = new float[size * size];
    const float PI = 3.14159265358979323846f;

    float sum = 0.0f;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = static_cast<float>(x) - static_cast<float>(radius);
            float dy = static_cast<float>(y) - static_cast<float>(radius);
            kernel[y * size + x] = expf(
                -(dx * dx + dy * dy) / (2.0f * sigma * sigma)
            ) / (2.0f * PI * sigma * sigma);
            sum += kernel[y * size + x];
        }
    }
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

unsigned char* cudaRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels
) {
    size_t rgbBytes = static_cast<size_t>(width) * height * channels;
    size_t grayBytes = static_cast<size_t>(width) * height;

    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));

    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgbBytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    getOptimalBlockGrid2D(rgbToGrayKernel, width, height, grid, block);

    rgbToGrayKernel<<<grid, block>>>(d_rgb, d_gray, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_gray = new unsigned char[grayBytes];
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, grayBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_rgb);
    cudaFree(d_gray);
    return h_gray;
}

unsigned char* cudaGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma
) {
    float* h_kernel = createFlatGaussKernel(kernelSize, sigma);

    size_t bytes = static_cast<size_t>(width) * height;
    size_t kbytes = static_cast<size_t>(kernelSize) * kernelSize * sizeof(float);

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kbytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kbytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    getOptimalBlockGrid2D(gaussianBlurKernel, width, height, grid, block);

    gaussianBlurKernel<<<grid, block>>>(
        d_in, d_out, width, height, d_kernel, kernelSize
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[bytes];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    delete[] h_kernel;
    return h_out;
}

unsigned char* cudaSobel(
    const unsigned char* h_in,
    int width,
    int height
) {
    size_t bytes = static_cast<size_t>(width) * height;

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    getOptimalBlockGrid2D(sobelKernel, width, height, grid, block);

    sobelKernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[bytes];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    return h_out;
}

unsigned char* cudaBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight
) {
    if (scale <= 0.0f) {
        *outWidth = 0;
        *outHeight = 0;
        return nullptr;
    }

    int newWidth = static_cast<int>(roundf(static_cast<float>(width) * scale));
    int newHeight = static_cast<int>(roundf(static_cast<float>(height) * scale));
    if (newWidth < 1) newWidth = 1;
    if (newHeight < 1) newHeight = 1;

    *outWidth = newWidth;
    *outHeight = newHeight;

    size_t inBytes = static_cast<size_t>(width) * height;
    size_t outBytes = static_cast<size_t>(newWidth) * newHeight;

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));

    dim3 grid, block;
    // Se calcula la ocupación en base al nuevo tamaño de salida
    getOptimalBlockGrid2D(bilinearResizeKernel, newWidth, newHeight, grid, block);

    bilinearResizeKernel<<<grid, block>>>(
        d_in, d_out, width, height, newWidth, newHeight
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* h_out = new unsigned char[outBytes];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    return h_out;
}