#ifndef CUDA_KERNELS_HPP
#define CUDA_KERNELS_HPP

struct CudaStageTimings {
    float hToD_ms;
    float kernel_ms;
    float dToH_ms;
    float total_ms;
};

unsigned char* cudaRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels,
    CudaStageTimings* timings = nullptr
);

unsigned char* cudaGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma,
    CudaStageTimings* timings = nullptr
);

unsigned char* cudaGaussianBlurSeparable(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma,
    CudaStageTimings* timings = nullptr
);

unsigned char* cudaSobel(
    const unsigned char* h_in,
    int width,
    int height,
    CudaStageTimings* timings = nullptr
);

unsigned char* cudaBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight,
    CudaStageTimings* timings = nullptr
);

#endif
