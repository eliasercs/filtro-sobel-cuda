#ifndef CUDA_KERNELS_HPP
#define CUDA_KERNELS_HPP

unsigned char* cudaRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels
);

unsigned char* cudaGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma
);

unsigned char* cudaSobel(
    const unsigned char* h_in,
    int width,
    int height
);

unsigned char* cudaBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight
);

#endif
