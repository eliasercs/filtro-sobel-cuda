#ifndef IMAGE_HPP
#define IMAGE_HPP

unsigned char* loadImage(
    const char* path,
    int* width,
    int* height,
    int* channels
);

unsigned char* convertToGrayScale(
    unsigned char* img,
    int width,
    int height,
    int channels
);

bool saveImage(
    const char* path,
    unsigned char* img,
    int width,
    int height
);

void freeImage(unsigned char* img);

float** createKernelGaussiano(
    int size,
    float sigma
);

void freeKernelGaussiano(
    float** kernel,
    int size
);

unsigned char* applyGaussianBlur(
    unsigned char* img,
    int width,
    int height,
    int kernelSize,
    float sigma
);

unsigned char* applySobelFilter(
    unsigned char* img,
    int width,
    int height
);

unsigned char* applyBilinearResize(
    unsigned char* img,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight
);

#endif