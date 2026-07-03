#ifndef TILE_KERNELS_HPP
#define TILE_KERNELS_HPP

unsigned char* tileRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels
);

unsigned char* tileGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma
);

unsigned char* tileSobel(
    const unsigned char* h_in,
    int width,
    int height
);

unsigned char* tileBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight
);

#endif
