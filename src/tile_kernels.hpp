#ifndef TILE_KERNELS_HPP
#define TILE_KERNELS_HPP

struct TileStageTimings {
    float hToD_ms;
    float kernel_ms;
    float dToH_ms;
    float total_ms;
};

unsigned char* tileRgbToGray(
    const unsigned char* h_rgb,
    int width,
    int height,
    int channels,
    TileStageTimings* timings = nullptr
);

unsigned char* tileGaussianBlur(
    const unsigned char* h_in,
    int width,
    int height,
    int kernelSize,
    float sigma,
    TileStageTimings* timings = nullptr
);

unsigned char* tileSobel(
    const unsigned char* h_in,
    int width,
    int height,
    TileStageTimings* timings = nullptr
);

unsigned char* tileBilinearResize(
    const unsigned char* h_in,
    int width,
    int height,
    float scale,
    int* outWidth,
    int* outHeight,
    TileStageTimings* timings = nullptr
);

#endif
