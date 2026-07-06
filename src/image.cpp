#include <iostream>

/* Código secuencial: se utiliza como referencia para las demás variantes de Cuda en C++ */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.hpp"

#include <cmath>

using namespace std;

unsigned char* loadImage(
    const char* path,
    int* width,
    int* height,
    int* channels
) {
    unsigned char* img = stbi_load(
        path, width, height, channels, 0
    );

    if (img == nullptr) {
        cout << "Error al cargar la imagen." << endl;
    }

    return img;
}

unsigned char* convertToGrayScale(
    unsigned char* img,
    int width,
    int height,
    int channels
) {
    unsigned char* gray = new unsigned char[width * height];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * channels;

            unsigned char r = img[i];
            unsigned char g = img[i + 1];
            unsigned char b = img[i + 2];

            gray[y * width + x] = static_cast<unsigned char>(
                0.299 * r + 0.587 * g + 0.114 * b
            );
        }
    }

    return gray;
}

bool saveImage(
    const char* path,
    unsigned char* img,
    int width,
    int height
) {
    return stbi_write_png(
        path, width, height, 1, img, width
    );
}

void freeImage(unsigned char* img) {
    stbi_image_free(img);
}

float** createKernelGaussiano(
    int size,
    float sigma
) {
    int radius = size / 2;

    float** kernel = new float*[size];

    for (int i = 0; i < size; i++) {
        kernel[i] = new float[size];
    }

    const float PI = 3.14159265358979323846f;

    float sum = 0.0f;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - radius;
            float dy = y - radius;

            kernel[y][x] = exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma)) /
                (2.0f * PI * sigma * sigma);

            sum += kernel[y][x];
        }
    }

    // Normalizar el kernel
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

void freeKernelGaussiano(
    float** kernel,
    int size
) {
    for (int i = 0; i < size; i++) {
        delete[] kernel[i];
    }

    delete[] kernel;
}

float* createFlatGaussKernel(
    int size,
    float sigma
) {
    int radius = size / 2;
    float* kernel = new float[size * size];
    const float PI = 3.14159265358979323846f;

    float sum = 0.0f;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = static_cast<float>(x - radius);
            float dy = static_cast<float>(y - radius);
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

float* createSeparableGaussKernel(
    int size,
    float sigma
) {
    int radius = size / 2;
    float* kernel = new float[size];
    const float PI = 3.14159265358979323846f;

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float dx = static_cast<float>(i - radius);
        kernel[i] = expf(-(dx * dx) / (2.0f * sigma * sigma))
                   / (sqrtf(2.0f * PI) * sigma);
        sum += kernel[i];
    }
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

unsigned char* applyGaussianBlur(
    unsigned char* img,
    int width,
    int height,
    int kernelSize,
    float sigma
) {
    int radius = kernelSize / 2;

    float** kernel = createKernelGaussiano(
        kernelSize, sigma
    );

    unsigned char* output = new unsigned char[width * height];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int xx = x + kx;
                    int yy = y + ky;

                    if (xx < 0) {
                        xx = 0;
                    }
                    if (xx >= width) {
                        xx = width - 1;
                    }

                    if (yy < 0) {
                        yy = 0;
                    }
                    if (yy >= height) {
                        yy = height - 1;
                    }

                    sum += img[yy * width + xx] * kernel[ky + radius][kx + radius];
                }
            }

            output[y * width + x] = static_cast<unsigned char>(sum);

        }
    }

    freeKernelGaussiano(kernel, kernelSize);

    return output;
}

unsigned char* applySobelFilter(
    unsigned char* img,
    int width,
    int height
) {
    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    unsigned char* output = new unsigned char[width * height];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = x + kx;
                    int yy = y + ky;

                    if (xx < 0) {
                        xx = 0;
                    }
                    if (xx >= width) {
                        xx = width - 1;
                    }

                    if (yy < 0) {
                        yy = 0;
                    }
                    if (yy >= height) {
                        yy = height - 1;
                    }

                    sumX += img[yy * width + xx] * gx[ky + 1][kx + 1];
                    sumY += img[yy * width + xx] * gy[ky + 1][kx + 1];
                }
            }

            // Saturación o normalización mediante clamping
            int magnitude = static_cast<int>(sqrt(sumX * sumX + sumY * sumY));
            if (magnitude > 255) {
                magnitude = 255;
            }

            output[y * width + x] = static_cast<unsigned char>(magnitude);
        }
    }

    return output;
}

unsigned char* applyBilinearResize(
    unsigned char* img,
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

    int newWidth = static_cast<int>(std::round(static_cast<float>(width) * scale));
    int newHeight = static_cast<int>(std::round(static_cast<float>(height) * scale));

    if (newWidth < 1) newWidth = 1;
    if (newHeight < 1) newHeight = 1;

    *outWidth = newWidth;
    *outHeight = newHeight;

    unsigned char* output = new unsigned char[newWidth * newHeight];

    for (int oy = 0; oy < newHeight; oy++) {
        for (int ox = 0; ox < newWidth; ox++) {
            float srcX = static_cast<float>(ox) *
                static_cast<float>(width) / static_cast<float>(newWidth);
            float srcY = static_cast<float>(oy) *
                static_cast<float>(height) / static_cast<float>(newHeight);

            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));

            if (x0 < 0) x0 = 0;
            if (x0 > width - 1) x0 = width - 1;
            if (x0 > width - 2) x0 = width - 2;

            if (y0 < 0) y0 = 0;
            if (y0 > height - 1) y0 = height - 1;
            if (y0 > height - 2) y0 = height - 2;

            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x1 > width - 1) x1 = width - 1;
            if (y1 > height - 1) y1 = height - 1;

            float dx = srcX - static_cast<float>(x0);
            float dy = srcY - static_cast<float>(y0);

            float a = static_cast<float>(img[y0 * width + x0]);
            float b = static_cast<float>(img[y0 * width + x1]);
            float c = static_cast<float>(img[y1 * width + x0]);
            float d = static_cast<float>(img[y1 * width + x1]);

            float value = a * (1.0f - dx) * (1.0f - dy) +
                          b * dx * (1.0f - dy) +
                          c * (1.0f - dx) * dy +
                          d * dx * dy;

            if (value < 0.0f) value = 0.0f;
            if (value > 255.0f) value = 255.0f;

            output[oy * newWidth + ox] = static_cast<unsigned char>(value);
        }
    }

    return output;
}
