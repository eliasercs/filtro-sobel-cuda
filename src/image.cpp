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

    for (int y = 0; y < kernelSize; y++) {
        for (int x = 0; x < kernelSize; x++) {
            std::cout << kernel[y][x] << " ";
        }
        std::cout << std::endl;
    }

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
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
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
