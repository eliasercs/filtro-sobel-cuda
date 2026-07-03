#include <iostream>
#include <filesystem>
#include <string>

#include "image.hpp"
#include "tile_kernels.hpp"

namespace fs = std::filesystem;

int processImageTile(
    const char* input,
    const char* output,
    int kernelSize,
    float scale
) {
    int width;
    int height;
    int channels;

    unsigned char* img = loadImage(input, &width, &height, &channels);
    if (img == nullptr) return 1;

    fs::path route(output);

    unsigned char* gray = tileRgbToGray(img, width, height, channels);

    std::string newName = (
        route.parent_path() /
        (route.stem().string() + "_gray" + route.extension().string())
    ).string();

    bool save = saveImage(newName.c_str(), gray, width, height);
    if (save) {
        std::cout << "Imagen en escala de grises guardada correctamente." << std::endl;
    } else {
        std::cout << "Error al guardar la imagen en escala de grises." << std::endl;
    }

    float sigma = (kernelSize - 1) / 6.0f;

    unsigned char* blur = tileGaussianBlur(
        gray, width, height, kernelSize, sigma
    );

    freeImage(img);
    delete[] gray;

    newName = (
        route.parent_path() /
        (route.stem().string() + "_blur_kernel_" + std::to_string(kernelSize) +
         "x" + std::to_string(kernelSize) + "_sigma_" + std::to_string(sigma) +
         route.extension().string())
    ).string();

    save = saveImage(newName.c_str(), blur, width, height);
    if (save) {
        std::cout << "Imagen con blur guardada correctamente." << std::endl;
    } else {
        std::cout << "Error al guardar la imagen con blur." << std::endl;
    }

    unsigned char* sobel = tileSobel(blur, width, height);

    delete[] blur;

    newName = (
        route.parent_path() /
        (route.stem().string() + "_sobel_kernel_" + std::to_string(kernelSize) +
         "x" + std::to_string(kernelSize) + "_sigma_" + std::to_string(sigma) +
         route.extension().string())
    ).string();

    save = saveImage(newName.c_str(), sobel, width, height);
    if (save) {
        std::cout << "Imagen con filtro de Sobel guardada correctamente." << std::endl;
    } else {
        std::cout << "Error al guardar la imagen con filtro de Sobel." << std::endl;
    }

    int outWidth = width;
    int outHeight = height;
    unsigned char* resized = tileBilinearResize(
        sobel, width, height, scale, &outWidth, &outHeight
    );

    delete[] sobel;

    if (resized != nullptr) {
        newName = (
            route.parent_path() /
            (route.stem().string() + "_resize_s" + std::to_string(scale) +
             "_to_" + std::to_string(outWidth) + "x" + std::to_string(outHeight) +
             route.extension().string())
        ).string();

        save = saveImage(newName.c_str(), resized, outWidth, outHeight);
        if (save) {
            std::cout << "Imagen redimensionada guardada correctamente "
                      << "(" << outWidth << "x" << outHeight << ")." << std::endl;
        } else {
            std::cout << "Error al guardar la imagen redimensionada." << std::endl;
        }
        delete[] resized;
    } else {
        std::cout << "Resize omitido (factor invalido)." << std::endl;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Uso: tile --instance=<small|medium|large|no-divisible> "
                     "--kernel-size=<kernel_size> [--scale=<factor>]\n";
        return 1;
    }

    std::string instance;
    int kernelSize = 0;
    float scale = 1.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--instance=", 0) == 0) {
            instance = arg.substr(11);
        } else if (arg.rfind("--kernel-size=", 0) == 0) {
            kernelSize = std::stoi(arg.substr(14));
        } else if (arg.rfind("--scale=", 0) == 0) {
            scale = std::stof(arg.substr(8));
        } else {
            std::cout << "Argumento desconocido: " << arg << std::endl;
            return 1;
        }
    }

    if (instance != "small" && instance != "medium" &&
        instance != "large" && instance != "no-divisible") {
        std::cout << "Instancia inválida." << std::endl;
        return 1;
    }
    if (kernelSize < 3 || kernelSize % 2 == 0) {
        std::cout << "El tamaño del kernel debe ser impar y mayor o igual a 3."
                  << std::endl;
        return 1;
    }
    if (scale <= 0.0f) {
        std::cout << "El factor de escala debe ser mayor a 0." << std::endl;
        return 1;
    }

    std::string directory = "../data/" + instance;
    std::string dirOutput = "../results/tile/" + instance;

    if (!fs::exists(directory)) {
        std::cout << "El directorio no existe.\n";
        return 1;
    }
    if (!fs::exists(dirOutput)) {
        fs::create_directories(dirOutput);
    }

    for (const auto& file : fs::directory_iterator(directory)) {
        if (!file.is_regular_file()) continue;

        std::string ext = file.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        std::cout << file.path() << std::endl;

        std::string fileInput = file.path().string();
        std::string fileOutput = (
            fs::path(dirOutput) / file.path().filename()
        ).string();

        processImageTile(
            fileInput.c_str(), fileOutput.c_str(), kernelSize, scale
        );
    }

    return 0;
}
