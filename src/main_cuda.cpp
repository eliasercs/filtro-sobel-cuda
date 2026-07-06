#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>

#include "image.hpp"
#include "cuda_kernels.hpp"

namespace fs = std::filesystem;

int processImageCuda(
    const char* input,
    const char* output,
    int kernelSize,
    float scale,
    const std::string& instance,
    bool separable = false
) {
    int width, height, channels;

    unsigned char* img = loadImage(input, &width, &height, &channels);
    if (img == nullptr) return 1;

    fs::path route(output);
    float sigma = (kernelSize - 1) / 6.0f;

    int repeticiones = 10;
    std::vector<CudaStageTimings> t_gray(repeticiones);
    std::vector<CudaStageTimings> t_blur(repeticiones);
    std::vector<CudaStageTimings> t_sobel(repeticiones);
    std::vector<CudaStageTimings> t_resize(repeticiones);

    std::cout << "\nProcesando imagen: " << fs::path(input).filename().string() << std::endl;

    std::cout << "Realizando iteracion de calentamiento de GPU...\n";
    CudaStageTimings t_dummy;
    unsigned char* w_gray = cudaRgbToGray(img, width, height, channels, &t_dummy);
    unsigned char* w_blur = separable
        ? cudaGaussianBlurSeparable(w_gray, width, height, kernelSize, sigma, &t_dummy)
        : cudaGaussianBlur(w_gray, width, height, kernelSize, sigma, &t_dummy);
    unsigned char* w_sobel = cudaSobel(w_blur, width, height, &t_dummy);
    int w_outW, w_outH;
    unsigned char* w_resize = cudaBilinearResize(w_sobel, width, height, scale, &w_outW, &w_outH, &t_dummy);
    delete[] w_gray; delete[] w_blur; delete[] w_sobel;
    if (w_resize != nullptr) delete[] w_resize;

    std::cout << "Iniciando " << repeticiones << " repeticiones de medicion...\n";

    unsigned char* final_gray = nullptr;
    unsigned char* final_blur = nullptr;
    unsigned char* final_sobel = nullptr;
    unsigned char* final_resized = nullptr;
    int outWidth = width;
    int outHeight = height;

    for (int i = 0; i < repeticiones; ++i) {
        CudaStageTimings ts;

        unsigned char* gray = cudaRgbToGray(img, width, height, channels, &t_gray[i]);

        unsigned char* blur = separable
            ? cudaGaussianBlurSeparable(
                gray, width, height, kernelSize, sigma, &t_blur[i]
              )
            : cudaGaussianBlur(
                gray, width, height, kernelSize, sigma, &t_blur[i]
              );

        unsigned char* sobel = cudaSobel(blur, width, height, &t_sobel[i]);

        int currentOutWidth = width;
        int currentOutHeight = height;
        unsigned char* resized = cudaBilinearResize(
            sobel, width, height, scale, &currentOutWidth, &currentOutHeight, &t_resize[i]
        );

        std::cout << "  Repeticion " << (i + 1) << " completada." << std::endl;

        if (i == repeticiones - 1) {
            final_gray = gray;
            final_blur = blur;
            final_sobel = sobel;
            final_resized = resized;
            outWidth = currentOutWidth;
            outHeight = currentOutHeight;
        } else {
            delete[] gray;
            delete[] blur;
            delete[] sobel;
            if (resized != nullptr) delete[] resized;
        }
    }

    auto sumTotal = [](const std::vector<CudaStageTimings>& v) {
        float s = 0.0f;
        for (const auto& t : v) s += t.total_ms;
        return s;
    };
    auto sumKernel = [](const std::vector<CudaStageTimings>& v) {
        float s = 0.0f;
        for (const auto& t : v) s += t.kernel_ms;
        return s;
    };

    float promedio_total = sumTotal(t_gray) / repeticiones
                         + sumTotal(t_blur) / repeticiones
                         + sumTotal(t_sobel) / repeticiones
                         + sumTotal(t_resize) / repeticiones;
    float promedio_kernel = sumKernel(t_gray) / repeticiones
                          + sumKernel(t_blur) / repeticiones
                          + sumKernel(t_sobel) / repeticiones
                          + sumKernel(t_resize) / repeticiones;

    float sum_sq = 0.0f;
    for (int i = 0; i < repeticiones; ++i) {
        float t = t_gray[i].total_ms + t_blur[i].total_ms
                + t_sobel[i].total_ms + t_resize[i].total_ms;
        sum_sq += (t - promedio_total) * (t - promedio_total);
    }
    float desviacion = std::sqrt(sum_sq / repeticiones);

    std::cout << "--> Tiempo promedio total (con transfer): " << promedio_total << " ms\n";
    std::cout << "--> Tiempo promedio kernels (sin transfer): " << promedio_kernel << " ms\n";
    std::cout << "--> Desviacion estandar (total): " << desviacion << " ms\n";

    double pixels = static_cast<double>(width) * height;
    double throughput = (pixels / 1.0e6) / (promedio_kernel / 1000.0);
    std::cout << "--> Throughput kernels: " << throughput << " MP/s\n";

    std::string csvPath = "../results/resultados.csv";
    bool fileExists = fs::exists(csvPath);
    std::ofstream file(csvPath, std::ios::app);

    if (file.is_open()) {
        if (!fileExists) {
            file << "Version,Instancia,Imagen,DimensionesOrig,KernelSize,Scale,Bloque,Repeticion,"
                 << "T_Gray_kernel_ms,T_Blur_kernel_ms,T_Sobel_kernel_ms,T_Resize_kernel_ms,"
                 << "T_Gray_HtoD_ms,T_Gray_DtoH_ms,T_Blur_HtoD_ms,T_Blur_DtoH_ms,"
                 << "T_Sobel_HtoD_ms,T_Sobel_DtoH_ms,T_Resize_HtoD_ms,T_Resize_DtoH_ms,"
                 << "T_Total_ms,Throughput_MPps,Herramienta\n";
        }
        std::string dims = std::to_string(width) + "x" + std::to_string(height);
        for (int i = 0; i < repeticiones; ++i) {
            float t_gray_h = t_gray[i].hToD_ms;
            float t_gray_k = t_gray[i].kernel_ms;
            float t_gray_d = t_gray[i].dToH_ms;
            float t_blur_h = t_blur[i].hToD_ms;
            float t_blur_k = t_blur[i].kernel_ms;
            float t_blur_d = t_blur[i].dToH_ms;
            float t_sob_h = t_sobel[i].hToD_ms;
            float t_sob_k = t_sobel[i].kernel_ms;
            float t_sob_d = t_sobel[i].dToH_ms;
            float t_res_h = t_resize[i].hToD_ms;
            float t_res_k = t_resize[i].kernel_ms;
            float t_res_d = t_resize[i].dToH_ms;
            float t_total = t_gray[i].total_ms + t_blur[i].total_ms
                          + t_sobel[i].total_ms + t_resize[i].total_ms;

            double mpps = (pixels / 1.0e6) / (t_gray_k + t_blur_k + t_sob_k + t_res_k) * 1000.0;

            file << "CUDA_Clasico," << instance << "," << fs::path(input).filename().string() << ","
                 << dims << "," << kernelSize << "," << scale << "," << (separable ? "Dinamico_API_Separable" : "Dinamico_API") << "," << (i + 1) << ","
                 << t_gray_k << "," << t_blur_k << "," << t_sob_k << "," << t_res_k << ","
                 << t_gray_h << "," << t_gray_d << "," << t_blur_h << "," << t_blur_d << ","
                 << t_sob_h << "," << t_sob_d << "," << t_res_h << "," << t_res_d << ","
                 << t_total << "," << mpps << ",CUDA_Events\n";
        }
        file.close();
    } else {
        std::cerr << "Error al abrir el archivo CSV para guardar los resultados.\n";
    }

    std::string newName;
    bool save;

    newName = (route.parent_path() / (route.stem().string() + "_gray" + route.extension().string())).string();
    save = saveImage(newName.c_str(), final_gray, width, height);
    if (!save) std::cout << "Error guardando gray.\n";
    delete[] final_gray;

    newName = (route.parent_path() / (route.stem().string() + "_blur_k" + std::to_string(kernelSize) + route.extension().string())).string();
    save = saveImage(newName.c_str(), final_blur, width, height);
    if (!save) std::cout << "Error guardando blur.\n";
    delete[] final_blur;

    newName = (route.parent_path() / (route.stem().string() + "_sobel_k" + std::to_string(kernelSize) + route.extension().string())).string();
    save = saveImage(newName.c_str(), final_sobel, width, height);
    if (!save) std::cout << "Error guardando sobel.\n";
    delete[] final_sobel;

    if (final_resized != nullptr) {
        newName = (route.parent_path() / (route.stem().string() + "_resize_s" + std::to_string(scale) + route.extension().string())).string();
        save = saveImage(newName.c_str(), final_resized, outWidth, outHeight);
        if (!save) std::cout << "Error guardando resize.\n";
        delete[] final_resized;
    }

    freeImage(img);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Uso: cuda --instance=<small|medium|large|no-divisible> --kernel-size=<size> [--scale=<factor>] [--separable]\n";
        return 1;
    }

    std::string instance;
    int kernelSize = 0;
    float scale = 1.0f;
    bool separable = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.rfind("--instance=", 0) == 0) {
            instance = arg.substr(11);
        } else if (arg.rfind("--kernel-size=", 0) == 0) {
            kernelSize = std::stoi(arg.substr(14));
        } else if (arg.rfind("--scale=", 0) == 0) {
            scale = std::stof(arg.substr(8));
        } else if (arg == "--separable") {
            separable = true;
        }
    }

    if (instance != "small" && instance != "medium" && instance != "large" && instance != "no-divisible") return 1;
    if (kernelSize < 3 || kernelSize % 2 == 0) return 1;
    if (scale <= 0.0f) return 1;

    std::string directory = "../data/" + instance;
    std::string dirOutput = "../results/cuda/" + instance;

    if (!fs::exists(directory)) return 1;
    if (!fs::exists(dirOutput)) fs::create_directories(dirOutput);

    for (const auto& file : fs::directory_iterator(directory)) {
        if (!file.is_regular_file()) continue;
        std::string ext = file.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        std::string fileInput = file.path().string();
        std::string fileOutput = (fs::path(dirOutput) / file.path().filename()).string();

        processImageCuda(fileInput.c_str(), fileOutput.c_str(), kernelSize, scale, instance, separable);
    }

    return 0;
}
