#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "image.hpp"

namespace fs = std::filesystem;
using clk = std::chrono::high_resolution_clock;

struct StageTimings {
    float total_ms;
};

int processImageCpu(
    const char* input,
    const char* output,
    int kernelSize,
    float scale,
    const std::string& instance
) {
    int width, height, channels;

    unsigned char* img = loadImage(input, &width, &height, &channels);
    if (img == nullptr) return 1;

    fs::path route(output);
    float sigma = (kernelSize - 1) / 6.0f;

    int repeticiones = 10;
    std::vector<StageTimings> t_gray(repeticiones);
    std::vector<StageTimings> t_blur(repeticiones);
    std::vector<StageTimings> t_sobel(repeticiones);
    std::vector<StageTimings> t_resize(repeticiones);

    std::cout << "\nProcesando imagen: " << fs::path(input).filename().string() << std::endl;

    std::cout << "Realizando iteracion de calentamiento de CPU...\n";
    StageTimings t_dummy;
    unsigned char* w_gray = convertToGrayScale(img, width, height, channels);
    unsigned char* w_blur = applyGaussianBlur(w_gray, width, height, kernelSize, sigma);
    unsigned char* w_sobel = applySobelFilter(w_blur, width, height);
    int w_outW, w_outH;
    unsigned char* w_resize = applyBilinearResize(w_sobel, width, height, scale, &w_outW, &w_outH);
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
        auto t0 = clk::now();
        unsigned char* gray = convertToGrayScale(img, width, height, channels);
        auto t1 = clk::now();
        t_gray[i].total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        t0 = clk::now();
        unsigned char* blur = applyGaussianBlur(gray, width, height, kernelSize, sigma);
        t1 = clk::now();
        t_blur[i].total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        t0 = clk::now();
        unsigned char* sobel = applySobelFilter(blur, width, height);
        t1 = clk::now();
        t_sobel[i].total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        int currentOutWidth = width;
        int currentOutHeight = height;
        t0 = clk::now();
        unsigned char* resized = applyBilinearResize(sobel, width, height, scale, &currentOutWidth, &currentOutHeight);
        t1 = clk::now();
        t_resize[i].total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

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

    auto sumT = [](const std::vector<StageTimings>& v) {
        float s = 0.0f;
        for (const auto& t : v) s += t.total_ms;
        return s;
    };

    float promedio_total = (sumT(t_gray) + sumT(t_blur) + sumT(t_sobel) + sumT(t_resize)) / repeticiones;

    float sum_sq = 0.0f;
    for (int i = 0; i < repeticiones; ++i) {
        float t = t_gray[i].total_ms + t_blur[i].total_ms
                + t_sobel[i].total_ms + t_resize[i].total_ms;
        sum_sq += (t - promedio_total) * (t - promedio_total);
    }
    float desviacion = std::sqrt(sum_sq / repeticiones);

    double pixels = static_cast<double>(width) * height;
    double throughput = (pixels / 1.0e6) / (promedio_total / 1000.0);
    std::cout << "--> Tiempo promedio total: " << promedio_total << " ms\n";
    std::cout << "--> Desviacion estandar: " << desviacion << " ms\n";
    std::cout << "--> Throughput: " << throughput << " MP/s\n";

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
            float total_i = t_gray[i].total_ms + t_blur[i].total_ms
                          + t_sobel[i].total_ms + t_resize[i].total_ms;
            file << "CPU_Secuencial," << instance << "," << fs::path(input).filename().string() << ","
                 << dims << "," << kernelSize << "," << scale << "," << "N/A" << "," << (i + 1) << ","
                 << t_gray[i].total_ms << "," << t_blur[i].total_ms << ","
                 << t_sobel[i].total_ms << "," << t_resize[i].total_ms << ","
                 << "0,0,0,0,0,0,0,0,"
                 << total_i << "," << (pixels / 1.0e6) / (total_i / 1000.0) << ",chrono\n";
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
        std::cout << "Uso: secuencial --instance=<small|medium|large|no-divisible> --kernel-size=<size> [--scale=<factor>]\n";
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
        }
    }

    if (instance != "small" && instance != "medium" && instance != "large" && instance != "no-divisible") return 1;
    if (kernelSize < 3 || kernelSize % 2 == 0) return 1;
    if (scale <= 0.0f) return 1;

    std::string directory = "../data/" + instance;
    std::string dirOutput = "../results/secuencial/" + instance;

    if (!fs::exists(directory)) return 1;
    if (!fs::exists(dirOutput)) fs::create_directories(dirOutput);

    for (const auto& file : fs::directory_iterator(directory)) {
        if (!file.is_regular_file()) continue;
        std::string ext = file.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        std::string fileInput = file.path().string();
        std::string fileOutput = (fs::path(dirOutput) / file.path().filename()).string();

        processImageCpu(fileInput.c_str(), fileOutput.c_str(), kernelSize, scale, instance);
    }

    return 0;
}
