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
    const std::string& instance
) {
    int width, height, channels;

    unsigned char* img = loadImage(input, &width, &height, &channels);
    if (img == nullptr) return 1;

    fs::path route(output);
    float sigma = (kernelSize - 1) / 6.0f;

    int repeticiones = 10;
    std::vector<float> tiempos_totales(repeticiones, 0.0f);
    std::vector<float> tiempos_gray(repeticiones, 0.0f);
    std::vector<float> tiempos_blur(repeticiones, 0.0f);
    std::vector<float> tiempos_sobel(repeticiones, 0.0f);
    std::vector<float> tiempos_resize(repeticiones, 0.0f);

    std::cout << "\nProcesando imagen: " << fs::path(input).filename().string() << std::endl;

    // --- CALENTAMIENTO (WARMUP) ---
    // Ejecutamos una vez sin medir para despertar la GPU y cargar el contexto CUDA
    std::cout << "Realizando iteracion de calentamiento de GPU...\n";
    unsigned char* w_gray = cudaRgbToGray(img, width, height, channels);
    unsigned char* w_blur = cudaGaussianBlur(w_gray, width, height, kernelSize, sigma);
    unsigned char* w_sobel = cudaSobel(w_blur, width, height);
    int w_outW, w_outH;
    unsigned char* w_resize = cudaBilinearResize(w_sobel, width, height, scale, &w_outW, &w_outH);
    
    // Liberamos la memoria del calentamiento
    delete[] w_gray; delete[] w_blur; delete[] w_sobel; 
    if(w_resize != nullptr) delete[] w_resize;

    std::cout << "Iniciando " << repeticiones << " repeticiones de medicion...\n";

    unsigned char* final_gray = nullptr;
    unsigned char* final_blur = nullptr;
    unsigned char* final_sobel = nullptr;
    unsigned char* final_resized = nullptr;
    int outWidth = width;
    int outHeight = height;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeticiones; ++i) {
        float ms_gray = 0, ms_blur = 0, ms_sobel = 0, ms_resize = 0;

        // =================================================================
        // ETAPA 1: Escala de Grises
        // =================================================================
        cudaEventRecord(start);
        unsigned char* gray = cudaRgbToGray(img, width, height, channels);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_gray, start, stop);

        // =================================================================
        // ETAPA 2: Gaussian Blur
        // =================================================================
        cudaEventRecord(start);
        unsigned char* blur = cudaGaussianBlur(gray, width, height, kernelSize, sigma);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_blur, start, stop);

        // =================================================================
        // ETAPA 3: Sobel
        // =================================================================
        cudaEventRecord(start);
        unsigned char* sobel = cudaSobel(blur, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_sobel, start, stop);

        // =================================================================
        // ETAPA 4: Resize Bilineal
        // =================================================================
        int currentOutWidth = width;
        int currentOutHeight = height;
        unsigned char* resized = nullptr;
        
        cudaEventRecord(start);
        resized = cudaBilinearResize(sobel, width, height, scale, &currentOutWidth, &currentOutHeight);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_resize, start, stop);

        // Guardar los tiempos separados
        tiempos_gray[i] = ms_gray;
        tiempos_blur[i] = ms_blur;
        tiempos_sobel[i] = ms_sobel;
        tiempos_resize[i] = ms_resize;
        tiempos_totales[i] = ms_gray + ms_blur + ms_sobel + ms_resize;

        std::cout << "  Repeticion " << (i + 1) << " completada.\n";

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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- CÁLCULO DE PROMEDIO Y DESVIACIÓN ESTÁNDAR ---
    float suma = std::accumulate(tiempos_totales.begin(), tiempos_totales.end(), 0.0f);
    float promedio = suma / repeticiones;

    float suma_varianza = 0.0f;
    for (float t : tiempos_totales) {
        suma_varianza += (t - promedio) * (t - promedio);
    }
    float desviacion = std::sqrt(suma_varianza / repeticiones);

    std::cout << "--> Tiempo promedio total: " << promedio << " ms\n";
    std::cout << "--> Desviacion estandar: " << desviacion << " ms\n";

    // --- EXPORTAR A CSV CON TODAS LAS MÉTRICAS DE LA RÚBRICA ---
    std::string csvPath = "../results/resultados.csv";
    bool fileExists = fs::exists(csvPath);
    std::ofstream file(csvPath, std::ios::app);
    
    if (file.is_open()) {
        if (!fileExists) {
            // Cabecera completa según rúbrica
            file << "Version,Instancia,Imagen,DimensionesOrig,KernelSize,Scale,Bloque,Repeticion,T_Gray_ms,T_Blur_ms,T_Sobel_ms,T_Resize_ms,T_Total_ms,Herramienta\n";
        }
        std::string dims = std::to_string(width) + "x" + std::to_string(height);
        for (int i = 0; i < repeticiones; ++i) {
            // Notar el campo "Dinamico_API" que reemplaza al antiguo tamaño estático de bloque
            file << "CUDA_Clasico," << instance << "," << fs::path(input).filename().string() << ","
                 << dims << "," << kernelSize << "," << scale << "," << "Dinamico_API," << (i + 1) << ","
                 << tiempos_gray[i] << "," << tiempos_blur[i] << "," << tiempos_sobel[i] << "," 
                 << tiempos_resize[i] << "," << tiempos_totales[i] << ",CUDA_Events\n";
        }
        file.close();
    } else {
        std::cerr << "Error al abrir el archivo CSV para guardar los resultados.\n";
    }

    // --- GUARDADO DE IMÁGENES ---
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
        std::cout << "Uso: cuda --instance=<small|medium|large|no-divisible> --kernel-size=<size> [--scale=<factor>]\n";
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
    std::string dirOutput = "../results/cuda/" + instance;

    if (!fs::exists(directory)) return 1;
    if (!fs::exists(dirOutput)) fs::create_directories(dirOutput);

    for (const auto& file : fs::directory_iterator(directory)) {
        if (!file.is_regular_file()) continue;
        std::string ext = file.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        std::string fileInput = file.path().string();
        std::string fileOutput = (fs::path(dirOutput) / file.path().filename()).string();

        processImageCuda(fileInput.c_str(), fileOutput.c_str(), kernelSize, scale, instance);
    }

    return 0;
}