CC      := nvcc
CXX     := nvcc
CXXFLAGS := -std=c++17 -arch=native

CCBIN   := "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"

SRC_DIR := src
BUILD_DIR := build

HEADERS_CPU := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h
HEADERS_CUDA := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h $(SRC_DIR)/cuda_kernels.hpp

.PHONY: all clean

all: $(BUILD_DIR)/secuencial.exe $(BUILD_DIR)/cuda.exe

$(BUILD_DIR):
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

$(BUILD_DIR)/secuencial.exe: $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp $(HEADERS_CPU) | $(BUILD_DIR)
	$(CC) -ccbin $(CCBIN) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp -o $@

$(BUILD_DIR)/cuda.exe: $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp $(HEADERS_CUDA) | $(BUILD_DIR)
	$(CC) -ccbin $(CCBIN) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp -o $@

clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
