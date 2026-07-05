CC      := nvcc
CXX     := nvcc
CXXFLAGS := -std=c++17 -arch=native
TILE_CXXFLAGS := -std=c++20 -arch=native -enable-tile

# En Windows se necesita --ccbin para localizar cl.exe; en Linux sobra
ifdef COMSPEC
CCBIN   := "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
CCBIN_FLAG := -ccbin $(CCBIN)
else
CCBIN_FLAG :=
endif

SRC_DIR := src
BUILD_DIR := build

HEADERS_CPU  := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h
HEADERS_CUDA := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h $(SRC_DIR)/cuda_kernels.hpp
HEADERS_TILE := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h $(SRC_DIR)/tile_kernels.hpp

.PHONY: all clean

all: $(BUILD_DIR)/secuencial.exe $(BUILD_DIR)/cuda.exe $(BUILD_DIR)/tile.exe

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/secuencial.exe: $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp $(HEADERS_CPU) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp -o $@

$(BUILD_DIR)/cuda.exe: $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp $(HEADERS_CUDA) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp -o $@

$(BUILD_DIR)/tile.exe: $(SRC_DIR)/image.cpp $(SRC_DIR)/tile_kernels.cu $(SRC_DIR)/main_tile.cpp $(HEADERS_TILE) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(TILE_CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/tile_kernels.cu $(SRC_DIR)/main_tile.cpp -o $@

clean:
	rm -rf $(BUILD_DIR)
