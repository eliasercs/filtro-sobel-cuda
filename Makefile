CC      := nvcc
CXX     := nvcc
CXXFLAGS := -std=c++17 -arch=native -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
TILE_CXXFLAGS := -std=c++20 -arch=native -enable-tile -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH

SRC_DIR    := src
BUILD_DIR  := build
SECUENCIAL := $(BUILD_DIR)/secuencial
CUDA_BIN   := $(BUILD_DIR)/cuda
TILE_BIN   := $(BUILD_DIR)/tile

ifdef COMSPEC
EXE_EXT := .exe
CCBIN   := "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
CCBIN_FLAG := -ccbin $(CCBIN)
else
EXE_EXT :=
CCBIN_FLAG :=
endif

HEADERS_CPU  := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h
HEADERS_CUDA := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h $(SRC_DIR)/cuda_kernels.hpp
HEADERS_TILE := $(SRC_DIR)/image.hpp $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h $(SRC_DIR)/tile_kernels.hpp

.PHONY: all clean

all: $(SECUENCIAL)$(EXE_EXT) $(CUDA_BIN)$(EXE_EXT) $(TILE_BIN)$(EXE_EXT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SECUENCIAL)$(EXE_EXT): $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp $(HEADERS_CPU) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/main.cpp -o $@

$(CUDA_BIN)$(EXE_EXT): $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp $(HEADERS_CUDA) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/main_cuda.cpp -o $@

$(TILE_BIN)$(EXE_EXT): $(SRC_DIR)/image.cpp $(SRC_DIR)/tile_kernels.cu $(SRC_DIR)/main_tile.cpp $(HEADERS_TILE) | $(BUILD_DIR)
	$(CC) $(CCBIN_FLAG) $(TILE_CXXFLAGS) $(SRC_DIR)/image.cpp $(SRC_DIR)/tile_kernels.cu $(SRC_DIR)/main_tile.cpp -o $@

clean:
	rm -rf $(BUILD_DIR)
