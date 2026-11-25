CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/squidDGX
TEST_BIN  := $(BUILD_DIR)/test_unit

CXXFLAGS  := -O3 -std=c++20 -mcpu=native -Icuda -Isrc -Ithird_party -I$(CUDA_PATH)/include \
             -Wall -Wextra -Wno-unused-parameter -fno-exceptions -fno-rtti
NVCCFLAGS := -O3 --use_fast_math -std=c++20 -gencode arch=compute_121a,code=sm_121a \
             --expt-relaxed-constexpr -Icuda
LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lwebp -lcurl -lpthread -lm

CUDA_HDRS := cuda/config.hpp cuda/weights.hpp cuda/kernels.hpp cuda/kernels_common.cuh
CUDA_SRCS := cuda/kernels_gemv.cu cuda/kernels_inkling.cu cuda/kernels_dspark.cu \
             cuda/kernels_prefill.cu cuda/kernels_vision.cu cuda/kernels_utils.cu
CUDA_OBJS := $(patsubst cuda/%.cu,$(BUILD_DIR)/cu_%.o,$(CUDA_SRCS))

CPP_HDRS  := src/tiger.hpp src/vram_arena.hpp src/safetensors.hpp src/weights.hpp \
             src/tokenizer.hpp src/vision.hpp src/engine.hpp src/server.hpp
CPP_SRCS  := src/safetensors.cpp src/weights.cpp src/tokenizer.cpp \
             src/vision.cpp src/engine.cpp src/server.cpp src/main.cpp
CPP_OBJS  := $(patsubst src/%.cpp,$(BUILD_DIR)/cpp_%.o,$(CPP_SRCS))

.PHONY: all test clean

all: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cu_%.o: cuda/%.cu $(CUDA_HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/cpp_%.o: src/%.cpp $(CPP_HDRS) $(CUDA_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(CUDA_OBJS) $(CPP_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_BIN): tests/test_unit.cpp src/tokenizer.cpp src/tiger.hpp src/tokenizer.hpp src/vram_arena.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_unit.cpp src/tokenizer.cpp -o $@ -lpthread -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
