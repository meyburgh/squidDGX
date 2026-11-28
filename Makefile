CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/squidDGX
TEST_BIN  := $(BUILD_DIR)/test_unit

# GPU targets (space-separated SM numbers; an "a" suffix selects the arch-specific
# target needed for hardware FP4 conversion). Defaults cover:
#   86   A10G, RTX A-series          89   L4, L40S, RTX 6000 Ada
#   90   H100, H200                  120a RTX PRO 4500/6000 Blackwell, RTX 50xx
#   121a DGX Spark (GB10)
# Build only what you need for faster compiles, e.g. `make CUDA_ARCHS=89`.
CUDA_ARCHS ?= 86 89 90 120a 121a
GENCODE   := $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

# Host CPU flags: native on ARM64 (Spark builds in place), portable AVX2 baseline on x86-64
# so an image built on one cloud host runs on another. Override with HOST_ARCH_FLAGS=...
UNAME_M   := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
HOST_ARCH_FLAGS ?= -mcpu=native
else
HOST_ARCH_FLAGS ?= -march=x86-64-v3
endif

CXXFLAGS  := -O3 -std=c++20 $(HOST_ARCH_FLAGS) -Icuda -Isrc -Ithird_party -I$(CUDA_PATH)/include \
             -Wall -Wextra -Wno-unused-parameter -fno-exceptions -fno-rtti
NVCCFLAGS := -O3 --use_fast_math -std=c++20 $(GENCODE) --threads 0 \
             --expt-relaxed-constexpr -Icuda
LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lwebp -lcurl -lpthread -lm

CUDA_HDRS := cuda/config.hpp cuda/weights.hpp cuda/kernels.hpp cuda/kernels_common.cuh
CUDA_SRCS := cuda/kernels_gemv.cu cuda/kernels_inkling.cu cuda/kernels_dspark.cu \
             cuda/kernels_prefill.cu cuda/kernels_vision.cu cuda/kernels_utils.cu
CUDA_OBJS := $(patsubst cuda/%.cu,$(BUILD_DIR)/cu_%.o,$(CUDA_SRCS))

CPP_HDRS  := src/tiger.hpp src/vram_arena.hpp src/safetensors.hpp src/weights.hpp src/json_constraint.hpp \
             src/tokenizer.hpp src/vision.hpp src/engine.hpp src/server.hpp
CPP_SRCS  := src/safetensors.cpp src/weights.cpp src/tokenizer.cpp src/json_constraint.cpp \
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

$(TEST_BIN): tests/test_unit.cpp src/tokenizer.cpp src/json_constraint.cpp src/json_constraint.hpp src/tiger.hpp src/tokenizer.hpp src/vram_arena.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_unit.cpp src/tokenizer.cpp src/json_constraint.cpp -o $@ -lpthread -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
