# Define the compiler
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -c -g $(shell pkg-config --cflags --libs opencv4)
NVCCFLAGS := -c -g -arch=compute_89 -code=sm_89 -allow-unsupported-compiler
LDFLAGS := -L/usr/local/cuda/lib64/ -lcudart -lcuda $(shell pkg-config --cflags --libs opencv4)

# Source files
CXX_SRC := main.cpp
NVCC_SRC := kernel.cu

# Object files
CXX_OBJ := .build/main.o
NVCC_OBJ := .build/kernel.o

# Executable
TARGET := main

# Build rules
all: $(TARGET)

$(TARGET): $(CXX_OBJ) $(NVCC_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(CXX_OBJ): $(CXX_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(NVCC_OBJ): $(NVCC_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f $(TARGET) $(CXX_OBJ) $(NVCC_OBJ)
