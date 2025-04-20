# Makefile for CUDA Vector Addition example

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_86

# If you are using a newer GPU architecture, you may want to update this value
# Common values: sm_35, sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89
# Check your GPU compute capability and adjust accordingly

# Target executable
TARGET = test

# Source files
SOURCES = test.cu

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)

# Debug build with debug info
debug: NVCC_FLAGS += -g -G
debug: all

.PHONY: all clean run debug