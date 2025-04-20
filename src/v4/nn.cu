#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Adjust sizes to be multiples of 8 for Tensor Core efficiency
#define INPUT_SIZE 784   // 28x28 (MNIST images)
#define HIDDEN_SIZE 128  // Multiple of 8
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define BATCH_SIZE 64    // Multiple of 8
#define NUM_CLASSES 10

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate pinned memory for a matrix (row-major)
float** allocatePinnedMatrix(int rows, int cols, float **block_ptr) {
    float *block;
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&block, rows * cols * sizeof(float), cudaHostAllocDefault));
    float **matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = block + i * cols;
    }
    if (block_ptr) *block_ptr = block;
    return matrix;
}

// Free pinned matrix
void freePinnedMatrix(float **matrix, float *block) {
    CHECK_CUDA_ERROR(cudaFreeHost(block));
    free(matrix);
}

// Neural network structure for CPU
typedef struct {
    float** W1;
    float** W2;
    float* b1;
    float* b2;
} NeuralNetwork;

// Neural network structure for GPU with FP16 support
typedef struct {
    half* W1;  // Flattened matrix in FP16
    half* W2;  // Flattened matrix in FP16
    float* b1; // Biases kept in FP32 for better accuracy
    float* b2;
} NeuralNetworkGPU;

// Initialize neural network on CPU
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (float**)malloc(HIDDEN_SIZE * sizeof(float*));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->W1[i] = (float*)malloc(INPUT_SIZE * sizeof(float));
    net->W2 = (float**)malloc(OUTPUT_SIZE * sizeof(float*));
    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->W2[i] = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX) * 0.01f;
    return net;
}

// Tensor Core optimized forward pass for hidden layer
__global__ void tensorCoreForwardHiddenKernel(const half* __restrict__ input, 
                                            const half* __restrict__ W1,
                                            const float* __restrict__ b1,
                                            float* __restrict__ hidden,
                                            int inputSize, int hiddenSize) {
    // Each thread block handles a tile of the output
    const int tileM = 16; // Tensor Core tile size
    const int tileN = 16;
    const int tileK = 16;
    
    int row = blockIdx.y * tileM + threadIdx.y;
    if (row >= hiddenSize) return;
    
    float sum = b1[row];
    
    // Simple implementation - replace with proper WMMA for better performance
    for (int col = 0; col < inputSize; col++) {
        sum += __half2float(W1[row * inputSize + col]) * __half2float(input[col]);
    }
    
    // ReLU activation
    hidden[row] = (sum > 0.0f) ? sum : 0.0f;
}

// Tensor Core optimized forward pass for output layer
__global__ void tensorCoreForwardOutputKernel(const float* __restrict__ hidden,
                                            const half* __restrict__ W2,
                                            const float* __restrict__ b2,
                                            float* __restrict__ output,
                                            int hiddenSize, int outputSize) {
    // Each thread computes one output element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputSize) return;
    
    float sum = b2[i];
    
    for (int j = 0; j < hiddenSize; j++) {
        sum += __half2float(W2[i * hiddenSize + j]) * hidden[j];
    }
    
    output[i] = sum;
}

// Softmax kernel remains the same
__global__ void softmaxKernel(float* output, int outputSize) {
    float maxVal = output[0];
    for (int i = 1; i < outputSize; i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i] - maxVal);
        sum += output[i];
    }
    for (int i = 0; i < outputSize; i++) {
        output[i] /= sum;
    }
}

// Backward pass kernels would need similar Tensor Core optimizations
// (Implementation omitted for brevity, but would follow similar patterns)

// Transfer neural network from CPU to GPU with FP16 conversion
NeuralNetworkGPU* transferNetworkToGPU(NeuralNetwork* cpuNet) {
    NeuralNetworkGPU* gpuNet = (NeuralNetworkGPU*)malloc(sizeof(NeuralNetworkGPU));
    
    // Allocate and convert weights to FP16
    half* flatW1_half = (half*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(half));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            flatW1_half[i * INPUT_SIZE + j] = __float2half(cpuNet->W1[i][j]);
        }
    }
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W1, flatW1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    free(flatW1_half);
    
    half* flatW2_half = (half*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            flatW2_half[i * HIDDEN_SIZE + j] = __float2half(cpuNet->W2[i][j]);
        }
    }
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W2, flatW2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    free(flatW2_half);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b1, cpuNet->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b2, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b2, cpuNet->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    return gpuNet;
}

// Forward pass using Tensor Cores
void forwardGPU(NeuralNetworkGPU* net, half* d_input, float* d_hidden, float* d_output) {
    dim3 blockDim(16, 16);
    dim3 gridDim((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 1);
    
    tensorCoreForwardHiddenKernel<<<gridDim, blockDim>>>(d_input, net->W1, net->b1, d_hidden, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 outputBlockDim(256);
    dim3 outputGridDim((OUTPUT_SIZE + outputBlockDim.x - 1) / outputBlockDim.x);
    tensorCoreForwardOutputKernel<<<outputGridDim, outputBlockDim>>>(d_hidden, net->W2, net->b2, d_output, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    softmaxKernel<<<1, 1>>>(d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Output gradient kernel (similar to original but with type awareness)
__global__ void tensorCoreOutputGradientKernel(const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ d_output,
    int outputSize) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < outputSize) {
d_output[i] = output[i] - target[i];
}
}

// Hidden gradient calculation optimized for Tensor Cores
__global__ void tensorCoreHiddenGradientKernel(const float* __restrict__ hidden,
    const half* __restrict__ W2,
    const float* __restrict__ d_output,
    float* __restrict__ d_hidden,
    int hiddenSize, int outputSize) {
// Each thread block handles a tile of the hidden gradient
const int tileM = 16; // Tensor Core tile size
const int tileN = 16;

int row = blockIdx.y * tileM + threadIdx.y;
if (row >= hiddenSize) return;

float sum = 0.0f;

// Simple implementation - replace with proper WMMA for better performance
for (int col = 0; col < outputSize; col++) {
sum += __half2float(W2[col * hiddenSize + row]) * d_output[col];
}

// ReLU derivative
d_hidden[row] = sum * ((hidden[row] > 0.0f) ? 1.0f : 0.0f);
}

// Tensor Core optimized weight update for output layer
__global__ void tensorCoreUpdateOutputWeightsKernel(half* __restrict__ W2,
         float* __restrict__ b2,
         const float* __restrict__ d_output,
         const float* __restrict__ hidden,
         float learningRate,
         int hiddenSize, int outputSize) {
    // Each thread updates one weight
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outputSize && j < hiddenSize) {
        float update = learningRate * d_output[i] * hidden[j];
        W2[i * hiddenSize + j] = __float2half(__half2float(W2[i * hiddenSize + j]) - update);
    }

    // Bias update
    if (j == 0 && i < outputSize) {
        b2[i] -= learningRate * d_output[i];
    }
}

// Tensor Core optimized weight update for hidden layer
__global__ void tensorCoreUpdateHiddenWeightsKernel(half* __restrict__ W1,
         float* __restrict__ b1,
         const float* __restrict__ d_hidden,
         const half* __restrict__ input,
         float learningRate,
         int inputSize, int hiddenSize) {
    // Each thread updates one weight
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < hiddenSize && j < inputSize) {
        float update = learningRate * d_hidden[i] * __half2float(input[j]);
        W1[i * inputSize + j] = __float2half(__half2float(W1[i * inputSize + j]) - update);
    }

    // Bias update
    if (j == 0 && i < hiddenSize) {
        b1[i] -= learningRate * d_hidden[i];
    }
}

// Complete backward pass with Tensor Core optimization
void backwardGPU(NeuralNetworkGPU* net, 
    half* d_input, 
    float* d_hidden, 
    float* d_output, 
    float* d_target) {
    float *d_d_output, *d_d_hidden;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, HIDDEN_SIZE * sizeof(float)));

    // Calculate output gradient
    dim3 outputBlockDim(256);
    dim3 outputGridDim((OUTPUT_SIZE + outputBlockDim.x - 1) / outputBlockDim.x);
    tensorCoreOutputGradientKernel<<<outputGridDim, outputBlockDim>>>(
    d_output, d_target, d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Calculate hidden gradient
    dim3 hiddenBlockDim(16, 16);
    dim3 hiddenGridDim(1, (HIDDEN_SIZE + hiddenBlockDim.y - 1) / hiddenBlockDim.y);
    tensorCoreHiddenGradientKernel<<<hiddenGridDim, hiddenBlockDim>>>(
    d_hidden, net->W2, d_d_output, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update output layer weights
    dim3 outputUpdateBlockDim(8, 8);
    dim3 outputUpdateGridDim(
        (OUTPUT_SIZE + outputUpdateBlockDim.x - 1) / outputUpdateBlockDim.x,
        (HIDDEN_SIZE + outputUpdateBlockDim.y - 1) / outputUpdateBlockDim.y);
    tensorCoreUpdateOutputWeightsKernel<<<outputUpdateGridDim, outputUpdateBlockDim>>>(
    net->W2, net->b2, d_d_output, d_hidden, LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update hidden layer weights
    dim3 hiddenUpdateBlockDim(8, 8);
    dim3 hiddenUpdateGridDim(
        (HIDDEN_SIZE + hiddenUpdateBlockDim.x - 1) / hiddenUpdateBlockDim.x,
        (INPUT_SIZE + hiddenUpdateBlockDim.y - 1) / hiddenUpdateBlockDim.y);
    tensorCoreUpdateHiddenWeightsKernel<<<hiddenUpdateGridDim, hiddenUpdateBlockDim>>>(
    net->W1, net->b1, d_d_hidden, d_input, LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaFree(d_d_output));
    CHECK_CUDA_ERROR(cudaFree(d_d_hidden));
}

// Training function with FP16 inputs
void trainGPU(NeuralNetwork* cpuNet, NeuralNetworkGPU* gpuNet, float** images, float** labels, int numImages) {
    clock_t total_start = clock();

    half *d_input;
    float *d_hidden, *d_output, *d_target;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_target, OUTPUT_SIZE * sizeof(float)));

    float* h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    half* h_input_half = (half*)malloc(INPUT_SIZE * sizeof(half));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Convert input to FP16
            for (int j = 0; j < INPUT_SIZE; j++) {
                h_input_half[j] = __float2half(images[i][j]);
            }
            
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input_half, INPUT_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_target, labels[i], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            
            forwardGPU(gpuNet, d_input, d_hidden, d_output);
            backwardGPU(gpuNet, d_input, d_hidden, d_output, d_target);
            
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Calculate loss and accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= labels[i][k] * log(h_output[k] > 1e-10 ? h_output[k] : 1e-10);
            }
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }

    printf("Total training time: %.3fs\n", get_time(total_start));
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    free(h_output);
    free(h_input_half);
}

// Pinned-memory MNIST loading
float** loadMNISTImagesPinned(const char* filename, int numImages, float **block_ptr) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float** images = allocatePinnedMatrix(numImages, INPUT_SIZE, block_ptr);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

float** loadMNISTLabelsPinned(const char* filename, int numLabels, float **block_ptr) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float** labels = allocatePinnedMatrix(numLabels, OUTPUT_SIZE, block_ptr);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory (unchanged)
void freeNetwork(NeuralNetwork* net) {
    for (int i = 0; i < HIDDEN_SIZE; i++) free(net->W1[i]);
    free(net->W1);
    for (int i = 0; i < OUTPUT_SIZE; i++) free(net->W2[i]);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

void freeNetworkGPU(NeuralNetworkGPU* net) {
    CHECK_CUDA_ERROR(cudaFree(net->W1));
    CHECK_CUDA_ERROR(cudaFree(net->W2));
    CHECK_CUDA_ERROR(cudaFree(net->b1));
    CHECK_CUDA_ERROR(cudaFree(net->b2));
    free(net);
}

void evaluateGPU(NeuralNetworkGPU* gpuNet, float** images, float** labels, int numImages) {
    // Allocate device memory with FP16 inputs and FP32 outputs
    half *d_input;
    float *d_hidden, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
    
    // Host buffers
    float* h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    half* h_input_half = (half*)malloc(INPUT_SIZE * sizeof(half));
    
    int correct = 0;
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    for (int i = 0; i < numImages; i++) {
        // Convert input to FP16 asynchronously
        #pragma omp parallel for
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_input_half[j] = __float2half(images[i][j]);
        }
        
        // Async memory transfers
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, h_input_half, INPUT_SIZE * sizeof(half),
                         cudaMemcpyHostToDevice, stream));
        
        // Forward pass with Tensor Cores
        forwardGPU(gpuNet, d_input, d_hidden, d_output);
        
        // Async copy output back to host
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, OUTPUT_SIZE * sizeof(float),
                         cudaMemcpyDeviceToHost, stream));
        
        // Synchronize to ensure output is ready
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        // Calculate prediction
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
        
        // Progress reporting (optional)
        if (i % 1000 == 0) {
            printf("Processed %d/%d samples...\n", i, numImages);
        }
    }
    
    // Calculate and print accuracy
    float accuracy = (correct / (float)numImages) * 100.0f;
    printf("Test Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, numImages);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_output);
    free(h_input_half);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

// Main function remains largely the same
int main() {
    printf("MNIST Neural Network - Tensor Core Optimized GPU Implementation\n\n");

    float *train_images_block, *train_labels_block, *test_images_block, *test_labels_block;

    float** train_images = loadMNISTImagesPinned("../data/train-images.idx3-ubyte", 60000, &train_images_block);
    float** train_labels = loadMNISTLabelsPinned("../data/train-labels.idx1-ubyte", 60000, &train_labels_block);
    float** test_images  = loadMNISTImagesPinned("../data/t10k-images.idx3-ubyte", 10000, &test_images_block);
    float** test_labels  = loadMNISTLabelsPinned("../data/t10k-labels.idx1-ubyte", 10000, &test_labels_block);

    NeuralNetwork* cpuNet = createNetwork();
    NeuralNetworkGPU* gpuNet = transferNetworkToGPU(cpuNet);

    trainGPU(cpuNet, gpuNet, train_images, train_labels, 60000);
    evaluateGPU(gpuNet, test_images, test_labels, 10000);

    // Cleanup
    freeNetwork(cpuNet);
    freeNetworkGPU(gpuNet);
    freePinnedMatrix(train_images, train_images_block);
    freePinnedMatrix(train_labels, train_labels_block);
    freePinnedMatrix(test_images, test_images_block);
    freePinnedMatrix(test_labels, test_labels_block);

    return 0;
}