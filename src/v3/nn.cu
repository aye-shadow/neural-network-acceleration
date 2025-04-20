#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// meow7
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

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

// Free allocated matrix memory (for non-pinned, normal matrices)
void freeMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Neural network structure for CPU
typedef struct {
    float** W1;
    float** W2;
    float* b1;
    float* b2;
} NeuralNetwork;

// Neural network structure for GPU
typedef struct {
    float* W1;  // Flattened matrix
    float* W2;  // Flattened matrix
    float* b1;
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
            net->W1[i][j] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX) * 0.01;
    return net;
}

// CUDA Kernels (unchanged)
__global__ void forwardHiddenKernel(float* input, float* W1, float* b1, float* hidden, int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        hidden[i] = __ldg(&b1[i]);
        for (int j = 0; j < inputSize; j++) {
            hidden[i] += __ldg(&W1[i * inputSize + j]) * __ldg(&input[j]);
        }
        hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
    }
}

__global__ void forwardOutputKernel(float* hidden, float* W2, float* b2, float* output, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize) {
        output[i] = __ldg(&b2[i]);
        for (int j = 0; j < hiddenSize; j++) {
            output[i] += __ldg(&W2[i * hiddenSize + j]) * __ldg(&hidden[j]);
        }
    }
}

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

__global__ void outputGradientKernel(float* output, float* target, float* d_output, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void hiddenGradientKernel(float* hidden, float* W2, float* d_output, float* d_hidden, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        d_hidden[i] = 0;
        for (int j = 0; j < outputSize; j++) {
            d_hidden[i] += W2[j * hiddenSize + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}

__global__ void updateOutputWeightsKernel(float* W2, float* b2, float* d_output, float* hidden, float learningRate, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < outputSize && j < hiddenSize) {
        W2[i * hiddenSize + j] -= learningRate * d_output[i] * hidden[j];
    }
    if (j == 0 && i < outputSize) {
        b2[i] -= learningRate * d_output[i];
    }
}

__global__ void updateHiddenWeightsKernel(float* W1, float* b1, float* d_hidden, float* input, float learningRate, int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < hiddenSize && j < inputSize) {
        W1[i * inputSize + j] -= learningRate * d_hidden[i] * input[j];
    }
    if (j == 0 && i < hiddenSize) {
        b1[i] -= learningRate * d_hidden[i];
    }
}

// Transfer neural network from CPU to GPU (unchanged)
NeuralNetworkGPU* transferNetworkToGPU(NeuralNetwork* cpuNet) {
    NeuralNetworkGPU* gpuNet = (NeuralNetworkGPU*)malloc(sizeof(NeuralNetworkGPU));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b1, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b2, OUTPUT_SIZE * sizeof(float)));
    float* flatW1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            flatW1[i * INPUT_SIZE + j] = cpuNet->W1[i][j];
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W1, flatW1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(flatW1);
    float* flatW2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            flatW2[i * HIDDEN_SIZE + j] = cpuNet->W2[i][j];
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W2, flatW2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(flatW2);
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b1, cpuNet->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b2, cpuNet->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    return gpuNet;
}

void transferNetworkToCPU(NeuralNetworkGPU* gpuNet, NeuralNetwork* cpuNet) {
    float* flatW1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* flatW2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(flatW1, gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flatW2, gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b1, gpuNet->b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b2, gpuNet->b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            cpuNet->W1[i][j] = flatW1[i * INPUT_SIZE + j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            cpuNet->W2[i][j] = flatW2[i * HIDDEN_SIZE + j];
    free(flatW1);
    free(flatW2);
}

void freeNetworkGPU(NeuralNetworkGPU* net) {
    CHECK_CUDA_ERROR(cudaFree(net->W1));
    CHECK_CUDA_ERROR(cudaFree(net->W2));
    CHECK_CUDA_ERROR(cudaFree(net->b1));
    CHECK_CUDA_ERROR(cudaFree(net->b2));
    free(net);
}

void forwardGPU(NeuralNetworkGPU* net, float* d_input, float* d_hidden, float* d_output) {
    int blockSize = 128;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    forwardHiddenKernel<<<hiddenGridSize, blockSize>>>(d_input, net->W1, net->b1, d_hidden, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    forwardOutputKernel<<<outputGridSize, blockSize>>>(d_hidden, net->W2, net->b2, d_output, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    softmaxKernel<<<1, 1>>>(d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void backwardGPU(NeuralNetworkGPU* net, float* d_input, float* d_hidden, float* d_output, float* d_target) {
    float *d_d_output, *d_d_hidden;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, HIDDEN_SIZE * sizeof(float)));
    int blockSize = 128;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    outputGradientKernel<<<outputGridSize, blockSize>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    hiddenGradientKernel<<<hiddenGridSize, blockSize>>>(d_hidden, net->W2, d_d_output, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 outputBlockDim(16, 16);
    dim3 outputGridDim((OUTPUT_SIZE + outputBlockDim.x - 1) / outputBlockDim.x,
                       (HIDDEN_SIZE + outputBlockDim.y - 1) / outputBlockDim.y);
    dim3 hiddenBlockDim(16, 16);
    dim3 hiddenGridDim((HIDDEN_SIZE + hiddenBlockDim.x - 1) / hiddenBlockDim.x,
                      (INPUT_SIZE + hiddenBlockDim.y - 1) / hiddenBlockDim.y);
    updateOutputWeightsKernel<<<outputGridDim, outputBlockDim>>>(
        net->W2, net->b2, d_d_output, d_hidden, LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    updateHiddenWeightsKernel<<<hiddenGridDim, hiddenBlockDim>>>(
        net->W1, net->b1, d_d_hidden, d_input, LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaFree(d_d_output));
    CHECK_CUDA_ERROR(cudaFree(d_d_hidden));
}

void trainGPU(NeuralNetwork* cpuNet, NeuralNetworkGPU* gpuNet, float** images, float** labels, int numImages) {
    clock_t total_start = clock();

    float *d_input, *d_hidden, *d_output, *d_target;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_target, OUTPUT_SIZE * sizeof(float)));

    float* h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Create a stream for async operations
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Async memory copies
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, images[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_target, labels[i], OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
            // Forward and backward pass (default stream)
            forwardGPU(gpuNet, d_input, d_hidden, d_output);
            backwardGPU(gpuNet, d_input, d_hidden, d_output, d_target);
            // Async copy output back to host
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
            // Synchronize to ensure output is ready before access
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
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
    transferNetworkToCPU(gpuNet, cpuNet);

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    free(h_output);

    // Destroy the stream
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

void evaluateGPU(NeuralNetworkGPU* gpuNet, float** images, float** labels, int numImages) {
    int correct = 0;
    float *d_input, *d_hidden, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));
    float* h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < numImages; i++) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        forwardGPU(gpuNet, d_input, d_hidden, d_output);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_output);
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

int main() {
    printf("MNIST Neural Network - Optimized GPU Implementation (V3)\n\n");

    // Pinned-memory blocks for images/labels
    float *train_images_block, *train_labels_block, *test_images_block, *test_labels_block;

    // Load MNIST dataset using pinned memory
    float** train_images = loadMNISTImagesPinned("../data/train-images.idx3-ubyte", 60000, &train_images_block);
    float** train_labels = loadMNISTLabelsPinned("../data/train-labels.idx1-ubyte", 60000, &train_labels_block);
    float** test_images  = loadMNISTImagesPinned("../data/t10k-images.idx3-ubyte", 10000, &test_images_block);
    float** test_labels  = loadMNISTLabelsPinned("../data/t10k-labels.idx1-ubyte", 10000, &test_labels_block);

    NeuralNetwork* cpuNet = createNetwork();
    NeuralNetworkGPU* gpuNet = transferNetworkToGPU(cpuNet);

    trainGPU(cpuNet, gpuNet, train_images, train_labels, 60000);
    evaluateGPU(gpuNet, test_images, test_labels, 10000);

    freeNetwork(cpuNet);
    freeNetworkGPU(gpuNet);
    freePinnedMatrix(train_images, train_images_block);
    freePinnedMatrix(train_labels, train_labels_block);
    freePinnedMatrix(test_images, test_images_block);
    freePinnedMatrix(test_labels, test_labels_block);

    return 0;
}