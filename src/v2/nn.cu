#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
//meow7
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Neural network structure for CPU
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Neural network structure for GPU
typedef struct {
    double* W1;  // Flattened matrix
    double* W2;  // Flattened matrix
    double* b1;
    double* b2;
} NeuralNetworkGPU;

// Initialize neural network on CPU
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// CUDA kernel for forward pass hidden layer computation
__global__ void forwardHiddenKernel(double* input, double* W1, double* b1, double* hidden, int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hiddenSize) {
        hidden[i] = b1[i];
        for (int j = 0; j < inputSize; j++) {
            hidden[i] += W1[i * inputSize + j] * input[j];
        }
        // ReLU activation
        hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
    }
}

// CUDA kernel for forward pass output layer computation
__global__ void forwardOutputKernel(double* hidden, double* W2, double* b2, double* output, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < outputSize) {
        output[i] = b2[i];
        for (int j = 0; j < hiddenSize; j++) {
            output[i] += W2[i * hiddenSize + j] * hidden[j];
        }
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxKernel(double* output, int outputSize) {
    // First find the maximum value for numerical stability
    double maxVal = output[0];
    for (int i = 1; i < outputSize; i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
        }
    }
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i] - maxVal);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < outputSize; i++) {
        output[i] /= sum;
    }
}

// CUDA kernel for computing output layer gradients
__global__ void outputGradientKernel(double* output, double* target, double* d_output, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < outputSize) {
        d_output[i] = output[i] - target[i];
    }
}

// CUDA kernel for computing hidden layer gradients
__global__ void hiddenGradientKernel(double* hidden, double* W2, double* d_output, double* d_hidden, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < hiddenSize) {
        d_hidden[i] = 0;
        for (int j = 0; j < outputSize; j++) {
            d_hidden[i] += W2[j * hiddenSize + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0); // ReLU derivative
    }
}

// CUDA kernel for updating output layer weights
__global__ void updateOutputWeightsKernel(double* W2, double* b2, double* d_output, double* hidden, double learningRate, int hiddenSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < outputSize && j < hiddenSize) {
        W2[i * hiddenSize + j] -= learningRate * d_output[i] * hidden[j];
    }
    
    // Update biases (only threads with j=0 to avoid redundant updates)
    if (j == 0 && i < outputSize) {
        b2[i] -= learningRate * d_output[i];
    }
}

// CUDA kernel for updating hidden layer weights
__global__ void updateHiddenWeightsKernel(double* W1, double* b1, double* d_hidden, double* input, double learningRate, int inputSize, int hiddenSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hiddenSize && j < inputSize) {
        W1[i * inputSize + j] -= learningRate * d_hidden[i] * input[j];
    }
    
    // Update biases (only threads with j=0 to avoid redundant updates)
    if (j == 0 && i < hiddenSize) {
        b1[i] -= learningRate * d_hidden[i];
    }
}

// Transfer neural network from CPU to GPU
NeuralNetworkGPU* transferNetworkToGPU(NeuralNetwork* cpuNet) {
    NeuralNetworkGPU* gpuNet = (NeuralNetworkGPU*)malloc(sizeof(NeuralNetworkGPU));
    
    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&gpuNet->b2, OUTPUT_SIZE * sizeof(double)));
    
    // Flatten W1 and copy to GPU
    double* flatW1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            flatW1[i * INPUT_SIZE + j] = cpuNet->W1[i][j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W1, flatW1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    free(flatW1);
    
    // Flatten W2 and copy to GPU
    double* flatW2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            flatW2[i * HIDDEN_SIZE + j] = cpuNet->W2[i][j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->W2, flatW2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    free(flatW2);
    
    // Copy biases to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b1, cpuNet->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpuNet->b2, cpuNet->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    return gpuNet;
}

// Transfer updated weights from GPU back to CPU
void transferNetworkToCPU(NeuralNetworkGPU* gpuNet, NeuralNetwork* cpuNet) {
    // Allocate temporary arrays for flattened weights
    double* flatW1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* flatW2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    // Copy weights from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(flatW1, gpuNet->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flatW2, gpuNet->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Copy biases from GPU
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b1, gpuNet->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpuNet->b2, gpuNet->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Unflatten weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            cpuNet->W1[i][j] = flatW1[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            cpuNet->W2[i][j] = flatW2[i * HIDDEN_SIZE + j];
        }
    }
    
    free(flatW1);
    free(flatW2);
}

// Free GPU network memory
void freeNetworkGPU(NeuralNetworkGPU* net) {
    CHECK_CUDA_ERROR(cudaFree(net->W1));
    CHECK_CUDA_ERROR(cudaFree(net->W2));
    CHECK_CUDA_ERROR(cudaFree(net->b1));
    CHECK_CUDA_ERROR(cudaFree(net->b2));
    free(net);
}

// Forward pass on GPU
void forwardGPU(NeuralNetworkGPU* net, double* d_input, double* d_hidden, double* d_output) {
    // Define block and grid dimensions
    int blockSize = 128;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    // Compute hidden layer activations
    forwardHiddenKernel<<<hiddenGridSize, blockSize>>>(d_input, net->W1, net->b1, d_hidden, INPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute output layer pre-activations
    forwardOutputKernel<<<outputGridSize, blockSize>>>(d_hidden, net->W2, net->b2, d_output, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply softmax activation (using a single thread for simplicity in this naive implementation)
    softmaxKernel<<<1, 1>>>(d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Backward pass on GPU
void backwardGPU(NeuralNetworkGPU* net, double* d_input, double* d_hidden, double* d_output, double* d_target) {
    // Allocate memory for gradients
    double *d_d_output, *d_d_hidden;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    
    // Define block and grid dimensions
    int blockSize = 128;
    int outputGridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    int hiddenGridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    
    // Compute output layer gradients
    outputGradientKernel<<<outputGridSize, blockSize>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute hidden layer gradients
    hiddenGradientKernel<<<hiddenGridSize, blockSize>>>(d_hidden, net->W2, d_d_output, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update weights
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
    
    // Free temporary memory
    CHECK_CUDA_ERROR(cudaFree(d_d_output));
    CHECK_CUDA_ERROR(cudaFree(d_d_hidden));
}

// Train network on GPU
void trainGPU(NeuralNetwork* cpuNet, NeuralNetworkGPU* gpuNet, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    // Allocate GPU memory for a single training example
    double *d_input, *d_hidden, *d_output, *d_target;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_target, OUTPUT_SIZE * sizeof(double)));
    
    // Allocate host memory for results
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            // Copy input and target to GPU
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_target, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            
            // Forward pass
            forwardGPU(gpuNet, d_input, d_hidden, d_output);
            
            // Backward pass
            backwardGPU(gpuNet, d_input, d_hidden, d_output, d_target);
            
            // Copy output back to host for loss and accuracy calculation
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            
            // Compute loss & accuracy
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
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    printf("Total training time: %.3fs\n", get_time(total_start));
    
    // Copy final weights back to CPU
    transferNetworkToCPU(gpuNet, cpuNet);
    
    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_target));
    free(h_output);
}

// Evaluate accuracy on test data using GPU
void evaluateGPU(NeuralNetworkGPU* gpuNet, double** images, double** labels, int numImages) {
    int correct = 0;
    
    // Allocate GPU memory
    double *d_input, *d_hidden, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    
    // Allocate host memory for results
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    for (int i = 0; i < numImages; i++) {
        // Copy input to GPU
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        
        // Forward pass
        forwardGPU(gpuNet, d_input, d_hidden, d_output);
        
        // Copy output back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    // Free memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_hidden));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_output);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
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

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network - GPU Implementation (V2)\n\n");
    
    // Load MNIST dataset
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    
    // Create and initialize network on CPU
    NeuralNetwork* cpuNet = createNetwork();
    
    // Transfer network to GPU
    NeuralNetworkGPU* gpuNet = transferNetworkToGPU(cpuNet);
    
    // Train network on GPU
    trainGPU(cpuNet, gpuNet, train_images, train_labels, 60000);
    
    // Evaluate on test data
    evaluateGPU(gpuNet, test_images, test_labels, 10000);
    
    // Free memory
    freeNetwork(cpuNet);
    freeNetworkGPU(gpuNet);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}