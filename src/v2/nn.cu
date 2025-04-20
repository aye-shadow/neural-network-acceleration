#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    printf("Allocating matrix...\n");

    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    printf("Freeing matrix...\n");

    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
__global__ void reluOnGPU(double* x, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void expKernel(double* x, double* sum, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] = exp(x[idx]);
        atomicAdd(sum, x[idx]);
    }
}
__global__ void normalizeKernel(double* x, double* sum, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] /= *sum;
    }
}
void softmaxOnGPU(double* x, int size) {
    // printf("Applying softmax activation function...\n");

    double* d_sum;
    cudaMalloc((void**)&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    dim3 blockSize(256);
    dim3 numBlocks((size + blockSize.x - 1) / blockSize.x);

    expKernel<<<numBlocks, blockSize>>>(x, d_sum, size);
    cudaDeviceSynchronize();

    normalizeKernel<<<numBlocks, blockSize>>>(x, d_sum, size);
    cudaDeviceSynchronize();

    cudaFree(d_sum);

    // printf("Done with softmax activation function...\n");
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
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

// Forward pass
__global__ void forwardPassPart1OnGPU(double* input, double* hidden, double* W1, double* b1, int inputSize, int hiddenSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenSize) {
        hidden[idx] = b1[idx];
        for (int j = 0; j < inputSize; j++)
            hidden[idx] += W1[idx * inputSize + j] * input[j];
    }
}
__global__ void forwardPassPart2OnGPU(double* hidden, double* output, double* W2, double* b2, int hiddenSize, int outputSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < outputSize) {
        output[idx] = b2[idx];
        for (int j = 0; j < hiddenSize; j++)
            output[idx] += W2[idx * hiddenSize + j] * hidden[j];
    }
}
double* flattenNeuralNetMatrices(double** neuralNetMatrix, int rows, int cols) {
    // printf("Flattening neural net matrices...\n");

    double* flatMatrix = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            flatMatrix[i * cols + j] = neuralNetMatrix[i][j];
    return flatMatrix;
}
void forwardPassOnGPU(double* W1, double* b1, double* W2, double* b2, double* input, double* hidden, double* output) {
    // printf("Forward passing (P1) on GPU...\n");

    dim3 blockSize(256);
    dim3 numBlocks((HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    forwardPassPart1OnGPU<<<numBlocks, blockSize>>>(input, hidden, W1, b1, INPUT_SIZE, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    // printf("Applying ReLU activation function...\n");

    reluOnGPU<<<numBlocks, 256>>>(hidden, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    // printf("Forward passing (P2) on GPU...\n");

    numBlocks = (OUTPUT_SIZE + blockSize.x - 1) / blockSize.x;
    forwardPassPart2OnGPU<<<numBlocks, blockSize>>>(hidden, output, W2, b2, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    softmaxOnGPU(output, OUTPUT_SIZE);

    // printf("Done with forward pass...\n");
}


// Backpropagation
__global__ void outputLayerGradient(double* d_output, double* target, double* output, int size) {
    // for (int i = 0; i < OUTPUT_SIZE; i++)
    //     h_output[i] = output[i] - target[i];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}
__global__ void hiddenLayerGradient(double* W2, double* d_output, double* d_hidden, double* hidden, int hidden_size, int output_size) {
    // for (int i = 0; i < HIDDEN_SIZE; i++) {
    //     h_hidden[i] = 0;
    //     for (int j = 0; j < OUTPUT_SIZE; j++)
    //         h_hidden[i] += net->W2[j][i] * h_output[j];
    //     h_hidden[i] *= (hidden[i] > 0);
    // }

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < hidden_size) {
        d_hidden[i] = 0;
        for (int j = 0; j < output_size; j++) {
            d_hidden[i] += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}
__global__ void updateWeightsPart1(double* W, double* d_output, double* hidden, int size) {
    // for (int i = 0; i < OUTPUT_SIZE; i++)
    //     for (int j = 0; j < HIDDEN_SIZE; j++)
    //         net->W2[i][j] -= LEARNING_RATE * h_output[i] * hidden[j];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W[idx * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[idx] * hidden[j];
        }
    }
}
__global__ void updateWeightsPart2(double* W, double* d_hidden, double* input, int size) {
    // for (int i = 0; i < HIDDEN_SIZE; i++)
    //     for (int j = 0; j < INPUT_SIZE; j++)
    //         net->W1[i][j] -= LEARNING_RATE * h_hidden[i] * input[j];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            W[idx * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[idx] * input[j];
        }
    }
}
__global__ void updateBiasesPart1(double* b, double* d_output, int size) {
    // for (int i = 0; i < OUTPUT_SIZE; i++)
    //     net->b2[i] -= LEARNING_RATE * h_output[i];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        b[idx] -= LEARNING_RATE * d_output[idx];
    }
}
__global__ void updateBiasesPart2(double* b, double* d_hidden, int size) {
    // for (int i = 0; i < HIDDEN_SIZE; i++)
    //     net->b1[i] -= LEARNING_RATE * h_hidden[i];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        b[idx] -= LEARNING_RATE * d_hidden[idx];
    }
}
void backwardPassOnGPU(double* W1, double* b1, double* W2, double* b2, double* input, double* hidden, double* output, double* target) {
    // printf("Backward passing on GPU...\n");

    double h_output[OUTPUT_SIZE], h_hidden[HIDDEN_SIZE];

    double* d_output, *d_hidden;
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMemcpy(d_output, h_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, h_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((OUTPUT_SIZE + blockSize.x - 1) / blockSize.x);
    outputLayerGradient<<<numBlocks, blockSize>>>(d_output, target, output, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    numBlocks = (HIDDEN_SIZE + blockSize.x - 1) / blockSize.x;
    hiddenLayerGradient<<<numBlocks, blockSize>>>(W2, d_output, d_hidden, hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    numBlocks = (OUTPUT_SIZE + blockSize.x - 1) / blockSize.x;
    updateWeightsPart1<<<numBlocks, blockSize>>>(W2, d_output, hidden, OUTPUT_SIZE);
    cudaDeviceSynchronize();
    
    numBlocks = (HIDDEN_SIZE + blockSize.x - 1) / blockSize.x;
    updateWeightsPart2<<<numBlocks, blockSize>>>(W1, d_hidden, input, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    numBlocks = (OUTPUT_SIZE + blockSize.x - 1) / blockSize.x;
    updateBiasesPart1<<<numBlocks, blockSize>>>(b2, d_output, OUTPUT_SIZE);
    cudaDeviceSynchronize();

    numBlocks = (HIDDEN_SIZE + blockSize.x - 1) / blockSize.x;
    updateBiasesPart2<<<numBlocks, blockSize>>>(b1, d_hidden, HIDDEN_SIZE);
    cudaDeviceSynchronize();
    
    cudaFree(d_output);
    cudaFree(d_hidden);
}

// Train network
double* moveImageToGPU(double* image, int numImages) {
    double* d_image;
    cudaMalloc((void**)&d_image, INPUT_SIZE * sizeof(double));
    cudaMemcpy(d_image, image, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    return d_image;
}
double* moveLabelToGPU(double* label, int numLabels) {
    double* d_label;
    cudaMalloc((void**)&d_label, OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(d_label, label, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    return d_label;
}
void trainOnGPU(double* d_W1, double* d_W2, double* d_b1, double* d_b2, double* d_hidden, double* d_output, double** images, double** labels, int numImages) {
    printf("Training on GPU...\n");

    cudaEvent_t total_start, total_end;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);

    cudaEventRecord(total_start);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_end;
        cudaEventCreate(&epoch_start);
        cudaEventCreate(&epoch_end);

        cudaEventRecord(epoch_start);
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            // move images[i] and labels[i] to GPU
            double* d_image = moveImageToGPU(images[i], numImages);
            double* d_label = moveLabelToGPU(labels[i], numImages);

            double h_hidden[HIDDEN_SIZE], h_output[OUTPUT_SIZE];

            forwardPassOnGPU(d_W1, d_b1, d_W2, d_b2, d_image, d_hidden, d_output);
            backwardPassOnGPU(d_W1, d_b1, d_W2, d_b2, d_image, d_hidden, d_output, d_label);

            cudaMemcpy(h_hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

            cudaMemset(d_output, 0, OUTPUT_SIZE * sizeof(double));
            cudaMemset(d_hidden, 0, HIDDEN_SIZE * sizeof(double));
            cudaFree(d_image);
            cudaFree(d_label);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) 
                loss -= labels[i][k] * log(h_output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        cudaEventRecord(epoch_end);
        cudaEventSynchronize(epoch_end);

        float epoch_time;
        cudaEventElapsedTime(&epoch_time, epoch_start, epoch_end);
        
        cudaEventDestroy(epoch_start);
        cudaEventDestroy(epoch_end);

        // Print epoch results
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, epoch_time / 1000);
    }

    cudaEventRecord(total_end);
    cudaEventSynchronize(total_end);

    float total_time;
    cudaEventElapsedTime(&total_time, total_start, total_end);

    cudaEventDestroy(total_start);
    cudaEventDestroy(total_end);

    printf("Total training time: %.3f ms\n", total_time);
}

// Evaluate accuracy on test data
void evaluateOnGPU(double* d_W1, double* d_W2, double* d_b1, double* d_b2, double* d_hidden, double* d_output, double** images, double** labels, int numImages) {
    printf("Evaluting on GPU...\n");

    int correct = 0;

    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

        double* d_image = moveImageToGPU(images[i], numImages);
        forwardPassOnGPU(d_W1, d_b1, d_W2, d_b2, d_image, d_hidden, d_output);
        cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_image);
        cudaFree(d_hidden);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) 
                pred = j;
            if (labels[i][j] > labels[i][actual]) 
                actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    printf("Loading MNIST images...\n");

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

            // fread(&pixel, sizeof(unsigned char), 1, file);
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
    printf("Loading MNIST labels...\n");

    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
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
    printf("Freeing network...\n");

    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();

    double* h_flat_W1 = flattenNeuralNetMatrices(net->W1, HIDDEN_SIZE, INPUT_SIZE);
    double* h_flat_W2 = flattenNeuralNetMatrices(net->W2, OUTPUT_SIZE, HIDDEN_SIZE);
    double* d_W1, *d_W2, *d_b1, *d_b2, *d_hidden, *d_output;
    cudaMalloc((void**)&d_W1, sizeof(double) * HIDDEN_SIZE * INPUT_SIZE);
    cudaMalloc((void**)&d_W2, sizeof(double) * OUTPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc((void**)&d_b1, sizeof(double) * HIDDEN_SIZE);
    cudaMalloc((void**)&d_b2, sizeof(double) * OUTPUT_SIZE);
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMemcpy(d_W1, h_flat_W1, sizeof(double) * HIDDEN_SIZE * INPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_flat_W2, sizeof(double) * OUTPUT_SIZE * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, sizeof(double) * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, sizeof(double) * OUTPUT_SIZE, cudaMemcpyHostToDevice);

    trainOnGPU(d_W1, d_W2, d_b1, d_b2, d_hidden, d_output, train_images, train_labels, 60000);
    evaluateOnGPU(d_W1, d_W2, d_b1, d_b2, d_hidden, d_output, test_images, test_labels, 10000);

    cudaFree(train_images);
    cudaFree(train_labels);
    cudaFree(test_images);
    cudaFree(test_labels);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    free(h_flat_W1);
    free(h_flat_W2);
    freeNetwork(net);

    return 0;
}

