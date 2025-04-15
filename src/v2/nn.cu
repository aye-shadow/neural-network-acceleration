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

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double* allocateGPUMemory(int size) {
    double* d_ptr;
    cudaMalloc((void**)&d_ptr, size * sizeof(double));
    return d_ptr;
}
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

// Activation functions
__global__ void reluOnGPU(double* x, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

__global__ void normalizeSoftmaxOnGPU(double* x, double* sum, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] /= *sum;
    }
}
void softmaxOnGPU(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    
    double* d_sum, *d_x;
    cudaMalloc((void**)&d_sum, sizeof(double));
    cudaMalloc((void**)&d_x, size * sizeof(double));
    cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 numBlocks((size + blockSize.x - 1) / blockSize.x);
    normalizeSoftmaxOnGPU<<<numBlocks, blockSize>>>(d_x, d_sum, size);

    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_sum);
}
void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
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
__global__ void forwardPassSnippet1OnGPU(double* input, double* hidden, double* W1, double* b1, int inputSize, int hiddenSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenSize) {
        hidden[idx] = b1[idx];
        for (int j = 0; j < inputSize; j++)
            hidden[idx] += W1[idx * inputSize + j] * input[j];
    }
}
void forwardPassOnGPU(NeuralNetwork* net, double* input, double* hidden, double* output) {

    reluOnGPU(hidden, HIDDEN_SIZE);

    softmaxOnGPU(output, OUTPUT_SIZE);
}
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

// Train network
void trainOnGPU(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();

    cudaEvent_t epoch_start, epoch_end;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forwardPassOnGPU(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
__global__ void normalizeGPUImages(unsigned char* d_pixels, double* d_images, int numImages) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numImages * INPUT_SIZE) {
        d_images[idx] = d_pixels[idx] / 255.0;
    }
}
double* loadMNISTImagesToGPU(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);

    unsigned char* h_pixels = (unsigned char*)malloc(numImages * INPUT_SIZE * sizeof(unsigned char));
    if (!h_pixels) {
        fprintf(stderr, "Error: Memory allocation failed for h_pixels\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(h_pixels, sizeof(unsigned char), numImages * INPUT_SIZE, file) != numImages * INPUT_SIZE) {
        fprintf(stderr, "Error: Failed to read image data\n");
        fclose(file);
        free(h_pixels);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    double* d_images;
    unsigned char* d_pixels;
    cudaMalloc((void**)&d_pixels, numImages * INPUT_SIZE * sizeof(unsigned char));
    cudaMalloc((void**)&d_images, numImages * INPUT_SIZE * sizeof(double));

    cudaMemcpy(d_pixels, h_pixels, numImages * INPUT_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 blockSize(256);
    dim3 numBlocks((numImages * INPUT_SIZE + blockSize.x - 1) / blockSize.x);
    normalizeGPUImages<<<numBlocks, blockSize>>>(d_pixels, d_images, numImages * INPUT_SIZE);

    cudaDeviceSynchronize();

    free(h_pixels);
    cudaFree(d_pixels);

    return d_images;
}
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

__global__ void oneHotEnconding(unsigned char* d_labels, double* d_labels_onehot, int numLabels, int outputSize) {
    // for (int i = 0; i < numLabels; i++) {
    //     unsigned char label;
    //     // fread(&label, sizeof(unsigned char), 1, file);
    //     if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
    //         fprintf(stderr, "Error: Failed to read label\n");
    //         fclose(file);
    //         exit(EXIT_FAILURE);
    //     }

    //     for (int j = 0; j < OUTPUT_SIZE; j++) {
    //         labels[i][j] = (j == label) ? 1.0 : 0.0;
    //     }
    // }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int j = idx % outputSize;
    int i = idx / outputSize;
    if (idx < numLabels * outputSize) {
        d_labels_onehot[idx] = (j == d_labels[i]) ? 1.0 : 0.0;
    }
}
double* loadMNISTLabelsToGPU(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);

    unsigned char* h_labels = (unsigned char*)malloc(numLabels * sizeof(unsigned char));
    if (!h_labels) {
        fprintf(stderr, "Error: Memory allocation failed for h_labels\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(h_labels, sizeof(unsigned char), numLabels, file) != numLabels) {
        fprintf(stderr, "Error: Failed to read label data\n");
        fclose(file);
        free(h_labels);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    unsigned char* d_labels;
    double* d_onehot;
    cudaMalloc((void**)&d_labels, numLabels * sizeof(unsigned char));
    cudaMemcpy(d_labels, h_labels, numLabels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int totalThreadsNeeded = numLabels * OUTPUT_SIZE;
    dim3 numThreads(256);
    dim3 numBlocks((totalThreadsNeeded + numThreads.x - 1) / numThreads.x);
    oneHotEnconding<<<numBlocks, numThreads>>>(d_labels, d_onehot, numLabels, OUTPUT_SIZE);

    cudaDeviceSynchronize();

    free(h_labels);
    cudaFree(d_labels);

    return d_onehot;
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
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}


// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double* train_images = loadMNISTImagesToGPU("../data/train-images.idx3-ubyte", 60000);
    double* train_labels = loadMNISTLabelsToGPU("../data/train-labels.idx1-ubyte", 60000);
    double* test_images = loadMNISTImagesToGPU("../data/t10k-images.idx3-ubyte", 10000);
    double* test_labels = loadMNISTLabelsToGPU("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    trainOnGPU(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    cudaFree(train_images);
    cudaFree(train_labels);
    cudaFree(test_images);
    cudaFree(test_labels);
    return 0;
}

