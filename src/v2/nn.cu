#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
//meow4
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define NUM_CLASSES 10
#define BLOCK_SIZE 256
#define SMALL_CONST 1e-10f

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Optimized neural network structure
typedef struct {
    double *d_W1, *d_W2, *d_b1, *d_b2;
    double *d_hidden, *d_output;
} GPU_NeuralNetwork;

// Fast ReLU activation
__global__ void reluKernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) x[idx] = fmax(0.0, x[idx]);
}

// Optimized softmax with warp-level reduction
__global__ void softmaxKernel(double* x, int size) {
    __shared__ double buffer[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Find max
    double max_val = -INFINITY;
    for(int i = idx; i < size; i += gridDim.x * blockDim.x) {
        max_val = fmax(max_val, x[i]);
    }
    buffer[tid] = max_val;
    __syncthreads();
    
    // Reduce max
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) buffer[tid] = fmax(buffer[tid], buffer[tid+s]);
        __syncthreads();
    }
    max_val = buffer[0];
    __syncthreads();
    
    // Compute exp and sum
    double sum = 0.0;
    for(int i = idx; i < size; i += gridDim.x * blockDim.x) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    buffer[tid] = sum;
    __syncthreads();
    
    // Reduce sum
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) buffer[tid] += buffer[tid+s];
        __syncthreads();
    }
    sum = buffer[0] + SMALL_CONST;
    
    // Normalize
    for(int i = idx; i < size; i += gridDim.x * blockDim.x) {
        x[i] /= sum;
    }
}

// Initialize network with Xavier initialization
GPU_NeuralNetwork* createGPU_Network() {
    GPU_NeuralNetwork* net = (GPU_NeuralNetwork*)malloc(sizeof(GPU_NeuralNetwork));
    
    // Allocate all device memory at once
    size_t total_size = (HIDDEN_SIZE*INPUT_SIZE + OUTPUT_SIZE*HIDDEN_SIZE + 
                        HIDDEN_SIZE + OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(double);
    double* d_memory;
    CUDA_CHECK(cudaMalloc(&d_memory, total_size));
    
    // Assign pointers
    net->d_W1 = d_memory;
    net->d_W2 = net->d_W1 + HIDDEN_SIZE*INPUT_SIZE;
    net->d_b1 = net->d_W2 + OUTPUT_SIZE*HIDDEN_SIZE;
    net->d_b2 = net->d_b1 + HIDDEN_SIZE;
    net->d_hidden = net->d_b2 + OUTPUT_SIZE;
    net->d_output = net->d_hidden + HIDDEN_SIZE;
    
    // Initialize weights on host
    double* h_W1 = (double*)malloc(HIDDEN_SIZE*INPUT_SIZE*sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE*HIDDEN_SIZE*sizeof(double));
    
    // Xavier initialization
    double scale1 = sqrt(2.0/INPUT_SIZE);
    double scale2 = sqrt(2.0/HIDDEN_SIZE);
    for(int i=0; i<HIDDEN_SIZE*INPUT_SIZE; i++) 
        h_W1[i] = (rand()/(double)RAND_MAX - 0.5) * scale1;
    for(int i=0; i<OUTPUT_SIZE*HIDDEN_SIZE; i++) 
        h_W2[i] = (rand()/(double)RAND_MAX - 0.5) * scale2;
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE*INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(net->d_b1, 0, HIDDEN_SIZE*sizeof(double)));
    CUDA_CHECK(cudaMemset(net->d_b2, 0, OUTPUT_SIZE*sizeof(double)));
    
    free(h_W1);
    free(h_W2);
    return net;
}

// Fused forward pass kernel
__global__ void forwardKernel(double* W1, double* b1, double* W2, double* b2,
                             double* input, double* hidden, double* output,
                             int input_size, int hidden_size, int output_size) {
    // Hidden layer
    for(int i=threadIdx.x; i<hidden_size; i+=blockDim.x) {
        double sum = b1[i];
        for(int j=0; j<input_size; j++) {
            sum += W1[i*input_size + j] * input[j];
        }
        hidden[i] = fmax(0.0, sum); // Fused ReLU
    }
    __syncthreads();
    
    // Output layer
    for(int i=threadIdx.x; i<output_size; i+=blockDim.x) {
        double sum = b2[i];
        for(int j=0; j<hidden_size; j++) {
            sum += W2[i*hidden_size + j] * hidden[j];
        }
        output[i] = sum;
    }
}

// Fused backward pass kernel
__global__ void backwardKernel(double* W1, double* b1, double* W2, double* b2,
                              double* input, double* hidden, double* output,
                              double* target, int input_size, int hidden_size, int output_size) {
    __shared__ double d_output[OUTPUT_SIZE];
    __shared__ double d_hidden[HIDDEN_SIZE];
    
    // Output gradient
    for(int i=threadIdx.x; i<output_size; i+=blockDim.x) {
        d_output[i] = output[i] - target[i];
    }
    __syncthreads();
    
    // Hidden gradient
    for(int i=threadIdx.x; i<hidden_size; i+=blockDim.x) {
        double sum = 0.0;
        for(int j=0; j<output_size; j++) {
            sum += W2[j*hidden_size + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0 ? 1.0 : 0.0);
    }
    __syncthreads();
    
    // Update weights
    for(int i=threadIdx.x; i<output_size; i+=blockDim.x) {
        for(int j=0; j<hidden_size; j++) {
            W2[i*hidden_size + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
        b2[i] -= LEARNING_RATE * d_output[i];
    }
    
    for(int i=threadIdx.x; i<hidden_size; i+=blockDim.x) {
        for(int j=0; j<input_size; j++) {
            W1[i*input_size + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
        b1[i] -= LEARNING_RATE * d_hidden[i];
    }
}

// Fast data loading
void loadMNISTData(const char* image_file, const char* label_file, 
                   double** d_images, double** d_labels, int num_items) {
    FILE *img_fp = fopen(image_file, "rb");
    FILE *lbl_fp = fopen(label_file, "rb");
    if(!img_fp || !lbl_fp) {
        fprintf(stderr, "Error opening files\n");
        exit(EXIT_FAILURE);
    }
    
    fseek(img_fp, 16, SEEK_SET);
    fseek(lbl_fp, 8, SEEK_SET);
    
    // Allocate pinned memory for faster transfer
    double *h_images, *h_labels;
    CUDA_CHECK(cudaMallocHost(&h_images, num_items*INPUT_SIZE*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_labels, num_items*OUTPUT_SIZE*sizeof(double)));
    
    // Batch read
    unsigned char *buffer = (unsigned char*)malloc(num_items*INPUT_SIZE);
    fread(buffer, sizeof(unsigned char), num_items*INPUT_SIZE, img_fp);
    for(int i=0; i<num_items*INPUT_SIZE; i++) 
        h_images[i] = buffer[i]/255.0;
    
    fread(buffer, sizeof(unsigned char), num_items, lbl_fp);
    for(int i=0; i<num_items; i++) {
        for(int j=0; j<OUTPUT_SIZE; j++) {
            h_labels[i*OUTPUT_SIZE + j] = (j == buffer[i]) ? 1.0 : 0.0;
        }
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(d_images, num_items*INPUT_SIZE*sizeof(double)));
    CUDA_CHECK(cudaMalloc(d_labels, num_items*OUTPUT_SIZE*sizeof(double)));
    
    // Async copy
    CUDA_CHECK(cudaMemcpyAsync(*d_images, h_images, num_items*INPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(*d_labels, h_labels, num_items*OUTPUT_SIZE*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    free(buffer);
    CUDA_CHECK(cudaFreeHost(h_images));
    CUDA_CHECK(cudaFreeHost(h_labels));
    fclose(img_fp);
    fclose(lbl_fp);
}

// Optimized training
void train(GPU_NeuralNetwork* net, double* d_images, double* d_labels, int num_train) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(1);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    for(int epoch=0; epoch<EPOCHS; epoch++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        int correct = 0;
        double loss = 0.0;
        
        for(int i=0; i<num_train; i++) {
            double* img = d_images + i*INPUT_SIZE;
            double* lbl = d_labels + i*OUTPUT_SIZE;
            
            // Forward + softmax
            forwardKernel<<<grid, block>>>(net->d_W1, net->d_b1, net->d_W2, net->d_b2,
                                         img, net->d_hidden, net->d_output,
                                         INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            softmaxKernel<<<grid, block>>>(net->d_output, OUTPUT_SIZE);
            
            // Backward
            backwardKernel<<<grid, block>>>(net->d_W1, net->d_b1, net->d_W2, net->d_b2,
                                           img, net->d_hidden, net->d_output, lbl,
                                           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            
            // Check accuracy (async)
            if(i % 1000 == 0) {
                double h_output[OUTPUT_SIZE], h_label[OUTPUT_SIZE];
                CUDA_CHECK(cudaMemcpy(h_output, net->d_output, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_label, lbl, OUTPUT_SIZE*sizeof(double), cudaMemcpyDeviceToHost));
                
                int pred = 0, actual = 0;
                for(int j=0; j<OUTPUT_SIZE; j++) {
                    if(h_output[j] > h_output[pred]) pred = j;
                    if(h_label[j] > h_label[actual]) actual = j;
                }
                if(pred == actual) correct++;
                
                for(int k=0; k<OUTPUT_SIZE; k++) {
                    if(h_label[k] > 0.5) {
                        loss -= log(h_output[k] + SMALL_CONST);
                    }
                }
            }
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
        
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n",
               epoch+1, loss/num_train, (correct*100.0)/(num_train/1000), time/1000.0f);
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Main function
int main() {
    printf("MNIST Neural Network - Optimized GPU Version\n");
    
    // Load data
    double *d_train_images, *d_train_labels;
    double *d_test_images, *d_test_labels;
    
    printf("Loading data...\n");
    loadMNISTData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", 
                 &d_train_images, &d_train_labels, 60000);
    loadMNISTData("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte", 
                 &d_test_images, &d_test_labels, 10000);
    
    // Create and train network
    GPU_NeuralNetwork* net = createGPU_Network();
    printf("Training...\n");
    train(net, d_train_images, d_train_labels, 60000);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_train_images));
    CUDA_CHECK(cudaFree(d_train_labels));
    CUDA_CHECK(cudaFree(d_test_images));
    CUDA_CHECK(cudaFree(d_test_labels));
    CUDA_CHECK(cudaFree(net->d_W1)); // Frees all memory (single allocation)
    free(net);
    
    return 0;
}