#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
//meow
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define NUM_CLASSES 10

// Error checking macro for CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer using CUDA events
class GpuTimer {
    cudaEvent_t start, stop;
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void RecordStart() { CUDA_CHECK(cudaEventRecord(start)); }
    void RecordStop() { CUDA_CHECK(cudaEventRecord(stop)); }
    float ElapsedMillis() {
        float ms;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// Activation functions
__global__ void reluKernel(double* x, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        x[idx] = fmax(0.0, x[idx]);
    }
}

__global__ void softmaxKernel(double* x, int size) {
    __shared__ double max_val;
    __shared__ double sum;
    
    double thread_max = -INFINITY;
    double thread_sum = 0.0;
    
    // Find max value for numerical stability
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        thread_max = fmax(thread_max, x[i]);
    }
    
    if (threadIdx.x == 0) {
        max_val = thread_max;
    }
    __syncthreads();
    
    // Subtract max and compute exp
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = exp(x[i] - max_val);
        thread_sum += x[i];
    }
    
    // Reduce sum
    atomicAdd(&sum, thread_sum);
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] /= sum;
    }
}

// Neural network structure (all on GPU)
typedef struct {
    double* d_W1;  // [HIDDEN_SIZE][INPUT_SIZE]
    double* d_W2;  // [OUTPUT_SIZE][HIDDEN_SIZE]
    double* d_b1;  // [HIDDEN_SIZE]
    double* d_b2;  // [OUTPUT_SIZE]
} GPU_NeuralNetwork;

// Initialize neural network on GPU
GPU_NeuralNetwork* createGPU_Network() {
    GPU_NeuralNetwork* net = (GPU_NeuralNetwork*)malloc(sizeof(GPU_NeuralNetwork));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    
    // Initialize weights and biases on host
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);
    
    return net;
}

// Forward pass on GPU
__global__ void forwardPassKernel(double* W1, double* b1, double* W2, double* b2,
                                 double* input, double* hidden, double* output,
                                 int input_size, int hidden_size, int output_size) {
    
    // Hidden layer computation
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double sum = b1[i];
        for (int j = 0; j < input_size; j++) {
            sum += W1[i * input_size + j] * input[j];
        }
        hidden[i] = sum;
    }
    __syncthreads();
    
    // ReLU activation
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        hidden[i] = fmax(0.0, hidden[i]);
    }
    __syncthreads();
    
    // Output layer computation
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        double sum = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            sum += W2[i * hidden_size + j] * hidden[j];
        }
        output[i] = sum;
    }
}

// Backpropagation on GPU
__global__ void backwardPassKernel(double* W1, double* b1, double* W2, double* b2,
                                  double* input, double* hidden, double* output, 
                                  double* target, int input_size, int hidden_size, int output_size) {
    
    // Shared memory for gradients
    __shared__ double d_output[OUTPUT_SIZE];
    __shared__ double d_hidden[HIDDEN_SIZE];
    
    // Compute output layer gradient
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        d_output[i] = output[i] - target[i];
    }
    __syncthreads();
    
    // Compute hidden layer gradient
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0 ? 1.0 : 0.0);
    }
    __syncthreads();
    
    // Update weights and biases (gradient descent)
    // W2 update
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        for (int j = 0; j < hidden_size; j++) {
            W2[i * hidden_size + j] -= LEARNING_RATE * d_output[i] * hidden[j];
        }
    }
    
    // W1 update
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        for (int j = 0; j < input_size; j++) {
            W1[i * input_size + j] -= LEARNING_RATE * d_hidden[i] * input[j];
        }
    }
    
    // b2 update
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        b2[i] -= LEARNING_RATE * d_output[i];
    }
    
    // b1 update
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        b1[i] -= LEARNING_RATE * d_hidden[i];
    }
}

// Load and prepare MNIST dataset on GPU
void loadMNISTData(const char* image_file, const char* label_file, 
                   double** d_images, double** d_labels, int num_items) {
    // Read images from file
    FILE* img_fp = fopen(image_file, "rb");
    if (!img_fp) {
        printf("Error opening %s\n", image_file);
        exit(1);
    }
    fseek(img_fp, 16, SEEK_SET);
    
    // Read labels from file
    FILE* lbl_fp = fopen(label_file, "rb");
    if (!lbl_fp) {
        printf("Error opening %s\n", label_file);
        exit(1);
    }
    fseek(lbl_fp, 8, SEEK_SET);
    
    // Allocate host memory
    double* h_images = (double*)malloc(num_items * INPUT_SIZE * sizeof(double));
    double* h_labels = (double*)malloc(num_items * OUTPUT_SIZE * sizeof(double));
    
    // Load data to host
    for (int i = 0; i < num_items; i++) {
        // Read image
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, img_fp) != 1) {
                fprintf(stderr, "Error reading image pixel\n");
                exit(EXIT_FAILURE);
            }
            h_images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
        
        // Read label
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, lbl_fp) != 1) {
            fprintf(stderr, "Error reading label\n");
            exit(EXIT_FAILURE);
        }
        
        // Convert label to one-hot encoding
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            h_labels[i * OUTPUT_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    }
    
    fclose(img_fp);
    fclose(lbl_fp);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)d_images, num_items * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)d_labels, num_items * OUTPUT_SIZE * sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(*d_images, h_images, num_items * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_labels, h_labels, num_items * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    free(h_images);
    free(h_labels);
}

// Train network on GPU
void trainNetwork(GPU_NeuralNetwork* net, double* d_train_images, double* d_train_labels, int num_train) {
    GpuTimer total_timer;
    total_timer.RecordStart();
    
    // Allocate device memory for temporary arrays
    double *d_hidden, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    
    dim3 blockDim(256);
    dim3 gridDim(1);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        GpuTimer epoch_timer;
        epoch_timer.RecordStart();
        
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < num_train; i++) {
            double* current_image = d_train_images + i * INPUT_SIZE;
            double* current_label = d_train_labels + i * OUTPUT_SIZE;
            
            // Forward pass
            forwardPassKernel<<<gridDim, blockDim>>>(
                net->d_W1, net->d_b1, net->d_W2, net->d_b2,
                current_image, d_hidden, d_output,
                INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
            );
            CUDA_CHECK(cudaGetLastError());
            
            // Apply softmax
            softmaxKernel<<<gridDim, blockDim>>>(d_output, OUTPUT_SIZE);
            CUDA_CHECK(cudaGetLastError());
            
            // Backward pass
            backwardPassKernel<<<gridDim, blockDim>>>(
                net->d_W1, net->d_b1, net->d_W2, net->d_b2,
                current_image, d_hidden, d_output, current_label,
                INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
            );
            CUDA_CHECK(cudaGetLastError());
            
            // Compute loss and accuracy (requires copying output to host)
            double h_output[OUTPUT_SIZE];
            double h_label[OUTPUT_SIZE];
            
            CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_label, current_label, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                loss -= h_label[k] * log(h_output[k] + 1e-10); // Add small value to avoid log(0)
            }
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (h_label[j] > h_label[actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        epoch_timer.RecordStop();
        float epoch_time = epoch_timer.ElapsedMillis() / 1000.0f;
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / num_train, (correct * 100.0) / num_train, epoch_time);
    }
    
    total_timer.RecordStop();
    float total_time = total_timer.ElapsedMillis() / 1000.0f;
    printf("Total training time: %.3fs\n", total_time);
    
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
}

// Evaluate network on GPU
void evaluateNetwork(GPU_NeuralNetwork* net, double* d_test_images, double* d_test_labels, int num_test) {
    double *d_hidden, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double)));
    
    dim3 blockDim(256);
    dim3 gridDim(1);
    
    int correct = 0;
    
    for (int i = 0; i < num_test; i++) {
        double* current_image = d_test_images + i * INPUT_SIZE;
        double* current_label = d_test_labels + i * OUTPUT_SIZE;
        
        // Forward pass
        forwardPassKernel<<<gridDim, blockDim>>>(
            net->d_W1, net->d_b1, net->d_W2, net->d_b2,
            current_image, d_hidden, d_output,
            INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Apply softmax
        softmaxKernel<<<gridDim, blockDim>>>(d_output, OUTPUT_SIZE);
        CUDA_CHECK(cudaGetLastError());
        
        // Check accuracy
        double h_output[OUTPUT_SIZE];
        double h_label[OUTPUT_SIZE];
        
        CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_label, current_label, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (h_label[j] > h_label[actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct * 100.0) / num_test);
    
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
}

// Free GPU network memory
void freeGPU_Network(GPU_NeuralNetwork* net) {
    CUDA_CHECK(cudaFree(net->d_W1));
    CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1));
    CUDA_CHECK(cudaFree(net->d_b2));
    free(net);
}

int main() {
    printf("MNIST Neural Network on GPU\n\n");
    
    // Load training data
    double *d_train_images, *d_train_labels;
    printf("Loading training data...\n");
    loadMNISTData("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte", 
                  &d_train_images, &d_train_labels, 60000);
    
    // Load test data
    double *d_test_images, *d_test_labels;
    printf("Loading test data...\n");
    loadMNISTData("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte", 
                  &d_test_images, &d_test_labels, 10000);
    
    // Create and train network
    GPU_NeuralNetwork* net = createGPU_Network();
    trainNetwork(net, d_train_images, d_train_labels, 60000);
    evaluateNetwork(net, d_test_images, d_test_labels, 10000);
    
    // Cleanup
    freeGPU_Network(net);
    CUDA_CHECK(cudaFree(d_train_images));
    CUDA_CHECK(cudaFree(d_train_labels));
    CUDA_CHECK(cudaFree(d_test_images));
    CUDA_CHECK(cudaFree(d_test_labels));
    
    return 0;
}