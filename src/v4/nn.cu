#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>
#include <time.h>

// Network configuration
#define INPUT_SIZE   784
#define HIDDEN_SIZE  128
#define OUTPUT_SIZE  10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define NUM_CLASSES 10
#define BATCH_SIZE 64  // For cuBLAS, batch > 1 is best

#define CHECK_CUDA(call) { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }

#define CHECK_CUBLAS(call) { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error: %s:%d: %d\n", __FILE__, __LINE__, s); exit(1); } }

float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Host utilities for loading data (as float, then convert to __half on upload)
float** alloc_matrix(int rows, int cols) {
    float **matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) matrix[i] = (float*)malloc(cols * sizeof(float));
    return matrix;
}
void free_matrix(float **mat, int rows) {
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}
float** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("Error opening %s\n", filename); exit(1); }
    fseek(file, 16, SEEK_SET);
    float** images = alloc_matrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++)
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, file);
            images[i][j] = pixel / 255.0f;
        }
    fclose(file);
    return images;
}
float** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("Error opening %s\n", filename); exit(1); }
    fseek(file, 8, SEEK_SET);
    float** labels = alloc_matrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, file);
        for (int j = 0; j < OUTPUT_SIZE; j++)
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
    }
    fclose(file);
    return labels;
}

// CUDA kernels for elementwise ops (ReLU, softmax, etc.)
__global__ void relu_forward(__half *x, __half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = __half2float(x[idx]);
        y[idx] = __float2half(val > 0 ? val : 0.0f);
    }
}
__global__ void relu_backward(__half *grad_out, __half *h, __half *grad_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float hv = __half2float(h[idx]);
        grad_in[idx] = hv > 0 ? grad_out[idx] : __float2half(0.0f);
    }
}
__global__ void softmax_forward(__half *x, __half *y, int N) {
    float maxval = -1e30f;
    for (int i = 0; i < N; ++i) {
        float val = __half2float(x[i]);
        if (val > maxval) maxval = val;
    }
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) sum += expf(__half2float(x[i]) - maxval);
    for (int i = 0; i < N; ++i)
        y[i] = __float2half(expf(__half2float(x[i]) - maxval) / sum);
}
// Loss gradient: dL/dz = y_pred - y_true (for softmax + crossentropy)
__global__ void loss_grad(__half *y_pred, __half *y_true, __half *grad, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) grad[i] = __float2half(__half2float(y_pred[i]) - __half2float(y_true[i]));
}

// Host-side structure for weights (host and device pointers)
typedef struct {
    __half *d_W1, *d_b1, *d_W2, *d_b2;
} Network;

// Helper: upload weights (float -> __half)
void upload_weights(__half *d_W, float **W, int rows, int cols) {
    float *tmp = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            tmp[i * cols + j] = W[i][j];
    __half *h_tmp = (__half*)malloc(rows * cols * sizeof(__half));
    for (int i = 0; i < rows * cols; i++)
        h_tmp[i] = __float2half(tmp[i]);
    CHECK_CUDA(cudaMemcpy(d_W, h_tmp, rows * cols * sizeof(__half), cudaMemcpyHostToDevice));
    free(tmp); free(h_tmp);
}
void upload_bias(__half *d_b, float *b, int n) {
    __half *h_tmp = (__half*)malloc(n * sizeof(__half));
    for (int i = 0; i < n; i++) h_tmp[i] = __float2half(b[i]);
    CHECK_CUDA(cudaMemcpy(d_b, h_tmp, n * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_tmp);
}

// Network initialization (host, then upload)
void random_init(float **W, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            W[i][j] = ((float)rand() / RAND_MAX) * 0.01f;
}
void random_init_bias(float *b, int n) {
    for (int i = 0; i < n; i++) b[i] = 0.0f;
}
Network* create_network() {
    Network *net = (Network*)malloc(sizeof(Network));
    CHECK_CUDA(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(__half)));

    float **W1 = alloc_matrix(HIDDEN_SIZE, INPUT_SIZE);
    float **W2 = alloc_matrix(OUTPUT_SIZE, HIDDEN_SIZE);
    float *b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float *b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    srand(time(NULL));
    random_init(W1, HIDDEN_SIZE, INPUT_SIZE);
    random_init(W2, OUTPUT_SIZE, HIDDEN_SIZE);
    random_init_bias(b1, HIDDEN_SIZE);
    random_init_bias(b2, OUTPUT_SIZE);
    upload_weights(net->d_W1, W1, HIDDEN_SIZE, INPUT_SIZE);
    upload_weights(net->d_W2, W2, OUTPUT_SIZE, HIDDEN_SIZE);
    upload_bias(net->d_b1, b1, HIDDEN_SIZE);
    upload_bias(net->d_b2, b2, OUTPUT_SIZE);
    free_matrix(W1, HIDDEN_SIZE);
    free_matrix(W2, OUTPUT_SIZE);
    free(b1); free(b2);
    return net;
}
void free_network(Network *net) {
    CHECK_CUDA(cudaFree(net->d_W1));
    CHECK_CUDA(cudaFree(net->d_b1));
    CHECK_CUDA(cudaFree(net->d_W2));
    CHECK_CUDA(cudaFree(net->d_b2));
    free(net);
}

__global__ void add_bias(__half *mat, const __half *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        mat[idx] = __hadd(mat[idx], bias[col]);
    }
}

__global__ void sum_columns(const __half* mat, __half* result, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += __half2float(mat[row * cols + col]);
        }
        result[col] = __float2half(sum);
    }
}

__global__ void weight_update_kernel(__half *W, const __half *grad, float lr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float w = __half2float(W[idx]);
        float g = __half2float(grad[idx]);
        W[idx] = __float2half(w - lr * g);
    }
}

// Training (batch-wise, cuBLAS, FP16, Tensor Cores)
void train(Network *net, float **train_images, float **train_labels, int num_images) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); // Tensor cores

    // Allocate workspace for a batch
    __half *d_input, *d_h1, *d_h1_relu, *d_logits, *d_pred, *d_label;
    __half *d_grad_logits, *d_grad_h1, *d_grad_W2, *d_grad_b2, *d_grad_W1, *d_grad_b1;
    CHECK_CUDA(cudaMalloc(&d_input,    BATCH_SIZE * INPUT_SIZE  * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_h1,       BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_h1_relu,  BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_logits,   BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_pred,     BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_label,    BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    // Gradients
    CHECK_CUDA(cudaMalloc(&d_grad_logits, BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_grad_h1,    BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_grad_W2,    OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_grad_b2,    OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_grad_W1,    HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_grad_b1,    HIDDEN_SIZE * sizeof(__half)));

    // Host batch buffers
    __half *h_input  = (__half*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(__half));
    __half *h_label  = (__half*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(__half));
    float  *h_pred   = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t start = clock();
        float loss = 0.0f;
        int correct = 0;
        for (int batch = 0; batch < num_images; batch += BATCH_SIZE) {
            int cur_batch = (batch + BATCH_SIZE > num_images) ? (num_images - batch) : BATCH_SIZE;
            // Prepare batch
            for (int i = 0; i < cur_batch; i++)
                for (int j = 0; j < INPUT_SIZE; j++)
                    h_input[i * INPUT_SIZE + j] = __float2half(train_images[batch + i][j]);
            for (int i = 0; i < cur_batch; i++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    h_label[i * OUTPUT_SIZE + j] = __float2half(train_labels[batch + i][j]);
            CHECK_CUDA(cudaMemcpy(d_input, h_input, cur_batch * INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_label, h_label, cur_batch * OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));

            // Forward: H1 = input x W1^T + b1
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                HIDDEN_SIZE, cur_batch, INPUT_SIZE,
                &alpha_h,
                net->d_W1, CUDA_R_16F, HIDDEN_SIZE,
                d_input,   CUDA_R_16F, cur_batch,
                &beta_h,
                d_h1, CUDA_R_16F, HIDDEN_SIZE,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // Add b1 and apply ReLU
            add_bias<<<(cur_batch * HIDDEN_SIZE + 255) / 256, 256>>>(d_h1, net->d_b1, cur_batch, HIDDEN_SIZE);
            CHECK_CUDA(cudaGetLastError());

            int threads = 256;
            relu_forward<<<(cur_batch * HIDDEN_SIZE + threads - 1) / threads, threads>>>(d_h1, d_h1_relu, cur_batch * HIDDEN_SIZE);

            // Output layer: logits = h1_relu x W2^T + b2
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                OUTPUT_SIZE, cur_batch, HIDDEN_SIZE,
                &alpha_h,
                net->d_W2, CUDA_R_16F, OUTPUT_SIZE,
                d_h1_relu, CUDA_R_16F, cur_batch,
                &beta_h,
                d_logits, CUDA_R_16F, OUTPUT_SIZE,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // Add b2, softmax
            add_bias<<<(cur_batch * OUTPUT_SIZE + 255) / 256, 256>>>(d_logits, net->d_b2, cur_batch, OUTPUT_SIZE);
            CHECK_CUDA(cudaGetLastError());

            for (int i = 0; i < cur_batch; i++) {
                softmax_forward<<<1,1>>>(d_logits + i * OUTPUT_SIZE, d_pred + i * OUTPUT_SIZE, OUTPUT_SIZE);
            }

            // Loss + accuracy (host)
            CHECK_CUDA(cudaMemcpy(h_pred, d_pred, cur_batch * OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
            for (int i = 0; i < cur_batch; i++) {
                int pred = 0, actual = 0;
                float maxval = -1e30f;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    float val = __half2float(h_pred[i * OUTPUT_SIZE + j]);
                    if (val > maxval) { maxval = val; pred = j; }
                    if (__half2float(h_label[i * OUTPUT_SIZE + j]) > 0.5f) actual = j;
                    loss -= __half2float(h_label[i * OUTPUT_SIZE + j]) * logf(val > 1e-6f ? val : 1e-6f);
                }
                if (pred == actual) correct++;
            }

            // Backward: dL/dz = y_pred - y_true
            loss_grad<<<(cur_batch * OUTPUT_SIZE + threads - 1) / threads, threads>>>(
                d_pred, d_label, d_grad_logits, cur_batch * OUTPUT_SIZE);

            // dL/dW2 = d_grad_logits^T x h1_relu
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                HIDDEN_SIZE, cur_batch, INPUT_SIZE,
                &alpha_h,
                net->d_W1, CUDA_R_16F, HIDDEN_SIZE,    // lda = HIDDEN_SIZE (W1 row stride)
                d_input,   CUDA_R_16F, INPUT_SIZE,     // ldb = INPUT_SIZE (input row stride)
                &beta_h,
                d_h1, CUDA_R_16F, HIDDEN_SIZE,         // ldc = HIDDEN_SIZE (output row stride)
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // dL/db2 = sum over batch of d_grad_logits
            int block = 128;
            sum_columns<<<(OUTPUT_SIZE + block - 1) / block, block>>>(d_grad_logits, d_grad_b2, cur_batch, OUTPUT_SIZE);

            // dL/dh1_relu = d_grad_logits x W2
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                HIDDEN_SIZE, cur_batch, OUTPUT_SIZE,
                &alpha_h,
                net->d_W2, CUDA_R_16F, HIDDEN_SIZE,
                d_grad_logits, CUDA_R_16F, OUTPUT_SIZE,
                &beta_h,
                d_grad_h1, CUDA_R_16F, HIDDEN_SIZE,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // dL/dh1 (ReLU backward)
            relu_backward<<<(cur_batch * HIDDEN_SIZE + threads - 1) / threads, threads>>>(
                d_grad_h1, d_h1, d_grad_h1, cur_batch * HIDDEN_SIZE);

            // dL/dW1 = d_grad_h1^T x input
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                HIDDEN_SIZE, INPUT_SIZE, cur_batch,
                &alpha_h,
                d_grad_h1, CUDA_R_16F, HIDDEN_SIZE,
                d_input, CUDA_R_16F, INPUT_SIZE,
                &beta_h,
                d_grad_W1, CUDA_R_16F, HIDDEN_SIZE,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // dL/db1 = sum over batch of d_grad_h1
            sum_columns<<<(HIDDEN_SIZE + block - 1) / block, block>>>(d_grad_h1, d_grad_b1, cur_batch, HIDDEN_SIZE);

            // Update: W -= lr * grad
            block = 256;
            int grid_w1 = (HIDDEN_SIZE * INPUT_SIZE + block - 1) / block;
            weight_update_kernel<<<grid_w1, block>>>(net->d_W1, d_grad_W1, LEARNING_RATE / cur_batch, HIDDEN_SIZE * INPUT_SIZE);

            int grid_w2 = (OUTPUT_SIZE * HIDDEN_SIZE + block - 1) / block;
            weight_update_kernel<<<grid_w2, block>>>(net->d_W2, d_grad_W2, LEARNING_RATE / cur_batch, OUTPUT_SIZE * HIDDEN_SIZE);

            int grid_b1 = (HIDDEN_SIZE + block - 1) / block;
            weight_update_kernel<<<grid_b1, block>>>(net->d_b1, d_grad_b1, LEARNING_RATE / cur_batch, HIDDEN_SIZE);

            int grid_b2 = (OUTPUT_SIZE + block - 1) / block;
            weight_update_kernel<<<grid_b2, block>>>(net->d_b2, d_grad_b2, LEARNING_RATE / cur_batch, OUTPUT_SIZE);
        }
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.2fs\n",
            epoch + 1, loss / num_images, 100.0 * correct / num_images, get_time(start));
    }

    // Cleanup
    free(h_input); free(h_label); free(h_pred);
    cudaFree(d_input); cudaFree(d_h1); cudaFree(d_h1_relu); cudaFree(d_logits); cudaFree(d_pred); cudaFree(d_label);
    cudaFree(d_grad_logits); cudaFree(d_grad_h1); cudaFree(d_grad_W2); cudaFree(d_grad_b2); cudaFree(d_grad_W1); cudaFree(d_grad_b1);
    cublasDestroy(handle);
}

__global__ void add_bias_kernel(__half* x, const __half* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = __float2half(__half2float(x[i]) + __half2float(b[i]));
    }
}

// Evaluation
void evaluate(Network *net, float **test_images, float **test_labels, int num_images) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    __half *d_input, *d_h1, *d_h1_relu, *d_logits, *d_pred;
    CHECK_CUDA(cudaMalloc(&d_input,    INPUT_SIZE  * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_h1,       HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_h1_relu,  HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_logits,   OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_pred,     OUTPUT_SIZE * sizeof(__half)));

    float *h_pred = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    int correct = 0;
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    for (int i = 0; i < num_images; i++) {
        __half h_input[INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++) h_input[j] = __float2half(test_images[i][j]);
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_SIZE, 1, INPUT_SIZE,
                                  &alpha_h, net->d_W1, CUDA_R_16F, HIDDEN_SIZE,
                                  d_input, CUDA_R_16F, INPUT_SIZE, &beta_h, d_h1, CUDA_R_16F, HIDDEN_SIZE,
                                  CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        int block = 256;
        add_bias_kernel<<<(HIDDEN_SIZE + block - 1)/block, block>>>(d_h1, net->d_b1, HIDDEN_SIZE);
        relu_forward<<<1, HIDDEN_SIZE>>>(d_h1, d_h1_relu, HIDDEN_SIZE);

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, OUTPUT_SIZE, 1, HIDDEN_SIZE,
                                  &alpha_h, net->d_W2, CUDA_R_16F, OUTPUT_SIZE,
                                  d_h1_relu, CUDA_R_16F, HIDDEN_SIZE, &beta_h, d_logits, CUDA_R_16F, OUTPUT_SIZE,
                                  CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        add_bias_kernel<<<(OUTPUT_SIZE + block - 1)/block, block>>>(d_logits, net->d_b2, OUTPUT_SIZE);
        softmax_forward<<<1,1>>>(d_logits, d_pred, OUTPUT_SIZE);

        CHECK_CUDA(cudaMemcpy(h_pred, d_pred, OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
        int pred = 0, actual = 0; float maxval = -1e30f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float val = __half2float(h_pred[j]);
            if (val > maxval) { maxval = val; pred = j; }
            if (test_labels[i][j] > 0.5f) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", 100.0 * correct / num_images);
    free(h_pred);
    cudaFree(d_input); cudaFree(d_h1); cudaFree(d_h1_relu); cudaFree(d_logits); cudaFree(d_pred);
    cublasDestroy(handle);
}

int main() {
    printf("MNIST Neural Net FP16 Tensor Core Version\n");
    float **train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    float **train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    float **test_images  = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    float **test_labels  = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    Network *net = create_network();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    free_network(net);
    free_matrix(train_images, 60000); free_matrix(train_labels, 60000);
    free_matrix(test_images, 10000);  free_matrix(test_labels, 10000);
    return 0;
}