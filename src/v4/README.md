# Version 3: Optimized GPU Implementation

This directory contains an optimized GPU implementation of the MNIST classification algorithm.

## Overview

This implementation uses CUDA to accelerate both the forward and backward passes of a simple fully connected neural network for MNIST digit classification. The network consists of:
- Input layer: 784 units (28x28 pixels)
- Hidden layer: 128 units, ReLU activation
- Output layer: 10 units, softmax activation

**Key optimizations in this version:**
- Uses CUDA pinned (page-locked) memory for faster host-to-device transfers of images and labels.
- Employs CUDA streams and asynchronous memory copies to overlap data transfers and computation.
- All major computations (forward, backward, weight updates) are performed on the GPU using custom CUDA kernels.
- Data is stored in `float` format for improved performance and reduced memory usage.

## Running the Implementation

To run the optimized GPU implementation, use the provided Makefile:

```bash
# Compile the code
make

# Run the program
make run
```

This will compile and execute the optimized GPU version of the algorithm, processing the MNIST dataset that you've downloaded and placed in the `data` directory.

**Expected MNIST files:**
- `data/train-images.idx3-ubyte`
- `data/train-labels.idx1-ubyte`
- `data/t10k-images.idx3-ubyte`
- `data/t10k-labels.idx1-ubyte`

## Expected Output

After running the implementation, you'll see performance metrics including:
- Training loss and accuracy per epoch (for 3 epochs)
- Total training time
- Testing accuracy

Example output:
```
Epoch 1 - Loss: ... - Train Accuracy: ...% - Time: ...s
Epoch 2 - Loss: ... - Train Accuracy: ...% - Time: ...s
Epoch 3 - Loss: ... - Train Accuracy: ...% - Time: ...s
Total training time: ...s
Test Accuracy: ...%
```

## Notes

- Training and evaluation are performed entirely on the GPU (except for loss/accuracy calculation).
- The implementation uses a single training example per step (no batching).
- Host-to-device and device-to-host memory transfers for training data use pinned memory and CUDA streams for better performance.
- The code is intended as a baseline for further optimization and study.
- Performance may vary based on your system specifications and GPU.