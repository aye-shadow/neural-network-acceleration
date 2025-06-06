# Version 1: Sequential Implementation

This directory contains the sequential implementation of the MNIST classification algorithm.

## Overview

The sequential implementation serves as the baseline for performance comparison with the optimized parallel versions in subsequent folders.

## Running the Implementation

To run the sequential implementation, simply use the provided makefile:

```bash
# Compile the code
make

# Run the program
make run
```

This will compile and execute the sequential version of the algorithm, processing the MNIST dataset that you've downloaded and placed in the data directory.

## Expected Output

After running the implementation, you'll see performance metrics including:
- Training time per epoch
- Total training time
- Testing accuracy
- A profiling report (profile_results.txt) generated by gprof

## Notes

- This implementation uses a single-threaded approach
- Performance may vary based on your system specifications
- Use this as a baseline to compare with other optimized versions
- The gprof profiling output helps identify performance bottlenecks for optimization