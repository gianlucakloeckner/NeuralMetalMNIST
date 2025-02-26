# MetalNeural

A high-performance neural network implementation for MNIST digit classification using Apple's Metal and Metal Performance Shaders (MPS) frameworks. This project demonstrates how to leverage GPU acceleration for machine learning tasks on Apple devices.

## Overview

MetalNeural implements a simple feedforward neural network with the following architecture:
- Input layer: 784 neurons (28x28 pixel MNIST images)
- Hidden layer: 128 neurons with batch normalization
- Output layer: 10 neurons (one for each digit)

The implementation includes:
- FP16 (half-precision) computations for improved performance
- Batch normalization for faster training convergence
- Adam optimizer for efficient weight updates
- Parallel command buffers for better GPU utilization

## Features

- **GPU-Accelerated Training**: Uses Metal Performance Shaders for matrix multiplications
- **Mixed Precision**: FP16/FP32 mixed precision for optimal performance
- **Modern Training Techniques**: 
  - Batch normalization
  - Adam optimization
  - Command buffer pipelining
- **MNIST Dataset Support**: Built-in loaders for the MNIST dataset

## Requirements

- macOS device with Metal support
- Xcode 14.0+ 
- C++17 compatible compiler
- MNIST dataset files:
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

## Project Structure

- **NeuralNetwork.h/mm**: Core neural network implementation
- **NeuralNetworkKernel.metal**: Metal shaders for batch normalization and Adam optimizer
- **main.mm**: Main application logic, data loading, training, and evaluation

## Getting Started

1. Clone the repository
2. Download the MNIST dataset files
3. Update the file paths in `main.mm` to point to your MNIST dataset files
4. Build and run the project in Xcode

```bash
git clone https://github.com/yourusername/MetalNeural.git
cd MetalNeural
# Download MNIST dataset (not included in repo)
# Update file paths in main.mm
open MetalNeural.xcodeproj
```

## Usage

The application will:
1. Load the MNIST training and test datasets
2. Train the neural network on the training set (60,000 images)
3. Evaluate the model on the test set (10,000 images)
4. Print out accuracy, loss, and a sample prediction

The default configuration:
- Batch size: 100
- Learning rate: 0.01
- Hidden layer size: 128 neurons

## Performance

The implementation is optimized for performance using several techniques:
- FP16 (half-precision) for weights and intermediate activations
- Batch normalization to improve training stability and speed
- Adam optimizer for adaptive learning rates
- Parallel command buffer execution for GPU pipelining

## Advanced Usage

To customize the neural network:
- Modify the network architecture in `NeuralNetwork.mm`
- Adjust hyperparameters in `main.mm`
- Extend the Metal kernels in `NeuralNetworkKernel.metal`

## Implementation Details

### Neural Network Architecture

```
Input (784) -> Hidden (128) -> BatchNorm -> Output (10)
```

### Key Components

1. **Matrix Multiplications**: Using MPS for efficient GPU-accelerated matrix operations
2. **Batch Normalization**: Custom Metal kernel implementation
3. **Adam Optimizer**: Efficient weight updates with momentum and adaptive learning rates
4. **Command Buffer Pipelining**: Parallel execution of GPU commands

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Apple's Metal Documentation](https://developer.apple.com/documentation/metal)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
