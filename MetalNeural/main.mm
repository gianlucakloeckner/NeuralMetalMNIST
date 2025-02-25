// main.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "NeuralNetwork.h"

#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>

// ----------------------------------------------------------------------
// Helper function to load MNIST images (IDX format)
float* loadMNISTImages(const char* filename, int* out_numImages, int* out_rows, int* out_cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening image file: " << filename << std::endl;
        return nullptr;
    }
    
    int32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = ntohl(magic);
    if (magic != 2051) {
        std::cerr << "Invalid MNIST image file magic number: " << magic << std::endl;
        return nullptr;
    }
    
    int32_t numImages = 0, numRows = 0, numCols = 0;
    file.read(reinterpret_cast<char*>(&numImages), 4);
    numImages = ntohl(numImages);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    numRows = ntohl(numRows);
    file.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = ntohl(numCols);
    
    *out_numImages = numImages;
    *out_rows = numRows;
    *out_cols = numCols;
    
    int imageSize = numRows * numCols;
    // Allocate a float array to hold all images (normalize pixel values to [0,1]).
    float* images = new float[numImages * imageSize];
    
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i * imageSize + j] = pixel / 255.0f;
        }
    }
    
    file.close();
    return images;
}

// ----------------------------------------------------------------------
// Helper function to load MNIST labels (IDX format)
// Converts each label into a one-hot encoded vector (length 10).
float* loadMNISTLabels(const char* filename, int* out_numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << filename << std::endl;
        return nullptr;
    }
    
    int32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = ntohl(magic);
    if (magic != 2049) {
        std::cerr << "Invalid MNIST label file magic number: " << magic << std::endl;
        return nullptr;
    }
    
    int32_t numLabels = 0;
    file.read(reinterpret_cast<char*>(&numLabels), 4);
    numLabels = ntohl(numLabels);
    *out_numLabels = numLabels;
    
    // Allocate one-hot encoded labels: numLabels x 10.
    float* labels = new float[numLabels * 10];
    memset(labels, 0, numLabels * 10 * sizeof(float));
    
    for (int i = 0; i < numLabels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (label < 10) {
            labels[i * 10 + label] = 1.0f;
        }
    }
    
    file.close();
    return labels;
}

// ----------------------------------------------------------------------
// Evaluate predictions: compute accuracy and cross–entropy loss.
void evaluateBatch(const float* predictions, const float* groundTruth, int batchSize, float &sumLoss, int &correctCount) {
    const float epsilon = 1e-8f;
    for (int i = 0; i < batchSize; i++) {
        const float* predRow = predictions + i * 10;
        const float* truthRow = groundTruth + i * 10;
        int predictedClass = 0;
        float maxProb = predRow[0];
        for (int j = 1; j < 10; j++) {
            if (predRow[j] > maxProb) {
                maxProb = predRow[j];
                predictedClass = j;
            }
        }
        
        int trueClass = 0;
        for (int j = 0; j < 10; j++) {
            if (truthRow[j] == 1.0f) {
                trueClass = j;
                break;
            }
        }
        
        if (predictedClass == trueClass)
            correctCount++;
        
        // Cross-entropy loss: -log(probability of true class)
        float prob = predRow[trueClass];
        sumLoss += -logf(prob + epsilon);
    }
}

// ----------------------------------------------------------------------
// Main function: load data, train with backpropagation, and validate.
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Create Metal device.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            return -1;
        }
        
        // Create neural network instance.
        NeuralNetwork *nn = [[NeuralNetwork alloc] initWithDevice:device];
        if (!nn) {
            std::cerr << "Failed to create NeuralNetwork instance." << std::endl;
            return -1;
        }
        
        // -------------------------------
        // Load MNIST training dataset.
        int numTrainImages = 0, trainRows = 0, trainCols = 0;
        const char *trainImagesPath = "/Users/giannes/Programmierung/MetalNeural/train-images.idx3-ubyte";
        float *trainImages = loadMNISTImages(trainImagesPath, &numTrainImages, &trainRows, &trainCols);
        if (!trainImages) {
            std::cerr << "Failed to load MNIST training images." << std::endl;
            return -1;
        }
        
        int numTrainLabels = 0;
        const char *trainLabelsPath = "/Users/giannes/Programmierung/MetalNeural/train-labels.idx1-ubyte";
        float *trainLabels = loadMNISTLabels(trainLabelsPath, &numTrainLabels);
        if (!trainLabels) {
            std::cerr << "Failed to load MNIST training labels." << std::endl;
            delete[] trainImages;
            return -1;
        }
        
        if (numTrainImages != numTrainLabels) {
            std::cerr << "Mismatch between number of training images and labels." << std::endl;
            delete[] trainImages;
            delete[] trainLabels;
            return -1;
        }
        
        // -------------------------------
        // Load MNIST test dataset.
        int numTestImages = 0, testRows = 0, testCols = 0;
        const char *testImagesPath = "/Users/giannes/Programmierung/MetalNeural/t10k-images.idx3-ubyte";
        float *testImages = loadMNISTImages(testImagesPath, &numTestImages, &testRows, &testCols);
        if (!testImages) {
            std::cerr << "Failed to load MNIST test images." << std::endl;
            delete[] trainImages;
            delete[] trainLabels;
            return -1;
        }
        
        int numTestLabels = 0;
        const char *testLabelsPath = "/Users/giannes/Programmierung/MetalNeural/t10k-labels.idx1-ubyte";
        float *testLabels = loadMNISTLabels(testLabelsPath, &numTestLabels);
        if (!testLabels) {
            std::cerr << "Failed to load MNIST test labels." << std::endl;
            delete[] trainImages;
            delete[] trainLabels;
            delete[] testImages;
            return -1;
        }
        
        if (numTestImages != numTestLabels) {
            std::cerr << "Mismatch between number of test images and labels." << std::endl;
            delete[] trainImages;
            delete[] trainLabels;
            delete[] testImages;
            delete[] testLabels;
            return -1;
        }
        
        // -------------------------------
        // Training parameters.
        const int trainBatchSize = 100;
        int numTrainBatches = numTrainImages / trainBatchSize;
        const float learningRate = 0.01f; // Set learning rate for backpropagation
        
        std::cout << "Starting training on " << numTrainImages
                  << " images in " << numTrainBatches << " batches." << std::endl;
        
        // Measure training time.
        auto startTrain = std::chrono::high_resolution_clock::now();
        
        // Training loop (one epoch).
        for (int batch = 0; batch < numTrainBatches; batch++) {
            float* batchImages = trainImages + batch * trainBatchSize * trainRows * trainCols;
            float* batchLabels = trainLabels + batch * trainBatchSize * 10;
            
            [nn trainOnMNISTBatch:batchImages labels:batchLabels batchSize:trainBatchSize learningRate:learningRate];
            
            if ((batch + 1) % 10 == 0) {
                std::cout << "Trained batch " << (batch + 1) << " / " << numTrainBatches << std::endl;
            }
        }
        
        auto endTrain = std::chrono::high_resolution_clock::now();
        double trainingTime = std::chrono::duration<double>(endTrain - startTrain).count();
        std::cout << "Training completed in " << trainingTime << " seconds." << std::endl;
        
        // -------------------------------
        // Evaluate on test set.
        const int testBatchSize = 100;
        int numTestBatches = numTestImages / testBatchSize;
        int totalCorrect = 0;
        float totalLoss = 0.0f;
        int totalSamples = numTestBatches * testBatchSize;
        
        std::cout << "Evaluating on " << totalSamples << " test images." << std::endl;
        
        auto startEval = std::chrono::high_resolution_clock::now();
        
        for (int batch = 0; batch < numTestBatches; batch++) {
            float* batchImages = testImages + batch * testBatchSize * testRows * testCols;
            float* batchLabels = testLabels + batch * testBatchSize * 10;
            
            float* predictions = [nn predict:batchImages batchSize:testBatchSize];
            evaluateBatch(predictions, batchLabels, testBatchSize, totalLoss, totalCorrect);
        }
        
        auto endEval = std::chrono::high_resolution_clock::now();
        double evaluationTime = std::chrono::duration<double>(endEval - startEval).count();
        
        float averageLoss = totalLoss / totalSamples;
        float accuracy = (float)totalCorrect / totalSamples * 100.0f;
        
        std::cout << "Evaluation completed in " << evaluationTime << " seconds." << std::endl;
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Average Cross-Entropy Loss: " << averageLoss << std::endl;
        
        // Optionally, run a prediction on the first test image.
        float* singlePrediction = [nn predict:testImages batchSize:1];
        std::cout << "Prediction for the first test image:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << singlePrediction[i] << " ";
        }
        std::cout << std::endl;
        
        // Clean up allocated memory.
        delete[] trainImages;
        delete[] trainLabels;
        delete[] testImages;
        delete[] testLabels;
    }
    return 0;
}
