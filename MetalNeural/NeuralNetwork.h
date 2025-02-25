//
//  NeuralNetwork.h
//  MetalNeural
//
//  Created by Klöckner Gian-Luca on 25.02.25.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface NeuralNetwork : NSObject

// Initialize the network with a Metal device.
- (instancetype)initWithDevice:(id<MTLDevice>)device;

// Train on a batch of MNIST images (flattened to 784 floats)
// with one–hot encoded labels (10 floats per sample), using Adam.
- (void)trainOnMNISTBatch:(float *)inputData
                   labels:(float *)labels
                batchSize:(NSUInteger)batchSize
             learningRate:(float)learningRate;

// Perform a prediction (forward pass) on a batch of MNIST images.
// Returns a pointer to the output (in FP32 converted from FP16).
- (float *)predict:(float *)inputData batchSize:(NSUInteger)batchSize;

@end

NS_ASSUME_NONNULL_END
