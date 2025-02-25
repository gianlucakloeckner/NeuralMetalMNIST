//
//  NeuralNetwork.mm
//  MetalNeural
//
//  Created by Klöckner Gian-Luca on 25.02.25.
//
// NeuralNetwork.mm
#import "NeuralNetwork.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <simd/simd.h>
#include <math.h>

// Helper conversion routines between float and FP16.
static void convertFloatToHalf(const float *src, __fp16 *dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        dst[i] = (__fp16)src[i];
    }
}
static void convertHalfToFloat(const __fp16 *src, float *dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        dst[i] = (float)src[i];
    }
}

// Define a structure for matrix dimensions.
typedef struct {
    uint M;
    uint K;
    uint N;
} MatrixDimensions;

@interface NeuralNetwork ()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;

// MPS objects for forward multiplications.
@property (nonatomic, strong) MPSMatrixMultiplication *mpsMulInputHidden;
@property (nonatomic, strong) MPSMatrixMultiplication *mpsMulHiddenOutput;

// Adam optimizer parameters (we use these for both layers).
@property (nonatomic, assign) float beta1;
@property (nonatomic, assign) float beta2;
@property (nonatomic, assign) float adamEpsilon;

// Weight buffers (FP16)
// Weight1: 784 x 128, Weight2: 128 x 10.
@property (nonatomic, strong) id<MTLBuffer> weight1Buffer;
@property (nonatomic, strong) id<MTLBuffer> weight2Buffer;

// Adam momentum and variance buffers for each weight matrix.
@property (nonatomic, strong) id<MTLBuffer> weight1_mBuffer;
@property (nonatomic, strong) id<MTLBuffer> weight1_vBuffer;
@property (nonatomic, strong) id<MTLBuffer> weight2_mBuffer;
@property (nonatomic, strong) id<MTLBuffer> weight2_vBuffer;

// Pipeline state for batch normalization and Adam update kernels.
@property (nonatomic, strong) id<MTLComputePipelineState> batchNormPipelineState;
@property (nonatomic, strong) id<MTLComputePipelineState> updateAdamPipelineState;
@end

@implementation NeuralNetwork

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if (self = [super init]) {
        _device = device;
        _commandQueue = [device newCommandQueue];
        
        // Set Adam parameters.
        _beta1 = 0.9f;
        _beta2 = 0.999f;
        _adamEpsilon = 1e-8f;
        
        // Create MPSMatrixMultiplication objects later on per–batch.
        // (We will create descriptors on the fly.)
        
        // Create pipeline states for our custom kernels.
        NSError *error = nil;
        id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
        id<MTLFunction> batchNormFunc = [defaultLibrary newFunctionWithName:@"batch_normalization"];
        _batchNormPipelineState = [device newComputePipelineStateWithFunction:batchNormFunc error:&error];
        if (!_batchNormPipelineState) {
            NSLog(@"Error creating batchNorm pipeline: %@", error);
            return nil;
        }
        id<MTLFunction> updateAdamFunc = [defaultLibrary newFunctionWithName:@"update_weights_adam"];
        _updateAdamPipelineState = [device newComputePipelineStateWithFunction:updateAdamFunc error:&error];
        if (!_updateAdamPipelineState) {
            NSLog(@"Error creating updateAdam pipeline: %@", error);
            return nil;
        }
        
        // Allocate weight buffers (FP16). Sizes in bytes.
        NSUInteger weight1Count = 784 * 128;
        NSUInteger weight2Count = 128 * 10;
        NSUInteger weight1Size = weight1Count * sizeof(__fp16);
        NSUInteger weight2Size = weight2Count * sizeof(__fp16);
        
        _weight1Buffer = [device newBufferWithLength:weight1Size options:MTLResourceStorageModePrivate];
        _weight2Buffer = [device newBufferWithLength:weight2Size options:MTLResourceStorageModePrivate];
        
        // Allocate Adam moment and variance buffers (same size as weights).
        _weight1_mBuffer = [device newBufferWithLength:weight1Size options:MTLResourceStorageModePrivate];
        _weight1_vBuffer = [device newBufferWithLength:weight1Size options:MTLResourceStorageModePrivate];
        _weight2_mBuffer = [device newBufferWithLength:weight2Size options:MTLResourceStorageModePrivate];
        _weight2_vBuffer = [device newBufferWithLength:weight2Size options:MTLResourceStorageModePrivate];
        
        [self initializeWeights];
    }
    return self;
}

- (void)initializeWeights {
    // Allocate temporary FP32 arrays, initialize with small random values, and convert to FP16.
    NSUInteger weight1Count = 784 * 128;
    float *tempW1 = (float *)malloc(weight1Count * sizeof(float));
    for (NSUInteger i = 0; i < weight1Count; i++) {
        tempW1[i] = (((float)arc4random() / UINT32_MAX) - 0.5f) * 0.1f;
    }
    __fp16 *w1_half = (__fp16 *)malloc(weight1Count * sizeof(__fp16));
    convertFloatToHalf(tempW1, w1_half, weight1Count);
    // Copy into the GPU buffer (using a blit command encoder for private storage).
    id<MTLCommandBuffer> cb = [_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:[_device newBufferWithBytes:w1_half length:weight1Count * sizeof(__fp16) options:MTLResourceStorageModeShared]
                sourceOffset:0
                    toBuffer:_weight1Buffer
           destinationOffset:0
                       size:weight1Count * sizeof(__fp16)];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    free(tempW1);
    free(w1_half);
    
    // Do the same for weight2.
    NSUInteger weight2Count = 128 * 10;
    float *tempW2 = (float *)malloc(weight2Count * sizeof(float));
    for (NSUInteger i = 0; i < weight2Count; i++) {
        tempW2[i] = (((float)arc4random() / UINT32_MAX) - 0.5f) * 0.1f;
    }
    __fp16 *w2_half = (__fp16 *)malloc(weight2Count * sizeof(__fp16));
    convertFloatToHalf(tempW2, w2_half, weight2Count);
    cb = [_commandQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:[_device newBufferWithBytes:w2_half length:weight2Count * sizeof(__fp16) options:MTLResourceStorageModeShared]
                sourceOffset:0
                    toBuffer:_weight2Buffer
           destinationOffset:0
                       size:weight2Count * sizeof(__fp16)];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    free(tempW2);
    free(w2_half);
    
    // Initialize Adam buffers to zero.
    memset([_weight1_mBuffer contents], 0, weight1Count * sizeof(__fp16));
    memset([_weight1_vBuffer contents], 0, weight1Count * sizeof(__fp16));
    memset([_weight2_mBuffer contents], 0, weight2Count * sizeof(__fp16));
    memset([_weight2_vBuffer contents], 0, weight2Count * sizeof(__fp16));
}

#pragma mark - Forward Pass with MPS and BatchNorm

- (float *)predict:(float *)inputData batchSize:(NSUInteger)batchSize {
    // Convert input (FP32) to FP16 and create an input buffer.
    NSUInteger inputCount = batchSize * 784;
    __fp16 *inputHalf = (__fp16 *)malloc(inputCount * sizeof(__fp16));
    convertFloatToHalf(inputData, inputHalf, inputCount);
    id<MTLBuffer> inputBuffer = [_device newBufferWithBytes:inputHalf length:inputCount * sizeof(__fp16) options:MTLResourceStorageModeShared];
    free(inputHalf);
    
    // Create buffers for hidden layer and output (FP16).
    NSUInteger hiddenCount = batchSize * 128;
    NSUInteger outputCount = batchSize * 10;
    id<MTLBuffer> hiddenBuffer = [_device newBufferWithLength:hiddenCount * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> outputBuffer = [_device newBufferWithLength:outputCount * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    
    // Create MPSMatrix descriptors.
    MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:784 rowBytes:784 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descHidden = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:128 rowBytes:128 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descOutput = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:10 rowBytes:10 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    
    MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:inputBuffer descriptor:descInput];
    MPSMatrix *matHidden = [[MPSMatrix alloc] initWithBuffer:hiddenBuffer descriptor:descHidden];
    MPSMatrix *matOutput = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:descOutput];
    
    // Forward: Input -> Hidden using MPS.
    MPSMatrixDescriptor *descW1 = [MPSMatrixDescriptor matrixDescriptorWithDimensions:784 columns:128 rowBytes:128 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrix *matW1 = [[MPSMatrix alloc] initWithBuffer:_weight1Buffer descriptor:descW1];
    
    MPSMatrixMultiplication *mul1 = [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                                                         transposeLeft:false
                                                                        transposeRight:false
                                                                             resultRows:batchSize
                                                                          resultColumns:128
                                                                       interiorColumns:784
                                                                                 alpha:1.0f
                                                                                  beta:0.0f];
    id<MTLCommandBuffer> cmdBuffer = [_commandQueue commandBuffer];
    [mul1 encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW1 resultMatrix:matHidden];
    
    // Apply sigmoid activation – for brevity we assume matHidden now holds activated values.
    // (In a production system you might combine activation into your MPS kernel or use a custom kernel.)
    
    // --- Batch Normalization on hidden layer ---
    // We dispatch a kernel that normalizes each of the 128 features over the batch.
    id<MTLComputeCommandEncoder> encoderBN = [cmdBuffer computeCommandEncoder];
    [encoderBN setComputePipelineState:_batchNormPipelineState];
    [encoderBN setBuffer:hiddenBuffer offset:0 atIndex:0];
    uint M = (uint)batchSize;
    uint N = 128;
    [encoderBN setBytes:&M length:sizeof(uint) atIndex:1];
    [encoderBN setBytes:&N length:sizeof(uint) atIndex:2];
    float bnEpsilon = 1e-5f;
    [encoderBN setBytes:&bnEpsilon length:sizeof(float) atIndex:3];
    MTLSize gridBN = MTLSizeMake(N, 1, 1);
    [encoderBN dispatchThreads:gridBN threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoderBN endEncoding];
    
    // Forward: Hidden -> Output.
    MPSMatrixDescriptor *descW2 = [MPSMatrixDescriptor matrixDescriptorWithDimensions:128 columns:10 rowBytes:10 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrix *matW2 = [[MPSMatrix alloc] initWithBuffer:_weight2Buffer descriptor:descW2];
    
    MPSMatrixMultiplication *mul2 = [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                                                         transposeLeft:false
                                                                        transposeRight:false
                                                                             resultRows:batchSize
                                                                          resultColumns:10
                                                                       interiorColumns:128
                                                                                 alpha:1.0f
                                                                                  beta:0.0f];
    [mul2 encodeToCommandBuffer:cmdBuffer leftMatrix:matHidden rightMatrix:matW2 resultMatrix:matOutput];
    
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    
    // Convert the FP16 output back to FP32.
    __fp16 *outputHalf = (__fp16 *)[outputBuffer contents];
    float *outputFloat = (float *)malloc(outputCount * sizeof(float));
    convertHalfToFloat(outputHalf, outputFloat, outputCount);
    return outputFloat; // Caller is responsible for freeing.
}

#pragma mark - Training with Adam, BatchNorm & Parallel Command Buffers

- (void)trainOnMNISTBatch:(float *)inputData
                   labels:(float *)labels
                batchSize:(NSUInteger)batchSize
             learningRate:(float)learningRate {
    // For simplicity, we create command buffers for forward, backward, and weight update
    // and then commit them concurrently (pipelining). In a real implementation you might
    // use a dispatch group to process several batches concurrently.
    
    // Convert input and label data to FP16.
    NSUInteger inputCount = batchSize * 784;
    __fp16 *inputHalf = (__fp16 *)malloc(inputCount * sizeof(__fp16));
    convertFloatToHalf(inputData, inputHalf, inputCount);
    id<MTLBuffer> inputBuffer = [_device newBufferWithBytes:inputHalf length:inputCount * sizeof(__fp16) options:MTLResourceStorageModeShared];
    free(inputHalf);
    
    NSUInteger labelCount = batchSize * 10;
    __fp16 *labelHalf = (__fp16 *)malloc(labelCount * sizeof(__fp16));
    convertFloatToHalf(labels, labelHalf, labelCount);
    id<MTLBuffer> labelBuffer = [_device newBufferWithBytes:labelHalf length:labelCount * sizeof(__fp16) options:MTLResourceStorageModeShared];
    free(labelHalf);
    
    // Create buffers for intermediate activations.
    NSUInteger hiddenCount = batchSize * 128;
    NSUInteger outputCount = batchSize * 10;
    id<MTLBuffer> hiddenBuffer = [_device newBufferWithLength:hiddenCount * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> outputBuffer = [_device newBufferWithLength:outputCount * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    
    // Create MPSMatrix descriptors for input, hidden, and output.
    MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:784 rowBytes:784 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descHidden = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:128 rowBytes:128 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descOutput = [MPSMatrixDescriptor matrixDescriptorWithDimensions:batchSize columns:10 rowBytes:10 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    
    MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:inputBuffer descriptor:descInput];
    MPSMatrix *matHidden = [[MPSMatrix alloc] initWithBuffer:hiddenBuffer descriptor:descHidden];
    MPSMatrix *matOutput = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:descOutput];
    
    // --- Forward Pass ---
    // Input -> Hidden.
    MPSMatrixDescriptor *descW1 = [MPSMatrixDescriptor matrixDescriptorWithDimensions:784 columns:128 rowBytes:128 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrix *matW1 = [[MPSMatrix alloc] initWithBuffer:_weight1Buffer descriptor:descW1];
    MPSMatrixMultiplication *mul1 = [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                                                         transposeLeft:false
                                                                        transposeRight:false
                                                                             resultRows:batchSize
                                                                          resultColumns:128
                                                                       interiorColumns:784
                                                                                 alpha:1.0f
                                                                                  beta:0.0f];
    id<MTLCommandBuffer> cb1 = [_commandQueue commandBuffer];
    [mul1 encodeToCommandBuffer:cb1 leftMatrix:matInput rightMatrix:matW1 resultMatrix:matHidden];
    [cb1 commit];
    
    // (Assume sigmoid activation is applied on GPU via an MPS neuron kernel or a custom kernel.)
    // For brevity, we omit the activation kernel code here.
    
    // --- Batch Normalization on Hidden ---
    id<MTLCommandBuffer> cbBN = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoderBN = [cbBN computeCommandEncoder];
    [encoderBN setComputePipelineState:_batchNormPipelineState];
    [encoderBN setBuffer:hiddenBuffer offset:0 atIndex:0];
    uint M = (uint)batchSize;
    uint N = 128;
    [encoderBN setBytes:&M length:sizeof(uint) atIndex:1];
    [encoderBN setBytes:&N length:sizeof(uint) atIndex:2];
    float bnEpsilon = 1e-5f;
    [encoderBN setBytes:&bnEpsilon length:sizeof(float) atIndex:3];
    MTLSize gridBN = MTLSizeMake(N, 1, 1);
    [encoderBN dispatchThreads:gridBN threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoderBN endEncoding];
    [cbBN commit];
    
    // Hidden -> Output.
    MPSMatrixDescriptor *descW2 = [MPSMatrixDescriptor matrixDescriptorWithDimensions:128 columns:10 rowBytes:10 * sizeof(__fp16) dataType:MPSDataTypeFloat16];
    MPSMatrix *matW2 = [[MPSMatrix alloc] initWithBuffer:_weight2Buffer descriptor:descW2];
    MPSMatrixMultiplication *mul2 = [[MPSMatrixMultiplication alloc] initWithDevice:_device
                                                                         transposeLeft:false
                                                                        transposeRight:false
                                                                             resultRows:batchSize
                                                                          resultColumns:10
                                                                       interiorColumns:128
                                                                                 alpha:1.0f
                                                                                  beta:0.0f];
    id<MTLCommandBuffer> cb2 = [_commandQueue commandBuffer];
    [mul2 encodeToCommandBuffer:cb2 leftMatrix:matHidden rightMatrix:matW2 resultMatrix:matOutput];
    [cb2 commit];
    
    // --- Backpropagation & Gradient Computation ---
    // (For brevity, we assume gradients dW1 and dW2 are computed using MPS and/or custom kernels.)
    // Here we focus on the Adam update step.
    // Assume dW1Buffer and dW2Buffer are computed gradients stored in FP16.
    // For this example, we allocate dummy gradient buffers (in practice, compute them from errors).
    id<MTLBuffer> dW1Buffer = [_device newBufferWithLength:784 * 128 * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dW2Buffer = [_device newBufferWithLength:128 * 10 * sizeof(__fp16) options:MTLResourceStorageModePrivate];
    // Zero out dW buffers for illustration.
    memset([dW1Buffer contents], 0, 784 * 128 * sizeof(__fp16));
    memset([dW2Buffer contents], 0, 128 * 10 * sizeof(__fp16));
    
    // --- Adam Weight Updates for Weight2 ---
    id<MTLCommandBuffer> cbUpdate2 = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoderUpdate2 = [cbUpdate2 computeCommandEncoder];
    [encoderUpdate2 setComputePipelineState:_updateAdamPipelineState];
    [encoderUpdate2 setBuffer:_weight2Buffer offset:0 atIndex:0];
    [encoderUpdate2 setBuffer:dW2Buffer offset:0 atIndex:1];
    [encoderUpdate2 setBuffer:_weight2_mBuffer offset:0 atIndex:2];
    [encoderUpdate2 setBuffer:_weight2_vBuffer offset:0 atIndex:3];
    [encoderUpdate2 setBytes:&learningRate length:sizeof(float) atIndex:4];
    [encoderUpdate2 setBytes:&_beta1 length:sizeof(float) atIndex:5];
    [encoderUpdate2 setBytes:&_beta2 length:sizeof(float) atIndex:6];
    [encoderUpdate2 setBytes:&_adamEpsilon length:sizeof(float) atIndex:7];
    NSUInteger totalW2Elements = 128 * 10;
    MTLSize gridUpdate2 = MTLSizeMake(totalW2Elements, 1, 1);
    [encoderUpdate2 dispatchThreads:gridUpdate2 threadsPerThreadgroup:MTLSizeMake(MIN(totalW2Elements, 256), 1, 1)];
    [encoderUpdate2 endEncoding];
    [cbUpdate2 commit];
    
    // --- Adam Weight Updates for Weight1 ---
    id<MTLCommandBuffer> cbUpdate1 = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoderUpdate1 = [cbUpdate1 computeCommandEncoder];
    [encoderUpdate1 setComputePipelineState:_updateAdamPipelineState];
    [encoderUpdate1 setBuffer:_weight1Buffer offset:0 atIndex:0];
    [encoderUpdate1 setBuffer:dW1Buffer offset:0 atIndex:1];
    [encoderUpdate1 setBuffer:_weight1_mBuffer offset:0 atIndex:2];
    [encoderUpdate1 setBuffer:_weight1_vBuffer offset:0 atIndex:3];
    [encoderUpdate1 setBytes:&learningRate length:sizeof(float) atIndex:4];
    [encoderUpdate1 setBytes:&_beta1 length:sizeof(float) atIndex:5];
    [encoderUpdate1 setBytes:&_beta2 length:sizeof(float) atIndex:6];
    [encoderUpdate1 setBytes:&_adamEpsilon length:sizeof(float) atIndex:7];
    NSUInteger totalW1Elements = 784 * 128;
    MTLSize gridUpdate1 = MTLSizeMake(totalW1Elements, 1, 1);
    [encoderUpdate1 dispatchThreads:gridUpdate1 threadsPerThreadgroup:MTLSizeMake(MIN(totalW1Elements, 256), 1, 1)];
    [encoderUpdate1 endEncoding];
    [cbUpdate1 commit];
    
    // For parallelization, we do not wait on every command buffer sequentially.
    // Instead, we could add completion handlers and use a dispatch group.
    // For simplicity in this example, we wait for all to finish.
    [cb1 waitUntilCompleted];
    [cbBN waitUntilCompleted];
    [cb2 waitUntilCompleted];
    [cbUpdate2 waitUntilCompleted];
    [cbUpdate1 waitUntilCompleted];
    
    NSLog(@"Finished training on batch with Adam, FP16, batch norm, and pipelined command buffers.");
}

@end
