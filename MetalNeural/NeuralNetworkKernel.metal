//
//  NeuralNetworkKernel.metal
//  MetalNeural
//
//  Created by Klöckner Gian-Luca on 25.02.25.
//
#include <metal_stdlib>
using namespace metal;

// Kernel for batch normalization on FP16 data.
// Each thread processes one feature (column) over M rows.
kernel void batch_normalization(
    device half *data [[ buffer(0) ]],
    constant uint &M [[ buffer(1) ]],
    constant uint &N [[ buffer(2) ]],
    constant float &epsilon [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
)
{
    if (gid >= N) return;
    float sum = 0.0;
    float sumSq = 0.0;
    for (uint i = 0; i < M; i++) {
        uint index = i * N + gid;
        float val = float(data[index]);
        sum += val;
        sumSq += val * val;
    }
    float mean = sum / float(M);
    float variance = sumSq / float(M) - mean * mean;
    float invStd = rsqrt(variance + epsilon);
    for (uint i = 0; i < M; i++) {
        uint index = i * N + gid;
        float val = float(data[index]);
        data[index] = half((val - mean) * invStd);
    }
}

// Kernel to update weights using the Adam optimizer.
// W, dW, mBuffer, and vBuffer are FP16 arrays.
kernel void update_weights_adam(
    device half *W         [[ buffer(0) ]],
    device half *dW        [[ buffer(1) ]],
    device half *mBuffer   [[ buffer(2) ]],
    device half *vBuffer   [[ buffer(3) ]],
    constant float &learningRate [[ buffer(4) ]],
    constant float &beta1  [[ buffer(5) ]],
    constant float &beta2  [[ buffer(6) ]],
    constant float &epsilon[[ buffer(7) ]],
    uint id [[ thread_position_in_grid ]]
)
{
    float W_val = float(W[id]);
    float dW_val = float(dW[id]);
    float m = float(mBuffer[id]);
    float v = float(vBuffer[id]);
    
    m = beta1 * m + (1.0f - beta1) * dW_val;
    v = beta2 * v + (1.0f - beta2) * dW_val * dW_val;
    
    // Optionally, apply bias correction here.
    float updated = W_val - learningRate * m / (sqrt(v) + epsilon);
    
    W[id] = half(updated);
    mBuffer[id] = half(m);
    vBuffer[id] = half(v);
}
