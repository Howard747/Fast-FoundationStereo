#include "gwc_volume_kernel.h"

#include <cuda_runtime.h>


__global__ void preprocess_rgb_to_planar_kernel(
    const uint8_t* __restrict__ src, 
    float* __restrict__ dst, 
    int width, int height, 
    float scale, 
    float meanR, float meanG, float meanB,
    float stdR, float stdG, float stdB) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    int area = width * height;

    // Load BGR from interleaved source
    uint8_t r = src[src_idx + 0];
    uint8_t g = src[src_idx + 1];
    uint8_t b = src[src_idx + 2];

    // Destination indices for Planar RGB
    int dst_idx_r = y * width + x;
    int dst_idx_g = area + dst_idx_r;
    int dst_idx_b = area * 2 + dst_idx_r;

    // Convert, Scale, and Normalize
    // If you don't want normalization, set means to 0 and stds to 1.
    dst[dst_idx_r] = ((float)r * scale - meanR) / stdR;
    dst[dst_idx_g] = ((float)g * scale - meanG) / stdG;
    dst[dst_idx_b] = ((float)b * scale - meanB) / stdB;
}

// Wrapper function to call from C++
void LaunchPreprocessKernel(
    const uint8_t* d_src, float* d_dst, 
    int width, int height, cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // ImageNet normalization constants (example)
    // Formula: (x/255.0 - mean) / std  => (x * (1/255.0) - mean) / std
    preprocess_rgb_to_planar_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, width, height, 
        1.0f, 
        0.0f, 0.0f, 0.0f, 
        1.0f, 1.0f, 1.0f
    );
}


// CUDA kernel for float to half conversion
__global__ void floatToHalfKernel(const float* input, half* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void convertFloatToHalf(const float* d_floatData, half* d_halfData, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    floatToHalfKernel<<<gridSize, blockSize>>>(d_floatData, d_halfData, size);
    cudaDeviceSynchronize();
}

// Simplified kernel for batch=1
__global__ void gwc_volume_kernel_simple(
    const half* refimg_fea,    // [C, H, W]
    const half* targetimg_fea, // [C, H, W]
    half* cost_volume,         // [G, D, H, W]
    int C, int H, int W,
    int D, int G,
    bool normalize)
{
    int C_g = C / G;
    
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z % D;
    int g = blockIdx.z / D;
    
    if (g >= G || d >= D || h >= H || w >= W)
        return;
    
    half dot_product = 0.0f;
    int tgt_w = w - d;
    
    if (tgt_w >= 0) {
        // Compute dot product
        for (int cg = 0; cg < C_g; ++cg) {
            int c = g * C_g + cg;
            
            int ref_idx = ((c * H) + h) * W + w;
            int tgt_idx = ((c * H) + h) * W + tgt_w;
            
            dot_product += refimg_fea[ref_idx] * targetimg_fea[tgt_idx];
        }
        
        // Normalize
        if (normalize) {
            half ref_norm = 0.0f;
            half tgt_norm = 0.0f;
            
            for (int cg = 0; cg < C_g; ++cg) {
                int c = g * C_g + cg;
                int ref_idx = ((c * H) + h) * W + w;
                int tgt_idx = ((c * H) + h) * W + tgt_w;
                
                ref_norm += refimg_fea[ref_idx] * refimg_fea[ref_idx];
                tgt_norm += targetimg_fea[tgt_idx] * targetimg_fea[tgt_idx];
            }
            
            ref_norm = hsqrt(ref_norm);
            tgt_norm = hsqrt(tgt_norm);

            const half epsilon = __float2half(1e-4f);
            
            if (ref_norm > epsilon && tgt_norm > epsilon) {
                dot_product = dot_product / (ref_norm * tgt_norm);
            } else {
                dot_product = __float2half(0.0f);
            }
        }
    }
    
    int out_idx = (((g * D) + d) * H + h) * W + w;
    cost_volume[out_idx] = dot_product;
}

// Launch function for your specific use case
void LaunchGwcVolumeKernel(
    const half* d_refimg_fea,
    const half* d_targetimg_fea,
    half* d_cost_volume,
    int B, int C, int H, int W,
    int D, int G,
    bool normalize,
    cudaStream_t stream)
{
    // Assert batch size is 1
    if (B != 1) {
        //printf("Warning: Batch size should be 1, got %d\n", B);
        return;
    }
    
    // Set up grid and block dimensions
    dim3 block(16, 16, 1);  // 16x16 = 256 threads per block
    
    dim3 grid(
        (W + block.x - 1) / block.x,   // W dimension
        (H + block.y - 1) / block.y,   // H dimension
        D * G                           // Disparity * Groups dimension
    );
    
    // Launch kernel
    gwc_volume_kernel_simple<<<grid, block, 0, stream>>>(
        d_refimg_fea, d_targetimg_fea, d_cost_volume,
        C, H, W, D, G, normalize
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        //printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
}