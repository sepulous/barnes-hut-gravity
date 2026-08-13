#include <cuda_runtime_api.h>

#include "gpu_info.h"

GPUInfo GetGPUInfo()
{
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id != cudaSuccess || device_count == 0) // Failed to find CUDA device; CPU only
    {
        return {
            .warp_size = 0,
            .max_threads_per_block = 0,
            .cuda_supported = false
        };
    }
    else
    {
        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, 0); // Use first available GPU

        return {
            .warp_size = static_cast<unsigned>(device_props.warpSize),
            .max_threads_per_block = static_cast<unsigned>(device_props.maxThreadsPerBlock),
            .cuda_supported = true
        };
    }
}
