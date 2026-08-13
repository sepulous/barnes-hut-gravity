#pragma once

struct GPUInfo
{
	unsigned warp_size;
	unsigned max_threads_per_block;
	bool cuda_supported;
};

GPUInfo GetGPUInfo();
