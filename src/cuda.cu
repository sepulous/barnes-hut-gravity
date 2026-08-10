#include <cmath>
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING 

#include <cstdint>
#include <cstring>

#include "cuda.cuh"

#include <cuda_runtime_api.h>
#include <cuda/cmath>

__global__ void KernelComputeAccelerations(const GPUParticle* __restrict__ particles,
	                                       const GPUTreeNode* __restrict__ tree,
	                                       float* __restrict__ accelerations,
	                                       uint32_t num_particles)
{
	uint32_t particle_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (particle_index >= num_particles)
		return;

	GPUParticle particle = particles[particle_index];

	float new_acceleration_x = 0;
	float new_acceleration_y = 0;

	uint32_t stack[36];
	stack[0] = 0;
	uint32_t stack_head = 1;

	while (stack_head > 0)
	{
		uint32_t node_index = stack[stack_head - 1];
		stack_head--;
		GPUTreeNode node = tree[node_index];

		// It's better to have this branch because all threads traverse the same tree in the same way,
		// so warp divergence is minimal. Making this branchless just adds extra work.
		if (!node.has_children)
		{
			for (uint32_t i = node.particles_start_index; i < node.particles_end_index; i++)
			{
				uint32_t not_particle = i != particle_index;

				GPUParticle other_particle = particles[i];

				float displacement_x = particle.position_x - other_particle.position_x;
				float displacement_y = particle.position_y - other_particle.position_y;
				float distance_squared = displacement_x * displacement_x + displacement_y * displacement_y;
				float inv_r3 = rsqrtf(distance_squared + 0.000001f); // 1 / sqrt(r^2 + ε^2)
				inv_r3 *= inv_r3 * inv_r3 * 6.67408e-9f * not_particle * other_particle.mass;
				new_acceleration_x += inv_r3 * displacement_x;
				new_acceleration_y += inv_r3 * displacement_y;
			}
		}
		else
		{
			float com_displacement_x = node.com_x - particle.position_x;
			float com_displacement_y = node.com_y - particle.position_y;
			float com_distance_squared = com_displacement_x * com_displacement_x + com_displacement_y * com_displacement_y;

			if (node.width * node.width < com_distance_squared * 0.64f)
			{
				float inv_r3 = rsqrtf(com_distance_squared + 0.000001f); // 1 / sqrt(r^2 + ε^2)
				inv_r3 *= inv_r3 * inv_r3 * 6.67408e-9f * node.total_mass;
				new_acceleration_x += inv_r3 * com_displacement_x;
				new_acceleration_y += inv_r3 * com_displacement_y;
			}
			else
			{
				stack[stack_head] = node.child_top_left;
				stack[stack_head + 1] = node.child_top_right;
				stack[stack_head + 2] = node.child_bottom_right;
				stack[stack_head + 3] = node.child_bottom_left;
				stack_head += 4;
			}
		}
	}

	accelerations[2 * particle_index] = new_acceleration_x;
	accelerations[2 * particle_index + 1] = new_acceleration_y;
}

void ComputeAccelerations(CUDAContext& cuda_context, std::vector<Particle>& particles, std::vector<QuadTree>& tree, float* new_accelerations)
{
	constexpr size_t THREADS_PER_BLOCK = 256;

	size_t blocks = (particles.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	// Move particle data to GPU
	for (int i = 0; i < particles.size(); i++)
	{
		Particle& particle = particles[i];
		cuda_context.particles[i] = GPUParticle {
			static_cast<float>(particle.position.x),
			static_cast<float>(particle.position.y),
			static_cast<float>(particle.mass)
		};
	}

	// Move tree to GPU
	for (int i = 0; i < tree.size(); i++)
	{
		QuadTree& node = tree[i];
		auto& node_particles = node.GetParticles();
		cuda_context.tree[i] = GPUTreeNode{
			static_cast<float>(node.GetCenterOfMass().x),
			static_cast<float>(node.GetCenterOfMass().y),
			static_cast<float>(node.GetTotalMass()),
			static_cast<float>(2 * node.GetExtents().x),
			node.GetChildren()[0],
			node.GetChildren()[1],
			node.GetChildren()[2],
			node.GetChildren()[3],
			node_particles.empty() ? 0 : static_cast<uint32_t>(&node_particles.front() - particles.data()),
			node_particles.empty() ? 0 : static_cast<uint32_t>(&node_particles.back() - particles.data() + 1),
			node.HasChildren()
		};
	}

	KernelComputeAccelerations<<<blocks, THREADS_PER_BLOCK>>>(cuda_context.particles, cuda_context.tree, cuda_context.accelerations, static_cast<uint32_t>(particles.size()));
	cudaDeviceSynchronize();

	cudaMemcpy(new_accelerations, cuda_context.accelerations, 2 * sizeof(float) * particles.size(), cudaMemcpyDeviceToHost);
}

CUDAContext::CUDAContext(size_t num_particles, size_t max_tree_depth)
{
	cudaMallocManaged((void**)&accelerations, 2 * sizeof(float) * num_particles);
	cudaMallocManaged((void**)&particles, sizeof(GPUParticle) * num_particles);
	cudaMallocManaged((void**)&tree, sizeof(GPUTreeNode) * std::pow(4, max_tree_depth));
}
