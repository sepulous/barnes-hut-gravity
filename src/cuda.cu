#include <cmath>
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING 

#include <cstdint>
#include <cstring>

#include <iostream>

#include "cuda.cuh"

#include <cuda_runtime_api.h>
#include <cuda/cmath>

constexpr float G = 6.6743e-9f;

__global__ void KernelComputeAccelerations(const GPUParticle* __restrict__ particles,
	                                       const GPUTreeNode* __restrict__ tree,
	                                       float* __restrict__ accelerations,
	                                       uint32_t num_particles,
										   float theta,
	                                       float softening)
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

		float com_displacement_x = node.com_x - particle.position_x;
		float com_displacement_y = node.com_y - particle.position_y;
		float com_distance_squared = com_displacement_x * com_displacement_x + com_displacement_y * com_displacement_y;

		if (node.width_squared < com_distance_squared * theta * theta)
		{
			float inv_r3 = rsqrtf(com_distance_squared + softening); // 1 / sqrt(r^2 + ε^2)
			inv_r3 *= inv_r3 * inv_r3 * G * node.total_mass;
			new_acceleration_x += inv_r3 * com_displacement_x;
			new_acceleration_y += inv_r3 * com_displacement_y;
		}
		else if (node.has_children)
		{
			stack[stack_head] = node.first_child_index;
			stack[stack_head + 1] = node.first_child_index + 1;
			stack[stack_head + 2] = node.first_child_index + 2;
			stack[stack_head + 3] = node.first_child_index + 3;
			stack_head += 4;
		}
		else
		{
			for (uint32_t i = node.particles_start_index; i < node.particles_end_index; i++)
			{
				uint32_t not_particle = i != particle_index;

				GPUParticle other_particle = particles[i];

				float displacement_x = other_particle.position_x - particle.position_x;
				float displacement_y = other_particle.position_y - particle.position_y;
				float distance_squared = displacement_x * displacement_x + displacement_y * displacement_y;
				float inv_r3 = rsqrtf(distance_squared + softening); // 1 / sqrt(r^2 + ε^2)
				inv_r3 *= inv_r3 * inv_r3 * G * not_particle * other_particle.mass;
				new_acceleration_x += inv_r3 * displacement_x;
				new_acceleration_y += inv_r3 * displacement_y;
			}
		}
	}

	accelerations[2 * particle_index] = new_acceleration_x;
	accelerations[2 * particle_index + 1] = new_acceleration_y;
}

void ComputeAccelerationsCUDA(CUDAContext& cuda_context, std::vector<Particle>& particles, std::vector<QuadTree>& tree, float* new_accelerations, SimulationSettings settings)
{
	size_t blocks = (particles.size() + settings.threads_per_block - 1) / settings.threads_per_block;

	// Move particle data to GPU
	for (int i = 0; i < particles.size(); i++)
	{
		Particle& particle = particles[i];
		cuda_context.particles[i] = GPUParticle {
			particle.position.x,
			particle.position.y,
			particle.mass
		};
	}

	// Move tree to GPU
	for (int i = 0; i < tree.size(); i++)
	{
		QuadTree& node = tree[i];
		auto& node_particles = node.GetParticles();
		cuda_context.tree[i] = GPUTreeNode{
			node.GetCenterOfMass().x,
			node.GetCenterOfMass().y,
			node.GetTotalMass(),
			node.GetWidth() * node.GetWidth(),
			node.GetFirstChildIndex(),
			node_particles.empty() ? 0 : static_cast<uint32_t>(&node_particles.front() - particles.data()),
			node_particles.empty() ? 0 : static_cast<uint32_t>(&node_particles.back() - particles.data() + 1),
			node.HasChildren()
		};
	}

	KernelComputeAccelerations<<<blocks, settings.threads_per_block>>>(
		cuda_context.particles,
		cuda_context.tree,
		cuda_context.accelerations,
		static_cast<uint32_t>(particles.size()),
		settings.theta,
		settings.softening
	);

	cudaMemcpy(new_accelerations, cuda_context.accelerations, 2 * sizeof(float) * particles.size(), cudaMemcpyDeviceToHost);
}

CUDAContext::CUDAContext(size_t particle_count, size_t max_tree_depth)
{
	this->particle_count = particle_count;
	this->max_tree_depth = max_tree_depth;
	cudaMallocManaged(reinterpret_cast<void**>(&accelerations), 2 * sizeof(float) * particle_count);
	cudaMallocManaged(reinterpret_cast<void**>(&particles), sizeof(GPUParticle) * particle_count);
	cudaMallocManaged(reinterpret_cast<void**>(&tree), sizeof(GPUTreeNode) * std::pow(4, max_tree_depth));
}

void CUDAContext::Realloc(size_t particle_count, size_t max_tree_depth)
{
	if (particle_count != this->particle_count)
	{
		float* new_accelerations = nullptr;
		cudaMallocManaged(reinterpret_cast<void**>(&new_accelerations), 2 * sizeof(float) * particle_count);
		cudaFree(accelerations);
		accelerations = new_accelerations;

		GPUParticle* new_particles = nullptr;
		cudaMallocManaged(reinterpret_cast<void**>(&new_particles), sizeof(GPUParticle) * particle_count);
		cudaFree(particles);
		particles = new_particles;

		this->particle_count = particle_count;
	}

	if (max_tree_depth != this->max_tree_depth)
	{
		GPUTreeNode* new_tree = nullptr;
		cudaMallocManaged(reinterpret_cast<void**>(&new_tree), sizeof(GPUTreeNode) * std::pow(4, max_tree_depth));
		cudaFree(tree);
		tree = new_tree;

		this->max_tree_depth = max_tree_depth;
	}
}
