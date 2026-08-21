#pragma once

#include <vector>

#include "particle.h"
#include "quad_tree.h"
#include "simulation_settings.h"

struct GPUParticle
{
	float position_x;
	float position_y;
	float mass;
};

struct GPUTreeNode
{
	float com_x;
	float com_y;
	float total_mass;
	float width_squared;
	uint32_t first_child_index;
	uint32_t particles_start_index;
	uint32_t particles_end_index;
	bool has_children;
};

struct CUDAContext
{
	float* accelerations = nullptr;
	GPUParticle* particles = nullptr;
	GPUTreeNode* tree = nullptr;
	size_t particle_count = 0;
	size_t max_tree_depth = 0;

	CUDAContext() = default;
	CUDAContext(size_t particle_count, size_t max_tree_depth);
	void Realloc(size_t particle_count, size_t max_tree_depth);
};

void ComputeAccelerationsCUDA(CUDAContext& cuda_context, std::vector<Particle>& particles, std::vector<QuadTree>& tree, std::vector<float>& new_accelerations, SimulationSettings settings);
