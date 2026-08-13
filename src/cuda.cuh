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
	float width;
	uint32_t child_top_left;
	uint32_t child_top_right;
	uint32_t child_bottom_right;
	uint32_t child_bottom_left;
	uint32_t particles_start_index;
	uint32_t particles_end_index;
	bool has_children;
};

struct CUDAContext
{
	float* accelerations = nullptr;
	GPUParticle* particles = nullptr;
	GPUTreeNode* tree = nullptr;

	CUDAContext() = default;
	CUDAContext(size_t num_particles, size_t max_tree_depth);
};

void ComputeAccelerationsCUDA(CUDAContext& cuda_context, std::vector<Particle>& particles, std::vector<QuadTree>& tree, float* new_accelerations, SimulationSettings settings);
