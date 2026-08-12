#pragma once

#include <cstdint>

struct SimulationSettings
{
    size_t particle_count;
    size_t leaf_capacity;
    size_t maximum_tree_depth; // up to limit determined by GPU memory
    size_t threads_per_block;
    float softening;
    float theta;
    float time_step;
};
