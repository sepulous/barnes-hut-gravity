#pragma once

#include <cstdint>

struct SimulationSettings
{
    unsigned max_steps;
    unsigned particle_count;
    unsigned leaf_capacity;
    unsigned maximum_tree_depth; // up to limit determined by GPU memory
    unsigned threads_per_block;
    float softening;
    float theta;
    float time_step;
};
