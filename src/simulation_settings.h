#pragma once

#include <cstdint>

struct SimulationSettings
{
    unsigned max_steps;
    unsigned leaf_capacity;
    unsigned maximum_tree_depth;
    unsigned threads_per_block;
    float gravity_strength;
    float softening;
    float theta;
    float time_step;
};
