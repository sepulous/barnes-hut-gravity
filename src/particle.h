#pragma once

#include <glm/glm.hpp>

struct Particle
{
    glm::dvec2 position;
    glm::dvec2 next_position;
    glm::dvec2 velocity;
    glm::dvec2 acceleration;
    double mass;
};
