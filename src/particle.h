#pragma once

#include <glm/glm.hpp>

struct Particle
{
    glm::vec2 position;
    glm::vec2 next_position;
    glm::vec2 velocity;
    glm::vec2 acceleration;
    float mass;
};
