#pragma once

#include <cstdint>

#include <glad/glad.h>
#include <glm/glm.hpp>

class Renderer
{
public:
	static void Init(size_t particle_count);
	static void SetParticlePosition(size_t index, const glm::dvec2& position);
	static void Render();

private:
	Renderer();

private:
	static size_t particle_count;
	static GLuint shader_program;
	static GLuint particle_vao;
	static GLuint particle_vbo;
	static float* positions;
};
