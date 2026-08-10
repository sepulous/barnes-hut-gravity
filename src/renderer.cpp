#include <iostream>

#include "renderer.h"

size_t Renderer::particle_count;
GLuint Renderer::shader_program;
GLuint Renderer::particle_vao;
GLuint Renderer::particle_vbo;
float* Renderer::positions;

static const char* POINT_VERTEX_SHADER = R"(
#version 460 core

layout(location = 0) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    gl_PointSize = 3.0;
}
)";

static const char* POINT_FRAGMENT_SHADER = R"(
#version 460 core

out vec4 color;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);

    if (length(coord) > 0.5)
        discard;

    color = vec4(1.0);
}
)";

void Renderer::Init(size_t particle_count)
{
	Renderer::particle_count = particle_count;

	//
	// Set up shader program
	//

    int success;
    char error_log[512];

    // Vertex shader
    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader_id, 1, &POINT_VERTEX_SHADER, NULL);
    glCompileShader(vertex_shader_id);

    // Check for errors
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex_shader_id, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to compile vertex shader.\n" << error_log << std::endl;
    }

    // Fragment shader
    GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader_id, 1, &POINT_FRAGMENT_SHADER, NULL);
    glCompileShader(fragment_shader_id);

    // Check for errors
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment_shader_id, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to compile fragment shader.\n" << error_log << std::endl;
    }

    // Create shader program
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader_id);
    glAttachShader(shader_program, fragment_shader_id);
    glLinkProgram(shader_program);
    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    // Check for errors
    glGetShaderiv(shader_program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader_program, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to link shader program.\n" << error_log << std::endl;
    }

    //
    // Set up particle rendering
    //

	glEnable(GL_PROGRAM_POINT_SIZE);

	glGenVertexArrays(1, &particle_vao);
	glBindVertexArray(particle_vao);

	glCreateBuffers(1, &particle_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);

	glNamedBufferStorage(
		particle_vbo,
		particle_count * (2 * sizeof(float)),
		nullptr,
		GL_MAP_WRITE_BIT |
		GL_MAP_PERSISTENT_BIT |
		GL_MAP_COHERENT_BIT
	);

	positions = static_cast<float*>(
		glMapNamedBufferRange(
			particle_vbo,
			0,
			particle_count * (2 * sizeof(float)),
			GL_MAP_WRITE_BIT |
			GL_MAP_PERSISTENT_BIT |
			GL_MAP_COHERENT_BIT
		)
	);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);
}

void Renderer::SetParticlePosition(size_t index, const glm::dvec2& position)
{
    assert(index >= 0 && index < particle_count);
    positions[2 * index] = static_cast<float>(position.x);
    positions[2 * index + 1] = static_cast<float>(position.y);
}

void Renderer::Render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program);
    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, particle_count);
}
