#include <iostream>

#include "shader_program.h"

// NOTE: This immediately compiles and links the given source code
ShaderProgram::ShaderProgram(const char* vertex_shader, const char* fragment_shader)
{
    int success;
    char error_log[512];

    //
    // Vertex shader
    //
    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader_id, 1, &vertex_shader, NULL);
    glCompileShader(vertex_shader_id);

    // Check for errors
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex_shader_id, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to compile vertex shader.\n" << error_log << std::endl;
    }

    //
    // Fragment shader
    //
    GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader_id, 1, &fragment_shader, NULL);
    glCompileShader(fragment_shader_id);

    // Check for errors
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment_shader_id, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to compile fragment shader.\n" << error_log << std::endl;
    }

    //
    // Create shader program
    //
    id_ = glCreateProgram();
    glAttachShader(id_, vertex_shader_id);
    glAttachShader(id_, fragment_shader_id);
    glLinkProgram(id_);
    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    // Check for errors
    glGetShaderiv(id_, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(id_, 512, NULL, error_log);
        std::cerr << "ERROR: Failed to link shader program.\n" << error_log << std::endl;
    }
}

void ShaderProgram::Use()
{
	glUseProgram(id_);
}
