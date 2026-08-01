#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.h"

Shader::Shader(const std::filesystem::path &fragment_shader_path, const std::filesystem::path &vertex_shader_path, const VertexAttribs &vertex_attribs)
{
    fragment_shader_path_ = fragment_shader_path;
    vertex_shader_path_ = vertex_shader_path;
    vertex_attribs_ = vertex_attribs;
}

Shader::Shader() {}

void Shader::SetFragmentShader(const std::filesystem::path &fragment_shader_path)
{
    fragment_shader_path_ = fragment_shader_path;
}

void Shader::SetVertexShader(const std::filesystem::path &vertex_shader_path, const VertexAttribs &vertex_attribs)
{
    vertex_shader_path_ = vertex_shader_path;
    vertex_attribs_ = vertex_attribs;
}

void Shader::Compile()
{
    int success;
    char error_log[512];

    //
    // Vertex shader
    //
    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    std::ifstream vertex_shader_file(vertex_shader_path_);
    std::string vertex_shader_source((std::istreambuf_iterator<char>(vertex_shader_file)), std::istreambuf_iterator<char>());
    const char *vertex_shader_source_cstr = vertex_shader_source.c_str();
    glShaderSource(vertex_shader_id, 1, &vertex_shader_source_cstr, NULL);
    glCompileShader(vertex_shader_id);

    // Check for errors
    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertex_shader_id, 512, NULL, error_log);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << error_log << std::endl;
    }

    //
    // Fragment shader
    //
    GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    std::ifstream fragment_shader_file(fragment_shader_path_);
    std::string fragment_shader_source((std::istreambuf_iterator<char>(fragment_shader_file)), std::istreambuf_iterator<char>());
    const char *fragment_shader_source_cstr = fragment_shader_source.c_str();
    glShaderSource(fragment_shader_id, 1, &fragment_shader_source_cstr, NULL);
    glCompileShader(fragment_shader_id);

    // Check for errors
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragment_shader_id, 512, NULL, error_log);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << error_log << std::endl;
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
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << error_log << std::endl;
    }
}

void Shader::Use()
{
    glUseProgram(id_);
}

GLuint Shader::GetID()
{
    return id_;
}

VertexAttribs Shader::GetVertexAttribs()
{
    return vertex_attribs_;
}
