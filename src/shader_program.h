#pragma once

#include <glad/glad.h>

class ShaderProgram
{
public:
    ShaderProgram(const char* vertex_shader, const char* fragment_shader);

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    ShaderProgram(ShaderProgram&&) = delete;
    ShaderProgram& operator=(ShaderProgram&&) = delete;

    void Use();

private:
    GLuint id_;
};
