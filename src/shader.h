#pragma once

#include <filesystem>
#include <vector>

#include <glad/glad.h>

using VertexAttribs = std::vector<std::pair<size_t, int>>;

class Shader
{
    private:
        GLuint id_;
        std::filesystem::path vertex_shader_path_;
        std::filesystem::path fragment_shader_path_;
        VertexAttribs vertex_attribs_;

    public:
        Shader();
        Shader(const std::filesystem::path &fragment_shader_path, const std::filesystem::path &vertex_shader_path, const VertexAttribs &vertex_attribs);

        Shader(const Shader&) = delete;
        Shader &operator=(const Shader&) = delete;

        Shader(Shader&&) = delete;
        Shader &operator=(Shader&&) = delete;

        void SetFragmentShader(const std::filesystem::path &fragment_shader_path);
        void SetVertexShader(const std::filesystem::path &vertex_shader_path, const VertexAttribs &vertex_attribs);
        void Compile();
        void Use();
        GLuint GetID();
        VertexAttribs GetVertexAttribs();
};
