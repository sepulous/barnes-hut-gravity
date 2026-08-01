#pragma once

const char* POINT_VERTEX_SHADER = R"(
#version 460 core

layout(location = 0) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    gl_PointSize = 3.0;
}
)";

const char* POINT_FRAGMENT_SHADER = R"(
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
