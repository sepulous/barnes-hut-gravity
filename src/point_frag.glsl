#version 460 core

out vec4 color;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);

    if (length(coord) > 0.5)
        discard;

    color = vec4(1.0);
}
