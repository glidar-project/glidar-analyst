#version 410 core

uniform mat4 modelViewProjection;

layout (location = 0) in vec3 posAttr;
layout (location = 1) in vec3 norAttr;

out vec3 vertexNormal;
out vec3 vertexPosition;

void main() {

    gl_Position = modelViewProjection * vec4(posAttr, 1.);

    vertexPosition = posAttr;
    vertexNormal = norAttr;
}