#version 410 core


layout (location = 0) in vec3 posAttr;


void main() {

    gl_Position = vec4(posAttr, 1.);

}