#version 330 core

layout(location = 0) out vec4 out_color;


in vec3 vertexNormal;
in vec3 vertexPosition;


uniform sampler1D terrainSampler;

void main() {

    vec3 lightDir_vspace = -normalize(vec3(1,1,1));

    vec4 terrainColor = texture(terrainSampler, (vertexPosition.z + 250.0f) / 1700.0f);

    vec3 color = clamp(dot(normalize(vertexNormal), lightDir_vspace), 0, 1) * terrainColor.rgb; // + vec3(0.3, 0., 0.);

    out_color = vec4(clamp(color,0,1), 1.);
//    out_color = vec4(abs(vertexNormal), 1.);
}