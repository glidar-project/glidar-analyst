#version 330 core

layout(location = 0) out vec4 out_color;

in vec3 gFragmentPosition;
in vec3 gFragmentNormal;

uniform sampler1D terrainSampler;

void main() {

    vec3 lightDir_vspace = -normalize(vec3(1,1,1));

    vec4 terrainColor = texture(terrainSampler, (gFragmentPosition.z + 250.0f) / 1700.0f);

    vec3 color = clamp(dot(gFragmentNormal, lightDir_vspace), 0, 1) * terrainColor.rgb; // + vec3(0.3, 0., 0.);

    out_color = vec4(clamp(color,0,1), 1.);
}