#version 330 core

layout(location = 0) out vec4 out_color;


in vec3 vertexNormal;
in vec3 vertexPosition;

uniform sampler2D textureSampler;
uniform sampler1D terrainSampler;

const float PI = 3.141592;

float atan2(in float y, in float x)
{
    bool s = (abs(x) > abs(y));
    return mix(PI/2.0 - atan(x,y), atan(y,x), s);
}

void main() {

    vec3 lightDir_vspace = -normalize(vec3(1,0,-1));

    //vec4 terrainColor = texture(terrainSampler, (vertexPosition.z + 250.0f) / 1700.0f);

    float lat = acos(vertexNormal.z);
    float lon = atan2(vertexNormal.y, vertexNormal.x);

    vec4 terrainColor = texture(textureSampler, vec2(lon/6.28f + 1.0f, lat/PI + 1.0f));

    vec3 color = (clamp(dot(normalize(vertexNormal), lightDir_vspace), 0, 1) + vec3(0.1, 0.1, 0.1)) * terrainColor.rgb;

    out_color = vec4(clamp(color,0,1), 1.);
//    out_color = vec4(abs(vertexNormal), 1.);
}