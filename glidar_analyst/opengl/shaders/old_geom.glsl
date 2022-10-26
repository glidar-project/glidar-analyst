#version 410 core

uniform sampler1D atomDataSampler;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float radiusScale;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec4 gFragmentPosition;

void main()
{

    int idx = int (gl_in[0].gl_Position.w);
    vec4 atomData = texelFetch(atomDataSampler, idx);

	vec4 p0 = modelViewMatrix * vec4(gl_in[0].gl_Position.xyz,1.0);

	vec3 up = radiusScale * vec3(0.0, 1.0, 0.0);
	vec3 right = radiusScale * vec3(1.0, 0.0, 0.0);


	gFragmentPosition = projectionMatrix*vec4(p0.xyz - right - up, 1.0);
	gl_Position = gFragmentPosition;
	EmitVertex();

	gFragmentPosition = projectionMatrix*vec4(p0.xyz + right - up, 1.0);
	gl_Position = gFragmentPosition;
	EmitVertex();

	gFragmentPosition = projectionMatrix*vec4(p0.xyz - right + up, 1.0);
	gl_Position = gFragmentPosition;
	EmitVertex();

	gFragmentPosition = projectionMatrix*vec4(p0.xyz + right + up, 1.0);
	gl_Position = gFragmentPosition;
	EmitVertex();

	EndPrimitive();
}