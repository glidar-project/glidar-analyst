#version 330 core

uniform mat4 modelViewProjection;

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

out vec3 gFragmentNormal;
out vec3 gFragmentPosition;

void main()
{
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;

	vec3 u = p0 - p1;
	vec3 v = p0 - p2;

	vec3 n = normalize(cross(u, v));

	gl_Position = modelViewProjection * vec4(p0.xyz, 1.0);
	gFragmentPosition = p0;
	gFragmentNormal = n;
	EmitVertex();

	gl_Position = modelViewProjection * vec4(p1.xyz, 1.0);
	gFragmentPosition = p1;
	gFragmentNormal = n;
	EmitVertex();

	gl_Position = modelViewProjection * vec4(p2.xyz, 1.0);
	gFragmentPosition = p2;
	gFragmentNormal = n;
	EmitVertex();

	EndPrimitive();
}