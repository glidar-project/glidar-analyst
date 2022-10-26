#version 460

uniform sampler1D colormapSampler;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

uniform float radiusScale;
uniform float threshold;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec4 gFragmentPosition;
flat out vec3 gSpherePosition;
flat out float gSphereRadius;
flat out vec3 gSphereColor;


void main()
{
    float w = gl_in[0].gl_Position.w;

	if (w < threshold) { return; }

    gSphereColor = texture(colormapSampler, w).rgb;

	float sphereRadius = radiusScale;

	// Output variables for fragmetn shading
	gSpherePosition = gl_in[0].gl_Position.xyz;
	gSphereRadius = sphereRadius;

    // The quads have to be computed in the view space
	vec4 p0 = modelViewMatrix * vec4(gl_in[0].gl_Position.xyz,1.0);
	vec4 p1 = modelViewMatrix * (vec4(gl_in[0].gl_Position.xyz,1.0)+vec4(0.0,sphereRadius,0.0,0.0));
	float radius = length(p1.xyz-p0.xyz);

	// up and right vectors in view space
	vec3 up = normalize(vec3(0, p0.z, -p0.y)) * radius;
	vec3 right = normalize(vec3(p0.z, 0, -p0.x)) * radius;

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