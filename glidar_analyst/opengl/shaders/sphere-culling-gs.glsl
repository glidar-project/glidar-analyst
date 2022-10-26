#version 450

// Ligand Coloring generated from http://vrl.cs.brown.edu/color
vec3 ligandColors[5] = {vec3(82,239,153),
                        vec3(56,90,58),
                        vec3(175,211,90),
                        vec3(53,151,33),
                        vec3(121,200,174)};

/************************************
*   Atom data [radius, element_number, partial_charges, ligand_idx]
*/
uniform sampler1D atomDataSampler;
uniform sampler1D colorSampler;
uniform sampler1D chargeColormapSampler;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float radiusScale;
uniform vec3 ligandPosition;
uniform vec2 shellRadii;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec4 gFragmentPosition;
flat out vec3 gSpherePosition;
flat out float gSphereRadius;
flat out vec3 gSphereColor;

// TODO: Compute the energy contributoion and send it to frag.
// Figure out how to get the charge in.
// Get the radius in, filter the atoms in given layer
flat out vec4 gAtomData;

void main()
{
    int idx = int(gl_in[0].gl_Position.w);
    vec4 atomData = texelFetch(atomDataSampler, idx, 0);
    gAtomData = atomData;

    bool molecule = false;

    // Ligand coloring
    if (atomData.w > 0) {
        gSphereColor = ligandColors[int(atomData.w) - 1] / 255.0;
    } else {
        int cidx = int(clamp(atomData.g, 1., 109.)) - 1;
        gSphereColor = texelFetch(colorSampler, cidx, 0).rgb;
        molecule = true;
    }

    // Sphere radius is expected to be stored in the 4th coordinate
//	float sphereRadius = gl_in[0].gl_Position.w*radiusScale;
    //Here the radius comes from the texture
	float sphereRadius = atomData.r * radiusScale;

	// Output variables for fragmetn shading
	gSpherePosition = gl_in[0].gl_Position.xyz;
	gSphereRadius = sphereRadius;

    float ligandDistance = length(gSpherePosition - ligandPosition);


    if (molecule && ligandDistance < shellRadii.y && ligandDistance > shellRadii.x) {
        float el_charge_potential = atomData.z / ligandDistance;
        gSphereColor = texture(chargeColormapSampler, el_charge_potential).rgb;
    } else if (molecule && ligandDistance < shellRadii.x) {
        return;
    }

    // The quads have to be computed in the view space
	vec4 p0 = modelViewMatrix * vec4(gl_in[0].gl_Position.xyz,1.0);
	vec4 p1 = modelViewMatrix * (vec4(gl_in[0].gl_Position.xyz,1.0)+vec4(0.0,sphereRadius,0.0,0.0));
	float radius = length(p1.xyz-p0.xyz);

	// up and right vectors in view space
//	vec3 up = vec3(0.0, 1.0, 0.0) * radius*sqrt(2.0);
	vec3 up = vec3(0.0, 1.0, 0.0) * radius*2.0;
//	vec3 right = vec3(1.0, 0.0, 0.0) * radius*sqrt(2.0);
	vec3 right = vec3(1.0, 0.0, 0.0) * radius*2.0;

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