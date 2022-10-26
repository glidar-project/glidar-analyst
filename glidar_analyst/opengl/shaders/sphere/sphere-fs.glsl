#version 410

uniform bool selected;

uniform mat3 normalMatrix;
uniform mat4 modelViewProjectionMatrix;
uniform mat4 inverseModelViewProjectionMatrix;

in vec4 gFragmentPosition;
flat in vec3 gSpherePosition;
flat in float gSphereRadius;
flat in vec3 gSphereColor;

out vec4 fragPosition;
out vec4 fragNormal;

layout(location = 0) out vec4 out_color;

struct Sphere
{			
	bool hit;
	vec3 near;
	vec3 far;
	vec3 normal;
};
																					
Sphere calcSphereIntersection(float r, vec3 origin, vec3 center, vec3 line)
{
	vec3 oc = origin - center;
	vec3 l = normalize(line);
	float loc = dot(l, oc);
	float under_square_root = loc * loc - dot(oc, oc) + r*r;
	if (under_square_root > 0)
	{
		float da = -loc + sqrt(under_square_root);
		float ds = -loc - sqrt(under_square_root);
		vec3 near = origin+min(da, ds) * l;
		vec3 far = origin+max(da, ds) * l;
		vec3 normal = (near - center);

		return Sphere(true, near, far, normal);
	}
	else
	{
		return Sphere(false, vec3(0), vec3(0), vec3(0));
	}
}

float depthCueing(vec3 pos) 
{
	vec4 clip_space_pos = modelViewProjectionMatrix * vec4(pos, 1.0);
	// float ndc_depth = clip_space_pos.z / clip_space_pos.w;
	float depth = clip_space_pos.w;
	// return clamp(depth - 10, 0,1); //depth * depth;
	return 0; //depth * depth;
}

float calcDepth(vec3 pos) 
{
	float far = gl_DepthRange.far; 
	float near = gl_DepthRange.near;
	vec4 clip_space_pos = modelViewProjectionMatrix * vec4(pos, 1.0);
	float ndc_depth = clip_space_pos.z / clip_space_pos.w;
	return 0.5 * (((far - near) * ndc_depth) + near + far);
}


void main()
{
	vec4 fragCoord = gFragmentPosition;
	fragCoord /= fragCoord.w;
	
	vec4 near = inverseModelViewProjectionMatrix*vec4(fragCoord.xy,-1.0,1.0);
	near /= near.w;

	vec4 far = inverseModelViewProjectionMatrix*vec4(fragCoord.xy,1.0,1.0);
	far /= far.w;

	vec3 V = normalize(far.xyz-near.xyz);	
	Sphere sphere = calcSphereIntersection(gSphereRadius, near.xyz, gSpherePosition, V);
	
	if (!sphere.hit)
		discard;

	float depth = calcDepth(sphere.near.xyz);
    if (selected) {
        depth = calcDepth(gSpherePosition.xyz);
    }
	fragPosition = vec4(sphere.near.xyz,length(sphere.near.xyz-near.xyz));
	fragNormal = vec4(sphere.normal,0.0);

	gl_FragDepth = depth;

    vec3 normal_vspace = normalize(normalMatrix * sphere.normal);
    vec3 lightDir_vspace = normalize(vec3(1,1,1));

    vec3 color = dot(normal_vspace, lightDir_vspace) * gSphereColor; // + vec4(0.2, 0.2, 0.1, 0.);
	if (selected) {
        color = vec3(1.,0.,1.);
    }
	out_color = vec4(mix(color, vec3(0.1), depthCueing(sphere.near.xyz)), 1);
}
