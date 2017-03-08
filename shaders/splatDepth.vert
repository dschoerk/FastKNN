#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec4 _splatAxes;
layout(location = 3) in vec3 color;

uniform mat4 mvMatrix;
uniform mat4 normalMatrix;
uniform float radiusFactor = 1;

out vec3 vNormal;
out vec4 ellipse;
out float visible;
out vec3 col;

// TEMP: 
//out float id;


void main()
{
	col = color;
	vec3 vPos = position.xyz;	
	vNormal = normal.xyz;

	// flip normal towards viewer
	if (dot(vPos, vNormal) > 0)
		vNormal = -vNormal;

	vec4 splatAxes = _splatAxes;
	ellipse.xyz = splatAxes.xyz;
	ellipse.xyz *= radiusFactor;
	ellipse.w = splatAxes.w;
	//----------------------------------


	// perform a shift along view ray (according to EWA Paper) to reduce silhouette artifacts
	float radius = length(ellipse.xyz);
	float fac = (vPos.z - radius) / vPos.z;
	vPos *= fac;				// Step 1) Shift Splat along view ray to overcome a depth difference of z_t
	ellipse.xyz *= fac;			// Step 2) Increase Splat size

	//-----------------------------------------------------
	gl_Position = vec4(vPos, 1);
	visible = 1.0;


	//TEMP: 
	//id = gl_VertexID;
}

