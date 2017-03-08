#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec4 splatAxes;
layout(location = 3) in vec3 inColor;

uniform float radiusFactor = 1;
uniform bool hasColor = false;

// just needed for per-vertex-culling
uniform sampler2D texDepth;
uniform mat4 projMatrix;


out vec3 vNormal;
out vec4 ellipse;
out vec3 color;
out float visible;


void main()
{
	vec3 vPos = position.xyz;
	vNormal = normal.xyz;

	// flip normal towards viewer
	if (dot(vPos, vNormal) > 0)
		vNormal = -vNormal;

	ellipse.xyz = splatAxes.xyz;
	ellipse.xyz *= radiusFactor;
	ellipse.w = splatAxes.w;
	//----------------------------------

	visible = 1;
	if (false)			// Enable for per-vertex culling
	{
		vec4 clipPos = projMatrix * vec4(vPos, 1);
		vec2 uvPos = (clipPos.xy / clipPos.w) * 0.5 + 0.5;
		float visibleDepth = texture(texDepth, uvPos).x;
		if (-vPos.z > visibleDepth)
			visible = 0;
	}
	//----------------------------------

	color = vec3(0.3, 0.4, 0.6);
	
	if(hasColor)
		color = inColor;
	gl_Position = vec4(vPos, 1);
}

