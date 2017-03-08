#version 330

uniform sampler2D texAccumNormalDepth;
uniform sampler2D texAccumColorWeight;

uniform mat4 projMatrixI;
uniform float near;
uniform float far;
uniform bool shaded;

// Lighting Parameters
uniform vec3 kads = vec3(0.2, 0.7, 0.4);
uniform float shininess = 100.0;

in vec2 texCoord;
out vec4 outColor;


vec3 restoreViewSpacePos(vec2 positionNDC, mat4 projMatrixI, float viewSpaceDepth)
{
	vec4 pNear = projMatrixI * vec4(positionNDC, -1, 1);
	vec4 pFar = projMatrixI * vec4(positionNDC, 1, 1);
	pNear /= pNear.w;
	pFar /= pFar.w;
	float t = (-viewSpaceDepth - pNear.z) / (pFar.z - pNear.z);
	return ((1 - t) * pNear + t * pFar).xyz;
}


void main()
{
	vec4 accumNormalDepth = texture(texAccumNormalDepth, texCoord);
	if (accumNormalDepth.w == 0)
		discard;

	vec4 accumColorCount = texture(texAccumColorWeight, texCoord);
	float invCount = 1.0 / accumColorCount.w;

	// normalize
	vec3  vNormal = normalize(accumNormalDepth.xyz * invCount);
	float vDepth = accumNormalDepth.w * invCount;
	vec3  vPos = restoreViewSpacePos(texCoord * 2 - 1, projMatrixI, vDepth);
	vec3 color = accumColorCount.xyz * invCount;


	// illumination
	//--------------------------------------------------------------------------------------------
	vec3 L = normalize(vec3(0, 0.4, 1));
	vec3 V = normalize(-vPos);
	vec3 H = normalize(L + V);
	vec3 ambient = kads.x * color;
	vec3 diffuse = kads.y * color * max(dot(vNormal, L), 0.0);
	vec3 specular = kads.z * color * pow(max(dot(vNormal, H), 0.0), shininess);

	outColor.xyz = ambient + diffuse + specular;
	//outColor.xyz = vec3(kads.x + kads.y * max(dot(vNormal, L), 0.0) + kads.z * pow(max(dot(vNormal, H), 0.0), shininess));
	
	if(!shaded)
		outColor.xyz = color;
	//outColor.xyz = accumColorCount.xyz;
	
	// Adjust Depth
	gl_FragDepth = ((far + near) / (far - near) + 2 * far*near / (-vDepth * (far - near))) * 0.5 + 0.5;
}