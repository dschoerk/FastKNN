#version 330

uniform sampler2D texDepth;
uniform float sigmaFactor = 3.0;

in vec4 tPosinvA2invB2;
in vec3 vPos;
in vec3 normal;
in vec3 col;

layout(location = 0) out vec4 outNormalDepth;
layout(location = 1) out vec4 outColorCount;

void main()
{
	mat2 invM = mat2(tPosinvA2invB2.z, 0, 0, tPosinvA2invB2.w);
	float mahalanobisDist = dot(tPosinvA2invB2.xy, invM * tPosinvA2invB2.xy);
	if (mahalanobisDist > 1.0)
		discard;

	float depth = -vPos.z;
	float visibleDepth = texelFetch(texDepth, ivec2(gl_FragCoord.xy), 0).x;

	// Per-Fragment Culling
	if (visibleDepth > 0 && depth > visibleDepth)
		discard;

	float x = mahalanobisDist * sigmaFactor;
	float weight = exp(-x * x * 0.5);		// - exp( - sigmaFactor*sigmaFactor * 0.5)	-> drags bell to floor s.t. border == 0

	outNormalDepth = vec4(normal, depth) * weight;
	outColorCount = vec4(col, 1) * weight;
}

