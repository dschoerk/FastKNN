#version 330

in vec4 tPosinvA2invB2;
in vec3 vPos;
in vec3 normal;
in vec3 col;

layout(location = 0) out vec3 depth;

void main()
{
	mat2 invM = mat2(tPosinvA2invB2.z, 0, 0, tPosinvA2invB2.w);
	float mahalanobisDist = dot(tPosinvA2invB2.xy, invM * tPosinvA2invB2.xy);
	if (mahalanobisDist > 1.0)
		discard;

	depth.x = -vPos.z;
	//depth = col;
}
/*
#version 330

in vec4 tPosinvA2invB2;
in vec3 vPos;
in vec3 normal;

layout(location = 0) out vec3 col;

void main()
{
	mat2 invM = mat2(tPosinvA2invB2.z, 0, 0, tPosinvA2invB2.w);
	float mahalanobisDist = dot(tPosinvA2invB2.xy, invM * tPosinvA2invB2.xy);
	if (mahalanobisDist > 1.0)
		discard;

	col = normal;
}*/