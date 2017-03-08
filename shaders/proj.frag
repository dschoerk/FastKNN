#version 330

in vec4 viewspacePosition;
in vec3 viewspaceNormal; // correct normals for error computation
in vec3 viewspaceColor;

layout(location=0) out vec4 vpos;
layout(location=1) out vec4 vnormal;
layout(location=2) out vec4 vcolor;

void main()
{
	vpos = viewspacePosition;
	vnormal = vec4(viewspaceNormal, 0);
	vcolor = vec4(viewspaceColor, 1);
}

