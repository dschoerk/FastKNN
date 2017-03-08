#version 330

uniform mat4 mvMatrix;
uniform mat4 mvpMatrix;
uniform mat4 modelMatrix = mat4(1);
uniform mat3 normalMatrix;

layout(location=0) in vec3 modelspacePosition;
layout(location=1) in vec3 modelspaceNormal; // correct normals for error computation
layout(location=2) in vec3 modelspaceColor;

out vec4 viewspacePosition;
out vec3 viewspaceNormal; // correct normals for error computation
out vec3 viewspaceColor;

void main()
{
	vec4 p = modelMatrix * vec4( modelspacePosition, 1 );
	gl_Position = mvpMatrix * modelMatrix * p;
	viewspacePosition = mvMatrix * modelMatrix * p;	// the 1 in the w-channel indicates presence of a point
	
	viewspaceNormal = vec3(transpose(inverse(mvMatrix)) * vec4(modelspaceNormal, 0));
	//if (dot(vec3(viewspacePosition), viewspaceNormal) > 0)
	//	viewspaceNormal = -viewspaceNormal;
	
	viewspaceColor = modelspaceColor;
}
