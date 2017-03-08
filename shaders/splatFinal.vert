#version 330

layout(location = 0) in vec2 uvPos;
out vec2 texCoord;

void main()
{
	gl_Position = vec4(2 * uvPos - 1, 0, 1);
	texCoord = uvPos;
}

