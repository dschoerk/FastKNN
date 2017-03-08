#pragma once

#include <string>
#include <iostream>
#include <fstream>

#include <GL/glew.h>

namespace Candy 
{
  namespace GL 
  {
    enum GLShaderType
    {
      VERTEX_SHADER = GL_VERTEX_SHADER,
      FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
      GEOMETRY_SHADER = GL_GEOMETRY_SHADER,
      TESS_EVALUATION_SHADER = GL_TESS_EVALUATION_SHADER,
      TESS_CONTROL_SHADER = GL_TESS_CONTROL_SHADER
    };

    class GLShader
    {
    friend class GLProgram;
    public:
      explicit GLShader(const char* filename, GLShaderType _type);
      ~GLShader();

    private:
      GLShaderType type;
      GLuint handle;
    };

    std::string readFile(const char *filePath);
  }
}