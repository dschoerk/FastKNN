#pragma once

#ifdef _WIN32
  #include <GL/glew.h>
#endif
#include "GLShader.h"

#include <vector>
#include <map>

namespace Candy
{
  namespace GL
  {
    class GLProgram
    {
    public:
      explicit GLProgram();
      void attachShader(const GLShader& shader);
      void link();
      void bind();
      int getNumberOfAttributes();
      int getNumberOfUniforms();
      int getAttributeByName(char const* name);
      int getUniformByName(char const* name);
      
      // uniform setters
      void setUniform(const char* name, float u);
	    void setUniform(const char* name, int u);
      void setUniform(const char* name, bool u);
      void setUniform(const char* name, const float u[16], bool transpose = false);
      void setTexture(const char* name, GLuint texUnit, GLuint tex);

    private:
      GLuint handle;
      std::map<std::string, GLuint> uniformCache;
    };
  }
}