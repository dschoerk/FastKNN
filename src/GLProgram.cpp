#include "GLProgram.h"

Candy::GL::GLProgram::GLProgram()
{
  handle = glCreateProgram();
}

void Candy::GL::GLProgram::attachShader(GLShader const& shader)
{
  glAttachShader(handle, shader.handle);
}

void Candy::GL::GLProgram::link()
{
  
  glLinkProgram(handle);

  // Check the compilation status and report any errors
  GLint shaderStatus = GL_TRUE;
  glGetShaderiv(handle, GL_LINK_STATUS, &shaderStatus);

  // If the shader failed to compile, display the info log and quit out
  if (shaderStatus == GL_FALSE)
  {
    GLint infoLogLength = 0;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &infoLogLength);
    std::cout << "fail " << infoLogLength << std::endl;

    if (infoLogLength > 0)
    {
      GLchar *strInfoLog = new GLchar[infoLogLength];
      glGetShaderInfoLog(handle, infoLogLength, NULL, strInfoLog);

      std::cout << " shader linking failed: " << std::string(strInfoLog, infoLogLength) << std::endl;
      delete[] strInfoLog;
    }
    else
      std::cout << " shader linking failed: no message" << std::endl;
  }
#ifdef _DEBUG
  else
  {
    std::cout << " shader linking OK" << std::endl;
  }
#endif


}

void Candy::GL::GLProgram::bind()
{
  glUseProgram(handle);
}

int Candy::GL::GLProgram::getNumberOfAttributes()
{
  GLint param;
  glGetProgramInterfaceiv(handle, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &param);
  return param;
}

int Candy::GL::GLProgram::getNumberOfUniforms()
{
  GLint param;
  glGetProgramInterfaceiv(handle, GL_UNIFORM, GL_ACTIVE_RESOURCES, &param);
  return param;
}

int Candy::GL::GLProgram::getAttributeByName(char const* name)
{
  return glGetProgramResourceIndex(handle, GL_PROGRAM_INPUT, name);
}

int Candy::GL::GLProgram::getUniformByName(char const* name)
{
  int loc;
  if (uniformCache.find(name) == uniformCache.end())
  {
    loc = glGetUniformLocation(handle, name);

    if (loc == -1)
      std::cout << "uniform " << name << " not found" << std::endl;

#ifdef _DEBUG
    std::cout << "get Uniform " << name << std::endl;
#endif
    uniformCache[name] = loc;
  }
  else
  {
    loc = uniformCache[name];
  }

  return loc;
}


void Candy::GL::GLProgram::setUniform(char const* name, float u)
{
  int loc = getUniformByName(name);
  //assert(loc > -1);
  glUniform1f(loc, u);
}

void Candy::GL::GLProgram::setUniform(char const* name, int u)
{
  int loc = getUniformByName(name);
  //assert(loc > -1);
  glUniform1i(loc, u);
}

void Candy::GL::GLProgram::setUniform(char const* name, bool u)
{
  int loc = getUniformByName(name);
  glUniform1i(loc, u);
}

void Candy::GL::GLProgram::setUniform(const char* name, const float u[16], bool transpose)
{
  int loc = getUniformByName(name);
  glUniformMatrix4fv(loc, 1, transpose, u);
}

void Candy::GL::GLProgram::setTexture(const char* name, GLuint texUnit, GLuint tex)
{
  int loc = getUniformByName(name);
  glActiveTexture(GL_TEXTURE0 + texUnit);
  glBindTexture(GL_TEXTURE_2D, tex);
  glUniform1i(loc, texUnit);

}