

#include "GLShader.h"

std::string Candy::GL::readFile(const char *filePath)
{
  std::string content;
  std::ifstream fileStream(filePath, std::ios::in);

  if (!fileStream.is_open()) {
    std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
    return "";
  }

  std::string line = "";
  while (!fileStream.eof()) {
    std::getline(fileStream, line);
    content.append(line + "\n");
  }

  fileStream.close();
  return content;
}

Candy::GL::GLShader::GLShader(const char* filename, GLShaderType _type) : type(_type)
{
  handle = glCreateShader(type);
  std::string source = readFile(filename);
  const char* c_ptr = source.c_str();
  int length = static_cast<int>(source.length());
  glShaderSource(handle, 1, &c_ptr, &length);

  glCompileShader(handle);

  // Check the compilation status and report any errors
  GLint shaderStatus = GL_TRUE;
  glGetShaderiv(handle, GL_COMPILE_STATUS, &shaderStatus);

  // If the shader failed to compile, display the info log and quit out
  if (shaderStatus == GL_FALSE)
  {
    GLint infoLogLength = 0;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &infoLogLength);

    GLchar *strInfoLog = new GLchar[infoLogLength + 1];
    glGetShaderInfoLog(handle, infoLogLength, NULL, strInfoLog);

    std::cout << " shader compilation failed: " << strInfoLog << " filename: " << filename << std::endl;
    delete[] strInfoLog;
  }
#ifdef _DEBUG
  else
  {
    std::cout << " shader compilation OK" << std::endl;
  }
#endif
}

Candy::GL::GLShader::~GLShader()
{
  glDeleteShader(handle); 
}