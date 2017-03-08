
#pragma once
#include <GL/glew.h>
#include <GLProgram.h>
#include <string>

class Splatter
{
public:
  Splatter(const std::string& resourcePath);
  void draw(unsigned numPoints, float projection[16], float iprojection[16], GLuint pointBuffer, GLuint colorBuffer, GLuint normalBuffer, GLuint splatAxisBuffer, float splatRadiusFactor, bool hasColor = false, bool isShaded = true);

private:
  Candy::GL::GLProgram splatDepthProg;
  Candy::GL::GLProgram splatAccumProg;
  Candy::GL::GLProgram splatFinalProg;

  GLuint quadVbo;
  GLuint mFboScreenSS;
  GLuint mRboScreenSS;

  GLuint depth;
  GLuint accumNormalDepth;
  GLuint accumColorWeight;
};
