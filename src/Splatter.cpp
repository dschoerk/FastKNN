#include "Splatter.h"

#include <iostream>
#include "glbase.hpp"

Splatter::Splatter(const std::string& resourcePath)
{
  // Splatting Shaders
  Candy::GL::GLShader splatDepthVS((resourcePath + std::string("/shaders/splatDepth.vert")).c_str(), Candy::GL::VERTEX_SHADER);
  Candy::GL::GLShader splatGS((resourcePath + std::string("/shaders/splatDepth.geom")).c_str(), Candy::GL::GEOMETRY_SHADER);
  Candy::GL::GLShader splatDepthFS((resourcePath + std::string("/shaders/splatDepth.frag")).c_str(), Candy::GL::FRAGMENT_SHADER);
  Candy::GL::GLShader splatAccumVS((resourcePath + std::string("/shaders/splatAccum.vert")).c_str(), Candy::GL::VERTEX_SHADER);
  Candy::GL::GLShader splatAccumFS((resourcePath + std::string("/shaders/splatAccum.frag")).c_str(), Candy::GL::FRAGMENT_SHADER);
  Candy::GL::GLShader splatFinalVS((resourcePath + std::string("/shaders/splatFinal.vert")).c_str(), Candy::GL::VERTEX_SHADER);
  Candy::GL::GLShader splatFinalFS((resourcePath + std::string("/shaders/splatFinal.frag")).c_str(), Candy::GL::FRAGMENT_SHADER);

  splatDepthProg.attachShader(splatDepthVS);
  splatDepthProg.attachShader(splatGS);
  splatDepthProg.attachShader(splatDepthFS);
  splatDepthProg.link();

  splatAccumProg.attachShader(splatAccumVS);
  splatAccumProg.attachShader(splatGS);
  splatAccumProg.attachShader(splatAccumFS);
  splatAccumProg.link();

  splatFinalProg.attachShader(splatFinalVS);
  splatFinalProg.attachShader(splatFinalFS);
  splatFinalProg.link();

  // vbo for fullscreen quad
  GLfloat vertices[] = { -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, 1.f };
  glGenBuffers(1, &quadVbo);
  glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);



  // Surface Splatting Screen FBO and RBO
  glGenFramebuffers(1, &mFboScreenSS);
  glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenSS);
  glGenRenderbuffers(1, &mRboScreenSS);
  glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenSS);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1, 1);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRboScreenSS);

  glGenTextures(1, &depth);
  glGenTextures(1, &accumNormalDepth);
  glGenTextures(1, &accumColorWeight);

  attachTexture(0, depth);
  attachTexture(1, accumNormalDepth);
  attachTexture(2, accumColorWeight);

  GLuint fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    std::cout << "Error creating FboScreenSS! (" << fboStatus << ")" << std::endl;

  glBindTexture(GL_TEXTURE_2D, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);

  // resize();
  glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenSS);		
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
  
  glBindTexture(GL_TEXTURE_2D, depth);				
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
  glBindTexture(GL_TEXTURE_2D, accumNormalDepth);	
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
  glBindTexture(GL_TEXTURE_2D, accumColorWeight);	
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
}

void Splatter::draw(unsigned numPoints, float projection[16], float iprojection[16], GLuint pointBuffer, GLuint colorBuffer, GLuint normalBuffer, GLuint splatAxisBuffer, float splatRadiusFactor, bool hasColor, bool isShaded)
{
  glViewport(0, 0, 1024, 1024);
  glClearColor(0, 0, 0, 0);
  glClearDepth(1.0f);

  // bind surface splatting screen FBO
  glBindFramebuffer(GL_FRAMEBUFFER, this->mFboScreenSS);
//  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glEnableVertexAttribArray(0);	glBindBuffer(GL_ARRAY_BUFFER, pointBuffer);	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
  glEnableVertexAttribArray(1);	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);	glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, NULL);
  glEnableVertexAttribArray(2);	glBindBuffer(GL_ARRAY_BUFFER, splatAxisBuffer);	glVertexAttribPointer(2, 4, GL_FLOAT, false, 0, NULL);
  glEnableVertexAttribArray(3);	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);	glVertexAttribPointer(3, 3, GL_FLOAT, false, 0, NULL);

  // 1. Depth Pass
  //-----------------------------------------------------------------------------
 
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  splatDepthProg.bind();
  splatDepthProg.setUniform("projMatrix", projection, true);
  splatDepthProg.setUniform("radiusFactor", splatRadiusFactor);
  glDrawArrays(GL_POINTS, 0, numPoints);
//  return;

  // 2. Accumulation Pass
  //-----------------------------------------------------------------------------

//  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
  glDrawBuffers(2, colorBuffers);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);

  splatAccumProg.bind();
  splatAccumProg.setTexture("texDepth", 0, depth);
  splatAccumProg.setUniform("projMatrix", projection, true);
  splatAccumProg.setUniform("radiusFactor", splatRadiusFactor);
  splatAccumProg.setUniform("hasColor", hasColor);
  glDrawArrays(GL_POINTS, 0, numPoints);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(3);

  glDisable(GL_BLEND);
//  return;

  // 3. Final rendering pass
  //-----------------------------------------------------------------------------
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  splatFinalProg.bind();
  splatFinalProg.setTexture("texAccumNormalDepth", 0, accumNormalDepth);
  splatFinalProg.setTexture("texAccumColorWeight", 1, accumColorWeight);
  splatFinalProg.setUniform("projMatrixI", iprojection, false);
  splatFinalProg.setUniform("near", 1.5f);
  splatFinalProg.setUniform("far", 1000.0f);
  splatFinalProg.setUniform("shaded", isShaded);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisableVertexAttribArray(0);

  //-----------------------------------------------------------------------------
  glUseProgram(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
