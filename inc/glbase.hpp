//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// (c) Reinhold Preiner 2014
//-----------------------------------------------------------------------------

#pragma once


// GL graphics includes
#ifdef _WIN32
  #include <GL/glew.h>
  #include <GL/wglew.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


//#define SOURCE( src )	#src
#define SOURCE( version, src )	"#version " #version "\n" #src

typedef unsigned int uint;


#define GLBASE_PRINT_LOG	0


#define CHECK_GL_ERROR(LABEL) { int e = glGetError(); if (e) cerr << "GL-Error @ " << LABEL << ": " << e << endl; }


inline void drawFullscreenQuad()
{
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);	glVertex2f(-1, -1);
	glTexCoord2f(1, 0);	glVertex2f(1, -1);
	glTexCoord2f(1, 1);	glVertex2f(1, 1);
	glTexCoord2f(0, 1);	glVertex2f(-1, 1);
	glEnd();
}


inline void showTexture(uint tex, uint textureProgram)
{
	glDisable(GL_DEPTH_TEST);
	glBindTexture(GL_TEXTURE_2D, tex);

	glUseProgram(textureProgram);
	drawFullscreenQuad();
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
}

inline uint createProgram(const char* vsSource, const char* gsSource, const char* fsSource, const char* programName = "")
{
#if GLBASE_PRINT_LOG
	if (programName) std::cout << "Creating " << programName << std::endl;
#endif

	const int bufSize = 1024;
	int success = 0;
	char info[bufSize] = { 0 };

	uint program = glCreateProgram();
	uint vs = glCreateShader(GL_VERTEX_SHADER);
	uint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vs, 1, &vsSource, 0);
	glShaderSource(fs, 1, &fsSource, 0);

	// Compile VS
	glCompileShader(vs);
	glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vs, bufSize, 0, info);
		if (*info)
			std::cerr << "VS error: " << info << std::endl;
		glDeleteShader(vs);
		return 0;
	}
	glAttachShader(program, vs);


	// Compile GS if given
	if (gsSource)
	{
		uint gs = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(gs, 1, &gsSource, 0);
		glCompileShader(gs);
		glGetShaderiv(gs, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(gs, bufSize, 0, info);
			if (*info)
				std::cerr << "GS error: " << info << std::endl;
			glDeleteShader(vs);
			return 0;
		}
		glAttachShader(program, gs);
	}

	// Compile FS if given
	if (fsSource)
	{
		glCompileShader(fs);
		glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fs, bufSize, 0, info);
			if (*info)
				std::cerr << "FS error: " << info << std::endl;
			glDeleteShader(vs);
			glDeleteShader(fs);
			return 0;
		}
		glAttachShader(program, fs);
	}


	// Linking
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		glGetProgramInfoLog(program, bufSize, 0, info);
		if (*info)
			std::cerr << info << std::endl;

		glDeleteProgram(program);
		glDeleteShader(vs);
		glDeleteShader(fs);
		program = 0;
	}

	return program;
}


inline uint createProgram(int vs, int gs, int fs, const char* programName = "")
{
#if GLBASE_PRINT_LOG
	if (programName) std::cout << "Linking " << programName << std::endl;
#endif
	const int bufSize = 1024;
	int success = 0;
	char info[bufSize] = { 0 };

	uint program = glCreateProgram();

	glAttachShader(program, vs);
	if (gs) glAttachShader(program, gs);
	if (fs) glAttachShader(program, fs);

	// Linking
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	glGetProgramInfoLog(program, bufSize, 0, info);
	if (!success)
	{
		if (*info)
		{
			std::cerr << "Linking " << programName << ": " << std::endl;
			std::cerr << info << std::endl;
		}

		glDeleteProgram(program);
		return 0;
	}
	return program;
}


inline uint createShader(int shaderType, uint count, const char** source, const char* name = "")
{
	// Debug Output
	std::string typeStr;
	switch (shaderType)
	{
	case GL_VERTEX_SHADER: typeStr = "VS"; break;
	case GL_GEOMETRY_SHADER: typeStr = "GS"; break;
	case GL_FRAGMENT_SHADER: typeStr = "FS"; break;
	}
	
	// Compile
	const int bufSize = 1024;
	int success = 0;
	char info[bufSize] = { 0 };

	uint shader = glCreateShader(shaderType);
	glShaderSource(shader, count, source, 0);

	// Compile FS
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, bufSize, 0, info);
		if (*info)
		{
			std::cerr << "Compiling " << typeStr << " " << name << ": " << std::endl;
			std::cerr << info << std::endl;
		}		
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}


inline uint loadShader(int shaderType, const char* filename, const char* prefix = "")
{
	// Load File
	std::ifstream file(filename);
	if (file.fail())
	{
		std::cerr << "Couldn't load file " << filename << std::endl;
		return 0;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string str = buffer.str();

	const char* source[2] = { prefix, str.c_str() };
	return createShader(shaderType, 2, source, filename);
}


inline uint loadShader(int shaderType, const std::string& filename, const char* prefix = "")
{
	return loadShader(shaderType, filename.c_str(), prefix);
}


// loads a shader of a given filename and appends it to list
inline uint loadShader(std::vector<uint>& list, int shaderType, const std::string& filename, const char* prefix = "")
{
	uint shader = loadShader(shaderType, filename.c_str(), prefix);
	list.push_back(shader);
	return shader;
}


inline void attachTexture(int channel, uint tex)
{
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + channel, GL_TEXTURE_2D, tex, 0);
}


inline void setDrawBuffer(uint channel)
{
	glDrawBuffer(GL_COLOR_ATTACHMENT0 + channel);
}
inline void setDrawBuffers(uint channel0, uint channel1)
{
	GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT0 + channel0, GL_COLOR_ATTACHMENT0 + channel1 };
	glDrawBuffers(2, colorBuffers);
}
inline void setDrawBuffers(uint channel0, uint channel1, uint channel2)
{
	GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT0 + channel0, GL_COLOR_ATTACHMENT0 + channel1, GL_COLOR_ATTACHMENT0 + channel2 };
	glDrawBuffers(3, colorBuffers);
}
inline void setDrawBuffers(uint channel0, uint channel1, uint channel2, uint channel3)
{
	GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT0 + channel0, GL_COLOR_ATTACHMENT0 + channel1, GL_COLOR_ATTACHMENT0 + channel2, GL_COLOR_ATTACHMENT0 + channel3 };
	glDrawBuffers(4, colorBuffers);
}


inline void bindUniformSampler(uint program, uint texUnit, const char* uniformName, uint tex)
{
	glActiveTexture(GL_TEXTURE0 + texUnit);
	glBindTexture(GL_TEXTURE_2D, tex);
	glUniform1i(glGetUniformLocation(program, uniformName), texUnit);
}

inline void bindUniformImage(uint program, uint imageUnit, const char* uniformName, uint tex, GLenum access, GLenum format)
{
	glBindImageTexture(imageUnit, tex, 0, false, 0, access, format);
	glUniform1i(glGetUniformLocation(program, uniformName), imageUnit);
}


#define PROFILING


#if FALSE
inline double getTime()
{
	return glutGet(GLUT_ELAPSED_TIME);
}
#else
inline double getTime()
{
/*#ifdef _WIN32
	static double msPerTick = 0;
	if (msPerTick == 0)
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		msPerTick = 1000.0 / double(frequency.QuadPart);
	}

	LARGE_INTEGER currentTime;
	QueryPerformanceCounter(&currentTime);
	return double(currentTime.QuadPart) * msPerTick;
#else
	return 0;
#endif*/

  glFinish();
  GLint64 tq_time;
  glGetInteger64v(GL_TIMESTAMP, &tq_time);
  return tq_time / 1000000.0;
}
#endif



static double g_timingStartTime;
inline void startTimer()
{
#if defined PROFILING
	glFinish();
	g_timingStartTime = getTime();
#endif
}

inline double stopTimer()
{
#if defined PROFILING
	glFinish();
	double stopTime = getTime();
	return stopTime - g_timingStartTime;
#else
	return 0;
#endif
}

