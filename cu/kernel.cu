
#include <GL/glew.h>
#ifdef _WIN32
  #include <GL/wglew.h>
#endif

#include "cmake_conf.h"

#include <GLFW/glfw3.h>
#include <AntTweakBar.h>
#include "GLProgram.h"
#include "autosplats.h"
#include "glbase.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>
#include <cutil_math.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "kernel.h"
#include "Splatter.h"
#include "PclKnn.h"

#define TIMING

    inline void TwEventMouseButtonGLFW3(GLFWwindow* window, int button, int action, int mods){ TwEventMouseButtonGLFW(button, action); }
    inline void TwEventMousePosGLFW3(GLFWwindow* window, double xpos, double ypos){ TwMouseMotion(int(xpos), int(ypos)); }
    inline void TwEventMouseWheelGLFW3(GLFWwindow* window, double xoffset, double yoffset){ TwEventMouseWheelGLFW((int)yoffset); }
    inline void TwEventKeyGLFW3(GLFWwindow* window, int key, int scancode, int action, int mods){ TwEventKeyGLFW(key, action); }
    inline void TwEventCharGLFW3(GLFWwindow* window, int codepoint){ TwEventCharGLFW(codepoint, GLFW_PRESS); }

// preprocessor definitions - compiletime options
#define USE_REGISTERS
#define SORTING

float glGetTime()
{
  glFinish();
  GLint64 tq_time;
  glGetInteger64v(GL_TIMESTAMP, &tq_time);
  return (float)(tq_time / 1000000.0);
}

static void error_callback(int error, const char* description)
{
  fputs(description, stdout);
}

bool keyPlus;
bool keyMinus;
bool keySpace;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  
  keyMinus = key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS;
  keyPlus = key == GLFW_KEY_KP_ADD && action == GLFW_PRESS;
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    keySpace = !keySpace;
}

static void windowsize_callback(GLFWwindow* window, int x, int y)
{
  TwWindowSize((int)x, (int)y);
}

float distanceToViewer = 300.0f;
static void mwheel_callback(GLFWwindow *, double x, double y)
{
  distanceToViewer *= 1.f + 0.02f * (float)y;
}

static void cursor_callback(GLFWwindow *, double x, double y)
{
  TwMouseMotion((int)x, (int)y);
}

typedef enum { Bunny, Armadillo, Dragon, Buddah, Gnome } ShownModel;
ShownModel shownModel = Bunny;
void TW_CALL modelSetCallback(const void *value, void *clientData)
{
  shownModel = *(ShownModel*)value;
}

void TW_CALL modelGetCallback(void *value, void *clientData)
{
  *(ShownModel*)value = shownModel;  // for instance
}

gms::mat4 perspective(float n, float f, float right, float top)
{
  float f_n = f - n;
  gms::mat4 proj(
    n / right, 0, 0, 0,
    0, n / top, 0, 0,
    0, 0, -(f + n) / f_n, -2.0f*f*n / f_n,
    0, 0, -1.f, 0);

  return proj;
}

/*std::pair<float, float> computeNormalErrorVsGroundtruth(unsigned k, unsigned numPointsInScreenspace, GLuint pointBuffer, GLuint normalBuffer, GLuint splatAxisBuffer, float3* Hpoints, float3* Hnormals, float3* HcorrectNormals, float4* HsplatAxis, bool inject)
{
  glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
  glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hnormals);
  glBindBuffer(GL_ARRAY_BUFFER, pointBuffer);
  glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hpoints);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return computeNormalError(k, numPointsInScreenspace, Hpoints, Hnormals, HcorrectNormals, HsplatAxis, inject);
}*/

// globals
__device__ unsigned int* g_numBuffer;
__device__ unsigned int* g_packedQuadtreeIndices;
__device__ float3* g_packedQuadtree;
__device__ float3* g_kBuffer;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRefNormals;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRefColors;

// extern kernel functions
#include "helper.cuh"
#include "sort_str.cuh"
#include "quadtree_kernels.cuh"
#include "candidate_kernel.cuh"

__global__
  void set_globals(
    unsigned int* packedQuadtreeIndices, 
    unsigned int* numBuffer, 
    float3* packedQuadtree,
    float3* knearestBuffer)
{
  g_numBuffer = numBuffer;
  g_packedQuadtree = packedQuadtree;
  g_packedQuadtreeIndices = packedQuadtreeIndices;
  g_kBuffer = knearestBuffer;
}

extern "C"
LoggingData transform_points(TransformationSettings settings, const std::vector<float3>& points, const std::vector<float3>& normals, const std::vector<float3>& colors, const Math::matrix4f& view_mat, const Math::matrix4f& proj_mat)
{
	printDeviceInfo();

  std::string resourcePath = conf_SOURCE_DIR;
	
  gms::mat4 projection = gms::mat4(proj_mat.data);
  projection.transpose();

  /// define some constants
//  const unsigned int num_byte = sizeof(float3)*points.size();
  const unsigned int resolution = settings.screenbuffer_size;
  const unsigned int depthbuffer_size = resolution * resolution;
  const unsigned int quadtree_size = settings.quadtree_size();
  const unsigned int THREADNUM = 32;

  /// only for debugging purposes
  float3* HpackedQuadtree = new float3[settings.screenbuffer_size*settings.screenbuffer_size];
  float3* HcorrectNormals = new float3[settings.screenbuffer_size*settings.screenbuffer_size];
  unsigned* HknearestIndex = new unsigned[settings.screenbuffer_size*settings.screenbuffer_size*settings.knn_kmax];
  unsigned* HpackedCoordinates = new unsigned[settings.screenbuffer_size*settings.screenbuffer_size];
  float3* Hnormals = new float3[settings.screenbuffer_size*settings.screenbuffer_size];
  float3* Hcolors = new float3[settings.screenbuffer_size*settings.screenbuffer_size];
//  float3* Hnormals2 = new float3[settings.screenbuffer_size*settings.screenbuffer_size];
  float4* HsplatAxis = new float4[settings.screenbuffer_size*settings.screenbuffer_size];
  
  // Autosplats Parameters
  gms::AutoSplats::Params asParam;
  asParam.K = 8;
  asParam.maxIters = 3;
  asParam.splatRadiusFactor = 1;

  GLFWwindow* window;

  if (!glfwInit())
  {
    std::cout << "GLFW Initialization failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  glfwSetErrorCallback(error_callback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  
  window = glfwCreateWindow(1024, 1024, "FastKnn", NULL, NULL);
  if (!window)
  {
    std::cout << "failed to create GLFW Window" << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  //glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_FALSE);

  glewExperimental = GL_TRUE; // why?
  GLenum err = glewInit();
  if (GLEW_OK != err)
  {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  glfwSetMouseButtonCallback(window, (GLFWmousebuttonfun)TwEventMouseButtonGLFW3);
  //glfwSetCursorPosCallback(window, (GLFWcursorposfun)TwEventMousePosGLFW3);
  glfwSetCursorPosCallback(window, cursor_callback);
  glfwSetScrollCallback(window, (GLFWscrollfun)TwEventMouseWheelGLFW3);
  glfwSetKeyCallback(window, (GLFWkeyfun)TwEventKeyGLFW3);
  glfwSetCharCallback(window, (GLFWcharfun)TwEventCharGLFW3);
  glfwSetWindowSizeCallback(window, windowsize_callback);

  // Tweakbar
  TwInit(TW_OPENGL_CORE, NULL);
  TwWindowSize(1024, 1024);
  TwBar *bar;
  bar = TwNewBar("Settings");
  TwDefine(" Settings size='240 420' ");
  TwDefine(" Settings refresh=.2 ");
  TwDefine(" GLOBAL help='This example shows how to integrate AntTweakBar with GLFW and OpenGL.' ");

  float splatSizeFactor = 1;
  TwAddVarRW(bar, "SplatSize", TW_TYPE_FLOAT, &splatSizeFactor, " label='Splat Size Factor' min=0 max=5 step=0.05 keyIncr=s keyDecr=S help='Factor for Splatsize' ");
  TwAddVarRW(bar, "Radius Multiplier Initial", TW_TYPE_FLOAT, &settings.initial_radius_multiplier, "min=0 max=10 step=0.05");
  TwAddVarRW(bar, "Radius Multiplier Iterative", TW_TYPE_FLOAT, &settings.iterative_radius_multiplier, "min=0 max=10 step=0.05");
  
  // stats
  unsigned int numPointsInScreenspace = 0;
  TwAddVarRO(bar, "Screenspace Points", TW_TYPE_UINT32, &numPointsInScreenspace, "");
  TwAddSeparator(bar, NULL, " group='Timing' ");
  float fps = 0;
  TwAddVarRO(bar, "FPS", TW_TYPE_FLOAT, &fps, " group='Timing'");
  float theoretical_fps = 0;
  TwAddVarRO(bar, "theoretical FPS", TW_TYPE_FLOAT, &theoretical_fps, " group='Timing'");
  float project_time = 0;
  TwAddVarRO(bar, "Projection", TW_TYPE_FLOAT, &project_time," group='Timing'");
  float knn_time = 0;
  TwAddVarRO(bar, "Knn", TW_TYPE_FLOAT, &knn_time, " group='Timing'");
  float interopOverhead_time = 0;
  TwAddVarRO(bar, "Interop Overhead", TW_TYPE_FLOAT, &interopOverhead_time, " group='Timing'");
  float splatVbo_time = 0;
  TwAddVarRO(bar, "SplatVbo", TW_TYPE_FLOAT, &splatVbo_time, " group='Timing'");
  float splat_time = 0;
  TwAddVarRO(bar, "Splatting", TW_TYPE_FLOAT, &splat_time, " group='Timing'");
  float total_time = 0;
  TwAddVarRO(bar, "Total Time", TW_TYPE_FLOAT, &total_time, " group='Timing'");
  
  
  /*TwEnumVal enumModels[] = { { Bunny, "Bunny" }, { Armadillo, "Armadillo" }, { Dragon, "Dragon" }, { Buddah, "Buddah" }, { Gnome, "Gnome" } };
  TwType twShownModel = TwDefineEnum("Model", enumModels, 5);
  TwAddVarCB(bar, "Model", twShownModel, modelSetCallback, modelGetCallback, 0, 0);*/
//  TwAddVarRW(bar, "KnnMode", twShownModel, &shownModel, NULL);

  TwAddSeparator(bar, NULL, " group='Settings' ");
  typedef enum { FastKnn, Autosplats, Pcl} KnnMode;
  KnnMode knnMode = FastKnn;
  TwEnumVal enumKnnModes[] = { { FastKnn, "FastKnn" }, { Autosplats , "Autosplats"}, { Pcl, "PCL" } };
  TwType twKnnMode = TwDefineEnum("KnnMode", enumKnnModes, 3);
  TwAddVarRW(bar, "KnnMode", twKnnMode, &knnMode, " group='Settings'");
  unsigned int maxIterations = 3;
  TwAddVarRW(bar, "Iterations", TW_TYPE_UINT32, &maxIterations, " group='Settings'");
  TwAddVarRW(bar, "Kmin", TW_TYPE_UINT32, &settings.knn_kmin, " group='Settings'");
  int traverseLevel = 8;
  TwAddVarRW(bar, "traverseLevel", TW_TYPE_UINT32, &traverseLevel, " group='Settings'");

  TwAddSeparator(bar, NULL, " group='Error' ");
  float fastknn_kdtree_error_mean = 0;
  float fastknn_kdtree_error_var = 0;
  float autosplats_kdtree_error_mean = 0;
  float autosplats_kdtree_error_var = 0;
  TwAddVarRO(bar, "FastKnn Err Mean", TW_TYPE_FLOAT, &fastknn_kdtree_error_mean, "group='Error'");
  TwAddVarRO(bar, "FastKnn Err Std", TW_TYPE_FLOAT, &fastknn_kdtree_error_var, "group='Error'");
  TwAddVarRO(bar, "Autosplats Err Mean", TW_TYPE_FLOAT, &autosplats_kdtree_error_mean, "group='Error'");
  TwAddVarRO(bar, "Autosplats Err Std", TW_TYPE_FLOAT, &autosplats_kdtree_error_var, "group='Error'");
  char compute_error = 0;
  TwAddVarRW(bar, "Compute Error", TW_TYPE_BOOL8, &compute_error, "group='Error'");
  
  TwAddSeparator(bar, NULL, " group='Shading' ");
  char useColor = 0;
  TwAddVarRW(bar, "use Color", TW_TYPE_BOOL8, &useColor, "group='Shading'");
  char isShaded = 1;
  TwAddVarRW(bar, "is Shaded", TW_TYPE_BOOL8, &isShaded, "group='Shading'");

  /*typedef enum { PhongShading, Normals } ShadingMode;
  ShadingMode shadingMode = PhongShading;
  TwEnumVal enumShadingModes[] = { { PhongShading, "Phong Shading" }, { Normals, "Normals as RGB" } };
  TwType twShadingMode = TwDefineEnum("Shading Mode", enumShadingModes, 2);
  TwAddVarRW(bar, "Shading Mode", twShadingMode, &shadingMode, NULL);*/

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  Splatter splatter(resourcePath);
  gms::AutoSplats as(resourcePath+std::string("/autosplats"));
  as.resize(1024, 1024, asParam);

  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, mwheel_callback);

  ////// generate point buffer /////////
  cudaGraphicsResource_t cuPointBuffer;
  GLuint pointBuffer;
  glGenBuffers(1, &pointBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, pointBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * depthbuffer_size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&cuPointBuffer, pointBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

  ////// generate normal buffer /////////
  cudaGraphicsResource_t cuNormalBuffer;
  GLuint normalBuffer;
  glGenBuffers(1, &normalBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * depthbuffer_size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&cuNormalBuffer, normalBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

  ////// generate color buffer /////////
  cudaGraphicsResource_t cuColorBuffer;
  GLuint colorBuffer;
  glGenBuffers(1, &colorBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * depthbuffer_size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&cuColorBuffer, colorBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

  ////// generate splatAxis buffer /////////
  cudaGraphicsResource_t cuSplatAxisBuffer;
  GLuint splatAxisBuffer;
  glGenBuffers(1, &splatAxisBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, splatAxisBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * depthbuffer_size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&cuSplatAxisBuffer, splatAxisBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

  ////// generate buffer for input points ///////
  GLuint inputPointBuffer;
  glGenBuffers(1, &inputPointBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, inputPointBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * points.size(), &points[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  GLuint inputColorsBuffer;
  glGenBuffers(1, &inputColorsBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, inputColorsBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * colors.size(), &colors[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  ////// generate buffer for input normals - used for correctness computation ///////
  GLuint inputNormalsBuffer;
  glGenBuffers(1, &inputNormalsBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, inputNormalsBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * normals.size(), &normals[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // projection shader
  GLuint projectionVS = loadShader(GL_VERTEX_SHADER, resourcePath + "/shaders/proj.vert");
  GLuint projectionFS = loadShader(GL_FRAGMENT_SHADER, resourcePath + "/shaders/proj.frag");
  GLuint projectionProg = createProgram(projectionVS, 0, projectionFS);

  // projection framebuffer and textures
  GLuint projectionFramebuffer;
  glGenFramebuffers(1, &projectionFramebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, projectionFramebuffer);
  
  GLuint viewspacePosTexture; // w=1 indicates a point at this pixel
  cudaGraphicsResource_t cuViewspacePosTexture;
  glGenTextures(1, &viewspacePosTexture);
  glBindTexture(GL_TEXTURE_2D, viewspacePosTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, viewspacePosTexture, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterImage(&cuViewspacePosTexture, viewspacePosTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

  // normal texture for correctness computation
  GLuint viewspaceNormalTexture; 
  cudaGraphicsResource_t cuViewspaceNormalTexture;
  glGenTextures(1, &viewspaceNormalTexture);
  glBindTexture(GL_TEXTURE_2D, viewspaceNormalTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, viewspaceNormalTexture, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterImage(&cuViewspaceNormalTexture, viewspaceNormalTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

  // normal texture for correctness computation
  GLuint viewspaceColorTexture;
  cudaGraphicsResource_t cuViewspaceColorTexture;
  glGenTextures(1, &viewspaceColorTexture);
  glBindTexture(GL_TEXTURE_2D, viewspaceColorTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, NULL);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, viewspaceColorTexture, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  CUDA_SAFE(cudaGraphicsGLRegisterImage(&cuViewspaceColorTexture, viewspaceColorTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

  GLuint depthRenderBuffer = 0;
  glGenRenderbuffers(1, &depthRenderBuffer);
  glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // check framebuffer completeness
  uint fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    cout << "Error creating Projection FBO (" << fboStatus << ")" << endl;
  
  // allocate and copy points on device
  float3* point_device_ptr;
  CUDA_SAFE(cudaMalloc(&point_device_ptr, sizeof(float3) * points.size()));
  CUDA_SAFE(cudaMemcpy(point_device_ptr, &points[0], sizeof(float3) * points.size(), cudaMemcpyHostToDevice));

  // global gpu mem pointers
  float3* DpackedQuadtree; size_t DpackedQuadtreeSize;
  float3* Dnormals; size_t DnormalsSize;
  float3* Dcolors; size_t DcolorsSize;
  float4* DsplatAxis; size_t DsplatAxisSize;
  
  // transform resources 
  float3* DpointBuffer = cudaMallocT<float3>(depthbuffer_size);
  float* DperfectRadius = cudaMallocT<float>(depthbuffer_size);

  // quadtree structure
  unsigned int* DnumBuffer =             cudaMallocT<unsigned int>(quadtree_size);
  unsigned int* DpackedCoordinateIndex = cudaMallocT<unsigned int>(depthbuffer_size);
  
  // output
  float3* DknearestBuffer = cudaMallocT<float3>(depthbuffer_size * settings.knn_kmax);
  unsigned int* DknearestIndex = cudaMallocT<unsigned int>(depthbuffer_size * settings.knn_kmax);
  unsigned int* DpackedQuadtreeIndices = cudaMallocT<unsigned int>(quadtree_size);
  float3* DpackedQuadtreeCorrectNormals = cudaMallocT<float3>(depthbuffer_size);
  float* Ddistances = cudaMallocT<float>(depthbuffer_size * KMAX);
   
  // number of valid nearest neighbors, maybe obsolete
  unsigned int* Dvalidpoints = cudaMallocT<unsigned int>(depthbuffer_size);

  float lastStamp = glGetTime();
  float absTime = 0;
  double mousePosX = 0, mousePosY = 0;
  float translateX = 0, translateY = 0;
  float yaw = 0, pitch = 0;

/*  while (!glfwWindowShouldClose(window))
  {
    float time = glfwGetTime() / 10.f;

    glfwSwapBuffers(window);
    glfwPollEvents();
  }*/

  texRef.addressMode[0] = cudaAddressModeMirror;
  texRef.addressMode[1] = cudaAddressModeMirror;
  texRef.filterMode = cudaFilterModePoint;
  texRef.normalized = false;

  texRefNormals.addressMode[0] = cudaAddressModeMirror;
  texRefNormals.addressMode[1] = cudaAddressModeMirror;
  texRefNormals.filterMode = cudaFilterModePoint;
  texRefNormals.normalized = false;

  texRefColors.addressMode[0] = cudaAddressModeMirror;
  texRefColors.addressMode[1] = cudaAddressModeMirror;
  texRefColors.filterMode = cudaFilterModePoint;
  texRefColors.normalized = false;

#ifdef TIMING
  cudaEvent_t knnEventStart, knnEventEnd;
  cudaEventCreate(&knnEventStart);
  cudaEventCreate(&knnEventEnd);

  cudaEvent_t knnKernelEventStart, knnKernelEventEnd;
  cudaEventCreate(&knnKernelEventStart);
  cudaEventCreate(&knnKernelEventEnd);

  cudaEvent_t splatVboComputeEventStart, splatVboComputeEventEnd;
  cudaEventCreate(&splatVboComputeEventStart);
  cudaEventCreate(&splatVboComputeEventEnd);
#endif

  float bestKnnTime = 99999999999.0f;
  int bestTraverseNum = 7;
  int frameIdAccu = 0;

  std::map<int, std::vector<float> > correctNumberOfPointsHistRelative;

  while (!glfwWindowShouldClose(window))
  {
    float time = glfwGetTime();
    float t = time - lastStamp;
    fps = 1.f / t;

    if (keySpace)
      absTime += t;
    lastStamp = time;

    if (keyPlus)
      asParam.splatRadiusFactor += 0.1f;
    if (keyMinus)
      asParam.splatRadiusFactor -= 0.1f;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
      distanceToViewer += 5.f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
      distanceToViewer -= 5.f;

    double posx, posy;
    glfwGetCursorPos(window, &posx, &posy);
    float mdx = (float)(posx - mousePosX);
    float mdy = (float)(posy - mousePosY);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
    {
      if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
      {
        translateX += mdx;
        translateY -= mdy;
      }
      else
      {
        yaw += mdx;
        pitch += mdy;
      }
    }
    mousePosX = posx;
    mousePosY = posy;

    float a = fmodf(absTime * 10, 360.f);
    Math::matrix4f view = Math::matrix4f::translationnMatrix(0, -25, -distanceToViewer) * Math::matrix4f::translationnMatrix(translateX, translateY, 0) *Math::matrix4f::rotationMatrix(pitch, yaw + a, 0);
    Math::matrix4f mvp = proj_mat * view;

    float begin_time = glGetTime();

    if (knnMode != Autosplats)
    {
      // draw input points to framebuffer
#ifdef TIMING
      float projectStart = glGetTime();
#endif
      
      glBindFramebuffer(GL_FRAMEBUFFER, projectionFramebuffer);
      GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
      glDrawBuffers(3, colorBuffers);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glUseProgram(projectionProg);

      glUniformMatrix4fv(glGetUniformLocation(projectionProg, "mvpMatrix"), 1, false, mvp.data);
      glUniformMatrix4fv(glGetUniformLocation(projectionProg, "mvMatrix"), 1, false, view.data);
      
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, inputPointBuffer);
      glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
      
      glEnableVertexAttribArray(1);
      glBindBuffer(GL_ARRAY_BUFFER, inputNormalsBuffer);
      glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, 0);

      glEnableVertexAttribArray(2);
      glBindBuffer(GL_ARRAY_BUFFER, inputColorsBuffer);
      glVertexAttribPointer(2, 3, GL_FLOAT, false, 0, 0);
      
      glDrawArrays(GL_POINTS, 0, (GLsizei)points.size());
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glDisableVertexAttribArray(0);
      glDisableVertexAttribArray(1);
      glDisableVertexAttribArray(2);
      glUseProgram(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);

#ifdef TIMING
      project_time = (glGetTime() - projectStart);
      cudaEventRecord(knnEventStart, 0);
      float computeStart = glGetTime();
#endif

      cudaChannelFormatDesc cd = cudaCreateChannelDesc<float4>();

      cudaArray_t viewSpacePointArray;
      CUDA_SAFE(cudaGraphicsMapResources(1, &cuViewspacePosTexture));
      CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&viewSpacePointArray, cuViewspacePosTexture, 0, 0));
      CUDA_SAFE(cudaBindTextureToArray(&texRef, viewSpacePointArray, &cd));

      cudaArray_t viewSpaceNormalArray;
      CUDA_SAFE(cudaGraphicsMapResources(1, &cuViewspaceNormalTexture));
      CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&viewSpaceNormalArray, cuViewspaceNormalTexture, 0, 0));
      CUDA_SAFE(cudaBindTextureToArray(&texRefNormals, viewSpaceNormalArray, &cd));

      cudaArray_t viewSpaceColorArray;
      CUDA_SAFE(cudaGraphicsMapResources(1, &cuViewspaceColorTexture));
      CUDA_SAFE(cudaGraphicsSubResourceGetMappedArray(&viewSpaceColorArray, cuViewspaceColorTexture, 0, 0));
      CUDA_SAFE(cudaBindTextureToArray(&texRefColors, viewSpaceColorArray, &cd));

      // mapping opengl buffers for splatting
      cudaGraphicsResource_t res[] = { cuPointBuffer, cuNormalBuffer, cuColorBuffer, cuSplatAxisBuffer };
      CUDA_SAFE(cudaGraphicsMapResources(4, res));
      CUDA_SAFE(cudaGraphicsResourceGetMappedPointer((void**)&DpackedQuadtree, &DpackedQuadtreeSize, cuPointBuffer));
      CUDA_SAFE(cudaGraphicsResourceGetMappedPointer((void**)&Dnormals, &DnormalsSize, cuNormalBuffer));
      CUDA_SAFE(cudaGraphicsResourceGetMappedPointer((void**)&Dcolors, &DcolorsSize, cuColorBuffer));
      CUDA_SAFE(cudaGraphicsResourceGetMappedPointer((void**)&DsplatAxis, &DsplatAxisSize, cuSplatAxisBuffer));

      copyNums << <1024 * 1024 / 512, 512 >> >(DnumBuffer, 1024); CUDA_SAFE_KERNEL

      // stage 2 count points on quadtree levels
      unsigned int offset = 0;
      unsigned int level = 1;
      for (unsigned int size = resolution / 2; size > 0; size /= 2)
      {
        quadtree_kernel << <size*size / 256 + 1, 256 >> >(DnumBuffer, size, offset, level); CUDA_SAFE_KERNEL
          offset += size * 4 * size;
        level++;
      }

      // stage 3 pack indices (= adressing for random access into quadtree)
      offset = quadtree_size;
      for (unsigned int size = 1; size < resolution; size <<= 1)
      {
        const int n = size*size;
        offset -= n;
        packQuadtreeIndices_kernel << <(n / 256) + 1, 256 >> >(DnumBuffer, DpackedQuadtreeIndices, size, offset); CUDA_SAFE_KERNEL
      }

      // stage 4 pack the quadtree to a dense array
      packQuadtree_kernel << <depthbuffer_size / 256, 256 >> >(DnumBuffer, DpackedQuadtreeIndices, DpointBuffer, DpackedQuadtree, Dcolors, DpackedQuadtreeCorrectNormals, DpackedCoordinateIndex, resolution); CUDA_SAFE_KERNEL

        // read number of screenspace points
      CUDA_SAFE(cudaMemcpy((void*)&numPointsInScreenspace, &DnumBuffer[quadtree_size - 1], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));


      set_globals << <1, 1 >> >(
        DpackedQuadtreeIndices,
        DnumBuffer,
        DpackedQuadtree,
        DknearestBuffer);

      //        knn_kernel_bruteforce_k8 << <(numPointsInScreenspace / THREADNUM) + 1, THREADNUM >> >(numPointsInScreenspace, kmax, DknearestIndex, Dnormals, DsplatAxis); CUDA_SAFE_KERNEL
#ifdef TIMING
      cudaEventRecord(knnKernelEventStart, 0);
#endif

#ifndef TIMING
      traverseLevel = 7;
#endif

      knn_kernel_quadtree_k8 << <(numPointsInScreenspace / THREADNUM) + 1, THREADNUM >> >(
        numPointsInScreenspace, maxIterations,
        settings.knn_kmin, traverseLevel,
        settings.iterative_radius_multiplier, settings.initial_radius_multiplier,
        DpackedCoordinateIndex,
        DknearestIndex, Dnormals, DsplatAxis, DperfectRadius, Dvalidpoints, Ddistances); CUDA_SAFE_KERNEL

#ifdef TIMING
      cudaEventRecord(knnKernelEventEnd, 0);
      cudaEventRecord(splatVboComputeEventStart, 0);
#endif
      computeSplattingVbos << <(numPointsInScreenspace / THREADNUM) + 1, THREADNUM >> >(
        numPointsInScreenspace, 
        settings.knn_kmin,
        DknearestIndex, 
        Dvalidpoints,
        Dnormals, 
        DsplatAxis); CUDA_SAFE_KERNEL
#ifdef TIMING
      cudaEventRecord(splatVboComputeEventEnd, 0);
#endif

      CUDA_SAFE(cudaUnbindTexture(&texRef));
      CUDA_SAFE(cudaUnbindTexture(&texRefNormals));
      CUDA_SAFE(cudaUnbindTexture(&texRefColors));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuPointBuffer));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuNormalBuffer));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuColorBuffer));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuSplatAxisBuffer));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuViewspacePosTexture));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuViewspaceNormalTexture));
      CUDA_SAFE(cudaGraphicsUnmapResources(1, &cuViewspaceColorTexture));

#ifdef TIMING
      cudaEventRecord(knnEventEnd, 0);
      cudaEventSynchronize(knnEventEnd);
      cudaEventSynchronize(knnKernelEventEnd);
      cudaEventSynchronize(splatVboComputeEventEnd);

      float knnTime, knnKernelTime, splatVboKernelTime;
      cudaEventElapsedTime(&knnTime, knnEventStart, knnEventEnd);
      cudaEventElapsedTime(&knnKernelTime, knnKernelEventStart, knnKernelEventEnd);
      cudaEventElapsedTime(&splatVboKernelTime, splatVboComputeEventStart, splatVboComputeEventEnd);

      knn_time = knnKernelTime;
      splatVbo_time = splatVboKernelTime;
      interopOverhead_time = knnTime - knnKernelTime - splatVboKernelTime;

      // traverse depth optimization
/*      if (frameIdAccu == 0)
      {
        bestKnnTime = 9999999.f;
        traverseLevel = 5;
        std::cout << "begin set traverse to 5" << std::endl;
      }
      else if (frameIdAccu >= 1 && frameIdAccu <= 4)
      {
        if (knnTime < bestKnnTime)
        {
          bestKnnTime = knnTime;
          bestTraverseNum = frameIdAccu + 4;
        }
        traverseLevel = frameIdAccu + 5;

        std::cout << frameIdAccu + 4 << " time " << knnTime << " best " << bestTraverseNum << std::endl;
      }
      else if (frameIdAccu == 5)
      {
        if (knnTime < bestKnnTime)
        {
          bestKnnTime = knnTime;
          bestTraverseNum = frameIdAccu + 4;
        }
      }
      else
        traverseLevel = bestTraverseNum;

      //std::cout << traverseLevel << std::endl;

      frameIdAccu++;
      frameIdAccu %= 300;*/

#endif
       
      if (compute_error)
      {
        CUDA_SAFE(cudaMemcpy(HcorrectNormals, &DpackedQuadtreeCorrectNormals[0], sizeof(float3) * numPointsInScreenspace, cudaMemcpyDeviceToHost));
        CUDA_SAFE(cudaMemcpy(HpackedQuadtree, &DpackedQuadtree[0], sizeof(float3) * numPointsInScreenspace, cudaMemcpyDeviceToHost));
        CUDA_SAFE(cudaMemcpy(HknearestIndex, &DknearestIndex[0], sizeof(unsigned) * KMAX * numPointsInScreenspace, cudaMemcpyDeviceToHost));
        glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hnormals);

        std::vector<double> errors;
        std::pair<float, float> err = computeNormalError(numPointsInScreenspace, Hnormals, HcorrectNormals, errors, HpackedQuadtree);
/*        std::ofstream error_file(settings.file + "_error_fastknn.txt");
        for (int i = 0; i < errors.size(); ++i)
          error_file << errors[i] << std::endl;
        error_file.close();*/

        fastknn_kdtree_error_mean = err.first;
        fastknn_kdtree_error_var = err.second;

        std::vector<unsigned> correctNeighbors;
        errorKdtree(settings.knn_kmin, KMAX, numPointsInScreenspace, Hnormals, HpackedQuadtree, HknearestIndex, errors, correctNeighbors);

        std::vector<int> correctNeighborHist(settings.knn_kmin+1, 0);
        for (int i = 0; i < correctNeighbors.size(); ++i)
          correctNeighborHist[correctNeighbors[i]]++;

        std::cout << "corr. neighbors: ";
        std::vector<float> relativeHist(settings.knn_kmin+1, 0);
        for (int i = 0; i <= settings.knn_kmin; i++)
        {
          relativeHist[i] = (float)correctNeighborHist[i] / (float)correctNeighbors.size();
          //std::cout << std::setprecision(2) << (float)correctNeighborHist[i] / (float)correctNeighbors.size() << " ";
        }
        //std::cout << std::endl;

        correctNumberOfPointsHistRelative[settings.knn_kmin] = relativeHist;



/*        std::ofstream neighbors_file(settings.file + std::to_string(maxIterations) + "_correctNeighbors.txt");
        for (int i = 0; i < correctNeighbors.size(); ++i)
          neighbors_file << correctNeighbors[i] << std::endl;
        neighbors_file.close();

        std::ofstream error_file_kd(settings.file + "_error_fastknn_vs_kdtree.txt");
        for (int i = 0; i < errors.size(); ++i)
          error_file_kd << errors[i] << std::endl;
        error_file_kd.close();*/


        /// inject errors as colors 
        const float minError = 0.0f;
        const float maxError = 90.f;
        for (int i = 0; i < errors.size(); ++i)
        {
          
          float e = errors[i];//1.f - cos(errors[i] * 3.141592f / 180.f);
          e = (e - minError) / (maxError - minError);
          e = e < 0 ? 0 : e;
          e = e > 1 ? 1 : e;

          float q = e * 4.f;
          
          if (q >= 0.f && q <= 1.f)
          {
            float3 f = { 0, q, 1 };
            Hcolors[i] = f;
          }
          else if (q > 1.f && q <= 2.f)
          {
            float3 f = { 0, 1, 1.f-q-1.f };
            Hcolors[i] = f;
          }
          else if (q > 2.f && q <= 3.f)
          {
            float3 f = { q-2.f, 1, 0 };
            Hcolors[i] = f;
          }
          else if (q > 3.f && q <= 4.f)
          {
            float3 f = { 1, 1.f-q-3.f, 0.f };
            Hcolors[i] = f;
          }
          else
          {
            float3 f = { 0,0,0 };
            Hcolors[i] = f;
          }

        }
        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hcolors);
        //////////////////////////////
      }
    }

    ///// cpu normal/axis computation /////
    if (knnMode == Pcl)
    {
      //computeNormalErrorVsGroundtruth(settings.knn_kmin, numPointsInScreenspace, pointBuffer, normalBuffer, splatAxisBuffer, HpackedQuadtree, Hnormals, HsplatAxis, true);

      // inject cpu outcome to gl buffers
      glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hnormals);
      glBindBuffer(GL_ARRAY_BUFFER, splatAxisBuffer);
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float4) * numPointsInScreenspace, HsplatAxis);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    
    if (knnMode != Autosplats)
    {
#ifdef TIMING
      float splatStart = glGetTime();
#endif
      splatter.draw(
        numPointsInScreenspace,
        projection,
        inverse(projection),
        pointBuffer, colorBuffer, normalBuffer, splatAxisBuffer, splatSizeFactor, useColor, isShaded);
#ifdef TIMING
      splat_time = (glGetTime() - splatStart);
#endif
    }

    if (knnMode == Autosplats)
    {
      gms::mat4 v = gms::mat4(view.data);
      v.transpose();
      asParam.splatRadiusFactor = splatSizeFactor;
      asParam.K = settings.knn_kmin;
      asParam.maxIters = maxIterations;
      as.render(inputPointBuffer, inputNormalsBuffer, (unsigned)points.size(), v, projection, 1.5f, 1000.0f, asParam);
      numPointsInScreenspace = *as.getNumScreenPointsPtr();

      if (compute_error)
      {
        glBindBuffer(GL_ARRAY_BUFFER, as.mASCorrectNormalsVbo);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, HcorrectNormals);
        glBindBuffer(GL_ARRAY_BUFFER, as.mASPositionVbo);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, HpackedQuadtree);
        glBindBuffer(GL_ARRAY_BUFFER, as.mASNormalsVbo);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float3) * numPointsInScreenspace, Hnormals);
        
        std::vector<double> errors;
        std::pair<float, float> err = computeNormalError(numPointsInScreenspace, Hnormals, HcorrectNormals, errors, HpackedQuadtree);
        string tempstr = settings.file + "_error_autosplats.txt";
        std::ofstream error_file(tempstr.c_str());
        for (int i = 0; i < errors.size(); ++i)
          error_file << errors[i] << std::endl;
        error_file.close();

        //std::pair<float,float> err = computeNormalErrorVsGroundtruth(settings.knn_kmin, numPointsInScreenspace, as.mASPositionVbo, as.mASNormalsVbo, as.mASSplatAxesVbo, HpackedQuadtree, Hnormals, HcorrectNormals, HsplatAxis, false);
        autosplats_kdtree_error_mean = err.first;
        autosplats_kdtree_error_var = err.second;
        //std::cout << "computed normal error in degree: " << err << "\B0 average error per point: " << err/(float)numPointsInScreenspace << "\B0" << std::endl;
      }

      project_time = (float)as.timings.project;
      
      knn_time = (float)(as.timings.total - as.timings.project - as.timings.splatting - as.timings.computeSplats);
      splatVbo_time = as.timings.computeSplats;
      interopOverhead_time = 0;
      splat_time = (float)as.timings.splatting;
    }

    total_time = project_time + knn_time + splatVbo_time + interopOverhead_time + splat_time;

    // copy timings to clipboard
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
      char tempstr[255];
      printf(tempstr, "%f %f %f %f %f", knn_time, project_time, splatVbo_time, interopOverhead_time, splat_time);
      glfwSetClipboardString(window, tempstr);
    }

    TwDraw();
    glfwSwapBuffers(window);
    glfwPollEvents();

    theoretical_fps = 1000.f / (total_time);
  }
  

  // clean up opengl resources
  glDeleteBuffers(1, &pointBuffer);
  glDeleteBuffers(1, &normalBuffer);
  glDeleteBuffers(1, &splatAxisBuffer);
  
  // clean up cuda resources
  CUDA_SAFE(cudaFree(DpointBuffer));
  CUDA_SAFE(cudaFree(DnumBuffer));
  CUDA_SAFE(cudaFree(DpackedCoordinateIndex));
  CUDA_SAFE(cudaFree(DknearestBuffer));
  CUDA_SAFE(cudaFree(DknearestIndex));
  CUDA_SAFE(cudaFree(DpackedQuadtreeIndices));
  CUDA_SAFE(cudaFree(Dvalidpoints));

  glfwDestroyWindow(window);
  glfwTerminate();
  cudaDeviceReset();

  LoggingData log;
  /*log.transformationTime = transform_points_time;
  log.quadtreeTime = quadtree_time;
  log.candidateTime = candidate_time;
  log.distanceTime = distance_time;
  log.copyTime = copy_time;*/
  log.processedPoints = points.size();
  //log.processedScreenspacePoints = numPointsInScreenspace;

  string tempstr(settings.file + "_correct_point_hist.txt");
//  std::ofstream of_hist(settings.file + "_correct_point_hist.txt");
  std::ofstream of_hist(tempstr.c_str());


  for (std::map<int, std::vector<float> >::iterator iter = correctNumberOfPointsHistRelative.begin(); iter != correctNumberOfPointsHistRelative.end(); ++iter)
  {
    for (int j = 0; j < iter->second.size(); ++j)
    {
      of_hist << std::setprecision(2) << iter->second[j] << " ";
    }
    of_hist << std::endl;
  }
    
  of_hist.close();
  
  return log;
}
