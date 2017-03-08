#include "cmake_conf.h"

//#define CORRECTNESS_TEST
#include <iostream>
#include <string>
#include "PCLReader.h"

#include <cuda_runtime.h>

#include "matrix4f.h"
#include "vector3f.h"
#include "kernel.h"
#include "datatypes.h"

int main(int argc, char** argv)
{
  PCLReader reader(conf_SOURCE_DIR);

  TransformationSettings settings;
  settings.knn_kmin  = 8;
  settings.knn_kmax = 64;
  settings.initial_radius_multiplier = 1.0f;
  settings.iterative_radius_multiplier = 1.44f;

  std::vector<std::string> param;
  for(int i=1;i < argc; i++)
  {
    param.push_back(argv[i]);
  }

  //settings.file = "armadillo.xyz";
  settings.file = "bunny.xyz";
  //settings.file = "dragon2.xyz";
  //settings.file = "happybuddah.xyz";
  //settings.file = "model_006.xyz";
  //settings.file = "gnome1.xyz";

  //settings.file = "scan000.3d";

  std::vector<float3> points;
  std::vector<float3> colors;
  std::vector<float3> normals;

  if(param.size() == 0)
  {
     reader.readFileTXT(settings.file, points, normals, colors);
    //reader.readPCD(settings.file, points);
  }
  else
  {
    settings.file = param[0];
    reader.readFileTXT(settings.file, points, normals, colors);
  }

  if (points.size() == 0)
  {
    std::cout << "ERROR - could not read data" << std::endl;
    return 1;
  }

  // create matrix on host
  Math::matrix4f proj_mat = Math::matrix4f::projectionMatrix(1.5f,1000.0f,1.0f / 2.f,1.0f / 2.f);//  * 
  Math::matrix4f view_mat = Math::matrix4f::translationnMatrix(0, 0, -350.f);// *Math::matrix4f::rotationMatrix(0, 180.f, 0);
  //transform_points(settings, points, view_mat, proj_mat, depth_buffer, HknearestBuffer, HnumBuffer);
  LoggingData log = transform_points(settings, points, normals, colors, view_mat, proj_mat);
  return 0;
}
