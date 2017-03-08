
#include <iostream>
#include "PCLReader.h"

#include "kernel.h"

#include "matrix4f.h"
#include "vector3f.h"

int main(int argc, char** argv)
{
  size_t numpoints = 5000000;
  PCLReader reader;
  std::cout << "Load PCL..." << std::endl;
  std::string file = "gnome1.xyz";
  
  std::vector<float3> points;
  points = reader.readFileTXT("gnome1.xyz");
  
  if(points.size() == 0)
  {
    std::cout << "ERROR - could not read data" << std::endl;
    return 1;
  }

  numpoints = points.size();

  // create matrix on host
  depth_pixel* depth_buffer = new depth_pixel[1024*1024];
  TransformationSettings settings;
  
  
  Math::matrix4f mat = Math::matrix4f::projectionMatrix(1.5f,1000.0f,1.0f,1.0f)  * Math::matrix4f::translationnMatrix(0,0,150.0f) * Math::matrix4f::rotationMatrix(0,0,0); 


  for(int k = 2; k <= 50; k++)
  {
    settings.knn_k = k;

    Math::vector3f* HknearestBuffer = new Math::vector3f[settings.screenbuffer_size*settings.screenbuffer_size*settings.knn_k];
    unsigned int* HnumBuffer = new unsigned int[settings.quadtree_size()];

    LoggingData log = transform_points(settings, points, mat, depth_buffer, HknearestBuffer, HnumBuffer);
    std::cout << "k=" << k << " " << log.execTime() << "ms" << std::endl;

    delete [] HknearestBuffer;
    delete [] HnumBuffer;

  }

 
  getchar();
  return 0;
}
