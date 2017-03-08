#include "PCLReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include<stdlib.h>

//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>

PCLReader::PCLReader(const std::string& resourcePath) : data_path(resourcePath)
{

}

/*void PCLReader::readPCD(std::string path, std::vector<float3>& points)
{
  std::cout << "big" << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  //if (pcl::io::loadPCDFile<pcl::PointXYZ>(data_path + "/data/" + path, *cloud) == -1) //* load the file
  //  return;

  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile(data_path + "/data/" + path, cloud_blob);
  pcl::fromPCLPointCloud2(cloud_blob, *cloud);

  std::cout << "bang" << std::endl;
  float3 vec;
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    vec.x = cloud->points[i].x;
    vec.y = cloud->points[i].y;
    vec.z = cloud->points[i].z;

    std::cout << vec.x << " " << vec.y << " " << vec.z << std::endl;
    points.push_back(vec);
  }
}*/

void PCLReader::readFileTXT(std::string path, std::vector<float3>& points, std::vector<float3>& normals, std::vector<float3>& colors, bool isAbsolutePath)
{
/*  points.clear();
  colors.clear();*/


  std::string component;

  float3 vec;
  float3 col;
  float3 n;

  if (!isAbsolutePath)
    path = (data_path + "/data/" + path);
  
  std::ifstream myfile(path.c_str());
  if (myfile.is_open())
  {
    std::string line;
    while (std::getline(myfile, line))
    {
      std::istringstream iss;
      iss.str(line);

      const float scale = 1.f;
      std::getline(iss, component, ' ');
      vec.x = scale * (float)atof(component.c_str());
      std::getline(iss, component, ' ');
      vec.y = scale * (float)atof(component.c_str());
      std::getline(iss, component, ' ');
      vec.z = scale * (float)atof(component.c_str());

      std::getline(iss, component, ' ');
      n.x = (float)atof(component.c_str());
      std::getline(iss, component, ' ');
      n.y = (float)atof(component.c_str());
      std::getline(iss, component, ' ');
      n.z = (float)atof(component.c_str());

      std::getline(iss, component, ' ');
      col.x = (float)atof(component.c_str()) / 255.f;
      std::getline(iss, component, ' ');
      col.y = (float)atof(component.c_str()) / 255.f;
      std::getline(iss, component, ' ');
      col.z = (float)atof(component.c_str()) / 255.f;

      points.push_back(vec);
      colors.push_back(col);
      normals.push_back(n);
    }
    myfile.close();
  }

//  else
//    throw FileNotFoundException(std::string("file not found ")+path);
  
}

std::vector<float3> PCLReader::createPoints(size_t numpoints, const float min, const float max)
{
  
  /*
  srand(0);
  for(int i=0;i<numpoints;i++)
  {
    v[i](0) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max-min))) + min;
    v[i](1) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max-min))) + min;
    v[i](2) = 100;
  }*/

  int w = 1024;
  int h = 1024;
  std::vector<float3> v;
  v.resize(w*h, float3());

  for(int x = 0; x < w; x++)
  {
    for(int y = 0; y < h; y++)
    {
      v[x*h+y].x = (((float)x / (float)w) - 0.5f) * 200.0f;
      v[x*h+y].y = (((float)y / (float)h) - 0.5f) * 200.0f;
      v[x*h+y].z = 150;
    }  
  }
  return v;
}
