#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <iostream>
#include <vector>

#include "cmake_conf.h"

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (conf_SOURCE_DIR + std::string("/data/gnome1.pcd"), *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
  
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud (cloud);

  // K nearest neighbor search
  int K = 8;

  std::cout << "start knn search" << std::endl;

  float avg = 0;
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    /*std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z << std::endl;*/
  
    const auto& searchPoint = cloud->points[i];

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);

    float avg_distance = 0.0f;
    for(int i=0;i<pointNKNSquaredDistance.size();i++)
    {
      avg_distance = std::max<float>(avg_distance, sqrt(pointNKNSquaredDistance[i]));
    }
    avg += avg_distance;
  }

  avg /= cloud->points.size ();

  std::cout << "average knn radius: " << avg << std::endl;

  getchar();
  return 0;
}