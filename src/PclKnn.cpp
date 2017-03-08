#include "PclKnn.h"

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Eigenvalues>

void heatmap(float minimum, float maximum, float value, float* rgb)
{
  value = std::min(std::max(value, minimum), maximum);
  float p = 4.f * (value - minimum) / (maximum - minimum);
  float i;
  if (p < 1.f)
  {
    i = p - 1.f;
    rgb[0] = 0.f; rgb[1] = i; rgb[2] = 1.f;
  }
  else if (p < 2.f)
  {
    i = p - 2.f;
    rgb[0] = 0.f; rgb[1] = 1.f; rgb[2] = 1.f - i;
  }
  else if (p < 3.f)
  {
    i = p - 3.f;
    rgb[0] = i; rgb[1] = 1.f; rgb[2] = 0.0f;
  }
  else if (p < 4.f)
  {
    i = p - 4.f;
    rgb[0] = 1.f; rgb[1] = 1.f - i; rgb[2] = 0.f;
  }
}

void errorKdtree(unsigned kmin, unsigned kmax, unsigned numPointsInScreenspace, float3* Hnormals, float3* HpackedQuadtree, unsigned* Hkindex, std::vector<double>& errors, std::vector<unsigned>& numCorrectNeighbors)
{
  typedef pcl::PointXYZ Ptype;
  pcl::PointCloud<Ptype>::Ptr cloud(new pcl::PointCloud<Ptype>);
  cloud->width = numPointsInScreenspace;
  cloud->height = 1;
  cloud->points.resize(cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    cloud->points[i].x = HpackedQuadtree[i].x;
    cloud->points[i].y = HpackedQuadtree[i].y;
    cloud->points[i].z = HpackedQuadtree[i].z;
  }

  pcl::KdTreeFLANN<Ptype> kdtree;
  kdtree.setInputCloud(cloud);
  errors.resize(numPointsInScreenspace, 0.0);
  numCorrectNeighbors.resize(numPointsInScreenspace, 0);
  const int knn = kmin;

  for (size_t i = 0; i < numPointsInScreenspace; ++i)
  {
    const Ptype& searchPoint = cloud->points[i];
    std::vector<int> pointIdxNKNSearch(knn);
    std::vector<float> pointNKNSquaredDistance(knn);
    kdtree.nearestKSearch(searchPoint, knn, pointIdxNKNSearch, pointNKNSquaredDistance);

    std::vector<double> data;
    for (int n = 0; n < pointIdxNKNSearch.size(); ++n)
    {
      //int idx = HknearestIndex[i*settings.knn_kmax + n];
      int idx = pointIdxNKNSearch[n];
      const float3 p = HpackedQuadtree[idx];
      data.push_back(p.x); data.push_back(p.y); data.push_back(p.z);
    }

    unsigned correctNum = 0;
    for (int n = 0; n < kmin; n++)
    {
      unsigned idx = Hkindex[kmax * i + n];
      if (std::find(pointIdxNKNSearch.begin(), pointIdxNKNSearch.end(), idx) != pointIdxNKNSearch.end())
        correctNum++;
    }
    numCorrectNeighbors[i] = correctNum;

    ////// eigenvector computation with eigen /////////
    Eigen::Map<Eigen::MatrixXd> mat = Eigen::Map<Eigen::MatrixXd>(data.data(), 3, data.size() / 3);
    Eigen::MatrixXd centered = mat.transpose().rowwise() - mat.transpose().colwise().mean();
    Eigen::MatrixXd cov = centered.adjoint() * centered;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    Eigen::MatrixXd eigvec = eig.eigenvectors();
    Eigen::Vector3d eigVec0 = Eigen::Vector3d(eigvec(0, 0), eigvec(1, 0), eigvec(2, 0));
    Eigen::Vector3d eigVec2 = Eigen::Vector3d(eigvec(0, 2), eigvec(1, 2), eigvec(2, 2));

    double cosAngle = std::min<double>(1.0, std::abs(Hnormals[i].x * eigVec0.x() + Hnormals[i].y * eigVec0.y() + Hnormals[i].z * eigVec0.z()));
    double angle = acos(cosAngle) * 180.0 / 3.141592;
    errors[i] = angle;

  }
}

std::pair<float, float> computeNormalError(unsigned numPointsInScreenspace, float3* Hnormals, float3* HcorrectNormals, std::vector<double>& errors, float3* HpackedQuadtree)
{
  
  
  unsigned ok = 0;
  unsigned equal_neighbour_num = 0;
  double normal_error_sum = 0;

  errors.resize(numPointsInScreenspace, 0.0);

  size_t valid = 0;
  double minError = 9999999.0;
  double maxError = 0.0;

  const int knn = 8;

  for (size_t i = 0; i < numPointsInScreenspace; ++i)
  {
    /*if (injectResults)
    {
      Hnormals[i].x = eigVec0.x();
      Hnormals[i].y = eigVec0.y();
      Hnormals[i].z = eigVec0.z();
    }*/

    /*    float normalError = 1.f - abs(eigvec(0, 0) * Hnormals[i].x + eigvec(1, 0) * Hnormals[i].y + eigvec(2, 0) * Hnormals[i].z);
      float heat[3];
      heatmap(0.0f, 0.05f, normalError, heat);
      Hnormals[i].x = heat[0];
      Hnormals[i].y = heat[1];
      Hnormals[i].z = heat[2];

      if (normalError < 0.1f) // less than 0.9° error
      ok++;*/

    //std::cout << HcorrectNormals[i].x << " " << HcorrectNormals[i].y << " " << HcorrectNormals[i].z << " " << std::endl;

    //double cosAngle = std::min<double>(1.0, std::abs(Hnormals[i].x * eigVec0.x() + Hnormals[i].y * eigVec0.y() + Hnormals[i].z * eigVec0.z()));
    

    double cosAngle = std::abs(Hnormals[i].x * HcorrectNormals[i].x + Hnormals[i].y * HcorrectNormals[i].y + Hnormals[i].z * HcorrectNormals[i].z);
    double angle = acos(cosAngle) * 180.0 / 3.141592;
    if (!(cosAngle >= 0 && cosAngle <= 1))
    {
      angle = 0;
//      std::cout << cosAngle << " " << angle << " " << Hnormals[i].x << " " << Hnormals[i].y << " " << Hnormals[i].z << std::endl;
    }

    errors[i] = angle;

/*    if (cosAngle < 0)
    {
      std::cout << "1> " << Hnormals[i].x << " " << Hnormals[i].y << " " << Hnormals[i].z << std::endl;
      std::cout << "2> " << HcorrectNormals[i].x << " " << HcorrectNormals[i].y << " " << HcorrectNormals[i].z << std::endl;
      std::cout << cosAngle << " " << Hnormals[i].x * HpackedQuadtree[i].x + Hnormals[i].y * HpackedQuadtree[i].y + Hnormals[i].z * HpackedQuadtree[i].z << " " <<
        HcorrectNormals[i].x * HpackedQuadtree[i].x + HcorrectNormals[i].y * HpackedQuadtree[i].y + HcorrectNormals[i].z * HpackedQuadtree[i].z << std::endl << std::endl;
    }*/

    normal_error_sum += angle;
    valid++;

 /*   float majorLength = sqrt(eig.eigenvalues()[2] / (pointIdxNKNSearch.size() - 1));
    float minorLength = sqrt(eig.eigenvalues()[1] / (pointIdxNKNSearch.size() - 1));

    //std::cout << "hm? " << HsplatAxis[i].x << " " << HsplatAxis[i].y;

    if (injectResults)
    {
      HsplatAxis[i].x = eigVec2.x() * majorLength * 1.5f;
      HsplatAxis[i].y = eigVec2.y() * majorLength * 1.5f;
      HsplatAxis[i].z = eigVec2.z() * majorLength * 1.5f;
      HsplatAxis[i].w = minorLength / majorLength;
    }*/
  }
  double mean = normal_error_sum / (double)valid;
  double var = 0.0f;
  for (size_t i = 0; i < numPointsInScreenspace; ++i)
  {
    double diff = errors[i] - mean;
    var += diff*diff;
  }
  var /= (double)(valid - 1);

//  std::cout << mean << " " << valid << " " << minError << " " << maxError << std::endl;

  /*for (int i = 0; i < cloud->points.size(); ++i)
  {
    std::cout << errors[i] << " ";
  }*/

//  std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;

  return std::make_pair(mean, sqrt(var));
}