#pragma once

#include <fstream>
#include <vector>
#include "vector3f.h"
#include <cutil_math.h>

/*
struct FileNotFoundException : public std::exception
{
  std::string message;
  FileNotFoundException(std::string ss) : message(ss) {}
  virtual ~FileNotFoundException(){}
  const char* what() const throw() { return message.c_str(); }
};
*/

class PCLReader
{
public:
  PCLReader(const std::string& resourcePath);
  void readFileTXT(std::string path, std::vector<float3>& points, std::vector<float3>& normals, std::vector<float3>& colors, bool isAbsolutePath = false);
  //void readPCD(std::string path, std::vector<float3>& points);

  std::vector<float3> createPoints(size_t num, const float min, const float max);


private:
  const std::string data_path;
};
