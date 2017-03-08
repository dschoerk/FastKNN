#pragma once

#include "matrix4f.h"
#include "vector3f.h"
#include "depth_pixel.h"
#include "datatypes.h"
#include <vector>

extern "C"
LoggingData transform_points(TransformationSettings settings, const std::vector<float3>& points, const std::vector<float3>& normals, const std::vector<float3>& colors, const Math::matrix4f& view_mat, const Math::matrix4f& proj_mat);

void printDeviceInfo();
//extern "C"
//  void build_quadtree(const std::vector<Math::vector3f>& points, const Math::matrix4f& mat, uint* out);