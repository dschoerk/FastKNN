#pragma once
#include <cuda_runtime.h>
#include <utility>
#include <vector>

std::pair<float, float> computeNormalError(unsigned numPointsInScreenspace, float3* Hnormals, float3* HcorrectNormals, std::vector<double>& errors, float3* HpackedQuadtree);
void errorKdtree(unsigned kmin, unsigned kmax, unsigned numPointsInScreenspace, float3* Hnormals, float3* HpackedQuadtree, unsigned* Hkindex, std::vector<double>& errors, std::vector<unsigned>& numCorrectNeighbors);