#pragma once

struct sort_str
{
private:
  float dist;
  unsigned int idx;

public:
  __device__ __forceinline__ float getDistance() const
  {
    //return __half2float(dist);
    return dist;
  }

  __device__ __forceinline__ void setDistance(float f)
  {
    //dist = __float2half_rn(f);
    dist = f;
  }

  __device__ __forceinline__ unsigned int getIdx() const
  {
    return idx;
  }

  __device__ __forceinline__ void setIdx(unsigned int i)
  {
    idx = i;
  }

  __device__ __forceinline__ bool operator<(sort_str const& rhs) const
  {
    return dist < rhs.dist;
  }

  __device__ __forceinline__ bool operator>(sort_str const& rhs) const
  {
    return dist > rhs.dist;
  }
};