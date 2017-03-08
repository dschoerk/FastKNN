#pragma once


#include <ostream>

/*#ifndef __device__
  #define __device__
  #define __host__
#else
  
#endif*/
#include <cuda_runtime.h>

namespace Math
{
  struct vector3f
  {
    vector3f()
    {
      data[0] = 0;
      data[1] = 0;
      data[2] = 0;
    }

    __device__ __host__
    vector3f(float x, float y, float z)
    {
	    data[0] = x;
		  data[1] = y;
      data[2] = z;
    }

    __device__ __host__
    const float& operator()(int index) const
    {
      return data[index];
    }

    __device__ __host__
    float& operator()(int index)
    {
      return data[index];
    }

    __device__ __host__
    vector3f operator-(const vector3f& v) const
    {
      return vector3f(data[0]-v(0), data[1]-v(1), data[2]-v(2));
    }

    __device__ __host__
    float length_sq() const
    {
      return data[0]*data[0]+data[1]*data[1]+data[2]*data[2];
    }

    std::ostream& toString(std::ostream & out) const
    {
      out << "[" << data[0] << " " << data[1] << " " << data[2] << "]";
      return out;
    }

  private:
    float data[3];
  };
}

//std::ostream& operator<< (std::ostream &o, const Math::vector3f &a);



