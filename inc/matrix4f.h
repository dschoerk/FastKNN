#pragma once

#include <math.h>
#include <vector3f.h>
#include <cutil_math.h>

namespace Math
{
  // Column major matrix
  struct matrix4f
  {
    float data[16];

    __host__ __device__
    matrix4f()
    {
      data[0] = 1;
      data[1] = 0;
      data[2] = 0;
      data[3] = 0;

      data[4] = 0;
      data[5] = 1;
      data[6] = 0;
      data[7] = 0;

      data[8] = 0;
      data[9] = 0;
      data[10] = 1;
      data[11] = 0;

      data[12] = 0;
      data[13] = 0;
      data[14] = 0;
      data[15] = 1;
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
    const float& operator()(int row, int col) const
    {
      return data[row+4*col];
    }

    __device__ __host__
    float& operator()(int row, int col)
    {
      return data[row+4*col];
    }

    __device__ __host__
    vector3f operator*(const vector3f& v) const
    {
      const matrix4f& m = *this;
      float x = v(0) * m(0,0) + v(1) * m(0,1) + v(2) * m(0,2) + m(0,3);
      float y = v(0) * m(1,0) + v(1) * m(1,1) + v(2) * m(1,2) + m(1,3);
      float z = v(0) * m(2,0) + v(1) * m(2,1) + v(2) * m(2,2) + m(2,3);
      float w = v(0) * m(3,0) + v(1) * m(3,1) + v(2) * m(3,2) + m(3,3);
      return vector3f(x/w,y/w,z/w);
    }

    __device__ __host__
    static vector3f mul(const float m[16], const vector3f& v)
    {
      float x = v(0) * m[0] + v(1) * m[4] + v(2) * m[8] + m[12];
      float y = v(0) * m[1] + v(1) * m[5] + v(2) * m[9] + m[13];
      float z = v(0) * m[2] + v(1) * m[6] + v(2) * m[10] + m[14];
      float w = v(0) * m[3] + v(1) * m[7] + v(2) * m[11] + m[15];
      return vector3f(x/w,y/w,z/w);
    }

    __device__ __host__
    static float3 mul(const float m[16], float3 v)
    {
      float x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
      float y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
      float z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
      float w = v.x * m[3] + v.y * m[7] + v.z * m[11] + m[15];
      
      return make_float3(x/w, y/w, z/w);
    }

    __device__ __host__
    static float4 mul(const float m[16], float4 v)
    {
      float x = v.x * m[0] + v.y * m[4] + v.z * m[8] + m[12];
      float y = v.x * m[1] + v.y * m[5] + v.z * m[9] + m[13];
      float z = v.x * m[2] + v.y * m[6] + v.z * m[10] + m[14];
      float w = v.x * m[3] + v.y * m[7] + v.z * m[11] + m[15];
      
      return make_float4(x, y, z, w);
    }

    matrix4f operator*(const matrix4f& n) const
    {
      const matrix4f& m = *this;
      matrix4f out;
      for(int row = 0;row <4;row++)
      {
        for(int col = 0;col<4;col++)
        {
          out(row,col) = m(row,0) * n(0,col) + m(row,1) * n(1,col) + m(row,2) * n(2,col) + m(row,3) * n(3,col);
        }
      }
      
      return out;
    }

    __host__ __device__
    static matrix4f projectionMatrix(float n, float f, float right, float top)
    {
      matrix4f proj;
      float f_n = f - n;
      proj(0,0) = n/right;
      proj(1,1) = n/top;
      proj(2,2) = - ( f + n ) / f_n;
      proj(2,3) = -2.0f * f * n / f_n;
      proj(3,3) = 0.0f;
      proj(3,2) = -1.0f;
      return proj;
    }

    /*__host__ __device__
    static matrix4f invProjectionMatrix(float near, float far, float right, float top)
    {
      matrix4f proj;
//      float f_n = 2 *far * near;
      proj(0,0) = right / near;
      proj(1,1) = top / near;
      proj(2,2) = 0;
      proj(3,2) = (far-near)/(-2.f*far*near);
      proj(3,3) = (far+near)/(2.0f*far*near);
      proj(2,3) = -1.f;
      return proj;
    }*/

    static matrix4f translationnMatrix(float x, float y, float z)
    {
      matrix4f t;
      t(0,3) = x;
      t(1,3) = y;
      t(2,3) = z;
      return t;
    }

    static matrix4f rotationMatrix(float x, float y, float z)
    {
      x *= 3.1415f / 180.0f;
      y *= 3.1415f / 180.0f;
      z *= 3.1415f / 180.0f;

      matrix4f X;
      X(1,1) = cos(x);
      X(2,1) = sin(x);
      X(1,2) = -sin(x);
      X(2,2) = cos(x);

      matrix4f Y;
      Y(0,0) = cos(y);
      Y(2,0) = -sin(y);
      Y(0,2) = sin(y);
      Y(2,2) = cos(y);

      matrix4f Z;
      Z(0,0) = cos(z);
      Z(1,0) = sin(z);
      Z(0,1) = -sin(z);
      Z(1,1) = cos(z);
      
      return Z*Y*X;
    }

    std::ostream& toString(std::ostream & out) const
    {
      const matrix4f& mat = *this;
      out << "[" << mat(0,0) << " " << mat(0,1) << " " << mat(0,2) << " " << mat(0,3) << "]" << std::endl;
      out << "[" << mat(1,0) << " " << mat(1,1) << " " << mat(1,2) << " " << mat(1,3) << "]" << std::endl;
      out << "[" << mat(2,0) << " " << mat(2,1) << " " << mat(2,2) << " " << mat(2,3) << "]" << std::endl;
      out << "[" << mat(3,0) << " " << mat(3,1) << " " << mat(3,2) << " " << mat(3,3) << "]";
      return out;
    }
  };
}

std::ostream& operator<< (std::ostream &o, const Math::matrix4f &a);
