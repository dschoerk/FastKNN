#pragma once

#include "matrix4f.h"

//__constant__ float view_matrix[16];
//__constant__ float projection_matrix[16];

__global__
void clearFbKernel(unsigned int* numBuffer, float* depthBuffer, float3* pointBuffer)
{
  const unsigned int idx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  
  depthBuffer[idx] = 0;
  numBuffer[idx] = 0;
}

__global__
  void transform_points_kernel(const int screen_width, const int screen_height, float3* points, unsigned int* numBuffer, float* depthBuffer, float3* pointBuffer,int num_vectors, int* mutex_ticket, int* mutex_turn, float v_matrix[16], float p_matrix[16])
{
  const unsigned int idx = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  float3 pv = Math::matrix4f::mul(v_matrix, points[idx]);
  float3 v = Math::matrix4f::mul(p_matrix, pv);//points[idx]; 
  
  //Math::matrix4f invProj = Math::matrix4f::invProjectionMatrix(1.5f,1000.0f,1.0f,1.0f);
  //float3 vv = Math::matrix4f::mul(invProj.data, pv);

  
  /*vv.x /= vv.w;
  vv.y /= vv.w;
  vv.z /= vv.w;
  vv.w /= vv.w;

  printf("%3.4f %3.4f %3.4f %3.4f - %3.4f %3.4f %3.4f %3.4f\n", pv.x, pv.y, pv.z, pv.w, vv.x, vv.y, vv.z, vv.w);
  
*/

  //printf("%f %f %f\n", v.x, v.y, v.z);

  int x_coord = (int)((v.x+1)*(screen_width/2) + 0.5f);
  int y_coord = (int)((v.y+1)*(screen_height/2) + 0.5f);

  // Culling
  if(x_coord < 0 || x_coord >= screen_width || y_coord < 0 || y_coord >= screen_height)
    return;

  unsigned int i = x_coord * screen_width + y_coord;

  // Perform depth test
  float depth = v.z;

  // lock buffer access
  // create a mutex to synchronize access to each pixel position
  // http://arxiv.org/pdf/1110.4623.pdf
  // using a spin-lock mutex, see Appendix alogrithm 2
  // FA Mutex lock - does only work for some pixels?????????? - sometimes no unlock, endless loop
  // FA Mutex would use only one atomic function, but needs two mutex variables per pixel instead of one

  while(atomicExch(&mutex_ticket[i], 1) != 0); // spin lock mutex
  //int ticket = atomicAdd(&mutex_ticket[i], 1); // FA Mutex lock
  //while(mutex_turn[i] != ticket);
  
  
  if(depthBuffer[i] == 0.0f || depth < depthBuffer[i])
  {
    // set the depth buffer pixel if we have a vertex with a lower depth
    // or if buffer_depth == 0.0f, which means it's not initialized yet.
    depthBuffer[i] = depth;
    pointBuffer[i] = pv;
    numBuffer[i] = 1;

    //surf2Dwrite<float3>(pv, surf_pointBuffer, x_coord * sizeof(float3), y_coord);

    
  }

  // unlock buffer access
  atomicExch ( &mutex_ticket[i] , 0 ) ;
  //mutex_turn[i]++;
}