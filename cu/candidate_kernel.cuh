#pragma once

#include "quicksort.cuh"
#include "sort_str.cuh"
#include "cutil_math.h"
#include "heapsort.cuh"

#define STACKSIZE 100
#define KMAX 64

template<typename T>
struct bbox
{
  T l,r,b,t;

  __device__ 
  bbox(T _l, T _r, T _t, T _b) : l(_l), r(_r), b(_b), t(_t)
  {}

  __device__ bool intersect(const bbox& rhs) const
  {
    return !(rhs.l > r || 
             rhs.r < l || 
             rhs.t > b ||
             rhs.b < t);
  }

  __device__ bool contains(const bbox& rhs) const
  {
    return l<rhs.l && r>rhs.r && t<rhs.t && b>rhs.b;
  }

  __device__ bool contains(const int2& p) const
  {
    return l <= p.x && p.x <= r && b <= p.y && p.y <= t;
  }
};

template<typename T, int size>
struct Stack
{
  T stack[size];
  int stack_top;

  __device__ __forceinline__ Stack():stack_top(0){}

  __device__ __forceinline__ void push(T e)
  {
//    if(!(stack_top < size))
//      printf("STACK OVERFLOW!!!!!!!!!!!!\n");
    stack_top++;
    stack[stack_top] = e;
  }

  __device__ __forceinline__ T pop()
  {
    T e = stack[stack_top];
    stack_top--;
    return e;
  }

  __device__ __forceinline__ bool isEmpty()
  {
    return stack_top == 0;
  }
};

struct stack_args
{
  unsigned levelx;
  unsigned levely;
  unsigned level;
};

__device__ int correctNumPoints;

__global__ 
void showCorrectNumPoints(int num)
{
//  printf("Correct Num Points %f\n", correctNumPoints / (float)num);
  correctNumPoints = 0;
}

__device__
void push_sub_levels(Stack<stack_args, STACKSIZE>& stack, const stack_args& e, const bbox<unsigned>& fullBbox, const unsigned cx, const unsigned cy, const unsigned s)
{
  bbox<unsigned> sub(cx-s, cx-1, cy-s, cy-1);
  if(sub.intersect(fullBbox))
  {
    stack_args c = {2*e.levelx, 2*e.levely, e.level+1};
    stack.push(c);
  }

  sub.l = cx-s;
  sub.r = cx-1;
  sub.t = cy;
  sub.b = cy+s-1;
  if(sub.intersect(fullBbox))
  {
    stack_args c = {2*e.levelx, 2*e.levely+1,   e.level+1};
    stack.push(c);
  }

  sub.l = cx;
  sub.r = cx+s-1;
  sub.t = cy-s;
  sub.b = cy-1;
  if(sub.intersect(fullBbox))
  {
    stack_args c = {2*e.levelx+1, 2*e.levely, e.level+1};
    stack.push(c);
  }

  sub.l = cx;
  sub.r = cx+s-1;
  sub.t = cy;
  sub.b = cy+s-1;
  if(sub.intersect(fullBbox))
  {
    stack_args c = {2*e.levelx+1, 2*e.levely+1,   e.level+1};
    stack.push(c);
  }
}

__device__
bool isect(const bbox<int>& sub, const int x, const int y, const float r_2)
{
  int dx = min(abs(x - sub.l), abs(sub.r - x));
  int dy = min(abs(y - sub.t), abs(sub.b - y));
  //printf("%d %d\n", dx, dy);
  return dx*dx + dy*dy < r_2;
//  return false;
}

__device__
void push_sub_levelsEx(Stack<stack_args, STACKSIZE>& stack, const stack_args& e, const unsigned cx, const unsigned cy, const unsigned s, unsigned x, unsigned y, float r)
{


  bbox<int> sub(cx - s, cx - 1, cy - s, cy - 1);
  if ((x >= sub.l-r && x <= sub.r+r && y >= sub.t-r && y <= sub.b+r))
  {
    stack_args c = { 2 * e.levelx, 2 * e.levely, e.level + 1 };
    stack.push(c);
  }

  sub.l = cx - s;
  sub.r = cx - 1;
  sub.t = cy;
  sub.b = cy + s - 1;
  if ((x >= sub.l - r && x <= sub.r + r && y >= sub.t - r && y <= sub.b + r))
  {
    stack_args c = { 2 * e.levelx, 2 * e.levely + 1, e.level + 1 };
    stack.push(c);
  }

  sub.l = cx;
  sub.r = cx + s - 1;
  sub.t = cy - s;
  sub.b = cy - 1;
  if ((x >= sub.l - r && x <= sub.r + r && y >= sub.t - r && y <= sub.b + r))
  {
    stack_args c = { 2 * e.levelx + 1, 2 * e.levely, e.level + 1 };
    stack.push(c);
  }

  sub.l = cx;
  sub.r = cx + s - 1;
  sub.t = cy;
  sub.b = cy + s - 1;
  if ((x >= sub.l - r && x <= sub.r + r && y >= sub.t - r && y <= sub.b + r))
  {
    stack_args c = { 2 * e.levelx + 1, 2 * e.levely + 1, e.level + 1 };
    stack.push(c);
  }
}

/*__device__ int2 project2d(float4 viewspace, const float* projection)
{
   //float4 vs = Math::matrix4f::mul(projection, viewspace);
   int2 p = {((1.5f*viewspace.x/-viewspace.z)+1.f)*512.f+.5f, ((1.5f*viewspace.y/-viewspace.z)+1.f)*512.f+.5f};

   return p;
}*/

__device__ int findMinLevel(int x, int y, int kmin)
{
  int s = 1024;
  int level = 0;
  {
    int numOnLevel=0;
    unsigned offset=0;
    for(;s > 0; s>>=1)
    {
      int qidx = offset + (x>>level) * s + (y>>level);
      numOnLevel = g_numBuffer[qidx];

      if(numOnLevel >= kmin)
        break;

      offset += s*s;
      level++;
    }
  }

  return level;
}

__device__
float radiusEstimate(int numOnLevel, int target, int level)
{
  float numInDisc = (float)numOnLevel * 4.f / 3.14592f;
  float radius = (float)(1 << level) / 2.f;
  return radius * sqrtf((float)target / numInDisc);
}
 
__device__ float estimateKRadius(int x, int y, int kmin, float magic)
{
  int s = 1024;
  int level = 0;
  int numOnLevel=0; // number of pixels in the level
  unsigned offset=0;

  int target = kmin;// (KMAX + kmin) / 2;

  for(;s > 0; s>>=1)
  {
    int qidx = offset + (x>>level) * s + (y>>level);
    numOnLevel = g_numBuffer[qidx];

    if (numOnLevel >= target) // find the first level where we can get at least target candidates
    {
      //printf("%d %d %d %d %f\n", level, numOnLevel, 1 << level, (1 << level)*(1 << level), sqrtff((float)numOnLevel / (float)((1 << level)*(1 << level))));
      break;
    }

    offset += s*s;
    level++;
  }

  float maxNumCandidates = (1 << level)*(1 << level); // maximum possible number of candidates
  float candidates = numOnLevel; // actual number of candidates
  float fillRate = candidates / maxNumCandidates;

  if (fillRate > 0.785f)
    return 1;
  
//  float candidateRadius = sqrtff(maxNumCandidates / 3.141592f); // radius to cover candidates
//  float targetRadius = magic * sqrtf(maxNumCandidates / 3.141592f * (float)kmin / candidates);
  float targetRadius = sqrtf(maxNumCandidates / 3.141592f * (float)target / candidates);

//  printf("%f %f\n", targetRadius, oldRad);
//  return oldRad;
  return targetRadius;//radiusEstimate(numOnLevel, target, level);
}

__global__ 
void knn_kernel_bruteforce_k8(
const int n,
unsigned* k_index,
float3* normals,
float4* splatAxis)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  const float3 q = g_packedQuadtree[idx];
   
  sort_str A, B, C, D, E, F, G, H;
  A.setDistance(99999999);
  B.setDistance(99999999);
  C.setDistance(99999999);
  D.setDistance(99999999);
  E.setDistance(99999999);
  F.setDistance(99999999);
  G.setDistance(99999999);
  H.setDistance(99999999);
  A.setIdx(0);
  B.setIdx(0);
  C.setIdx(0);
  D.setIdx(0);
  E.setIdx(0);
  F.setIdx(0);
  G.setIdx(0);
  H.setIdx(0);

  for (int i = 0; i < n; i++)
  {
    float3 p = g_packedQuadtree[i];
    const float3 diff = p - q;
    const float distance_sq = dot(diff, diff);

    if (distance_sq < A.getDistance())
    {
      A.setDistance(distance_sq); // O(1)
      A.setIdx(i); // O(1)
      heap_drown(A, B, C, D, E, F, G, H);
    }
  }

  k_index[idx * KMAX + 0] = A.getIdx();
  k_index[idx * KMAX + 1] = B.getIdx();
  k_index[idx * KMAX + 2] = C.getIdx();
  k_index[idx * KMAX + 3] = D.getIdx();
  k_index[idx * KMAX + 4] = E.getIdx();
  k_index[idx * KMAX + 5] = F.getIdx();
  k_index[idx * KMAX + 6] = G.getIdx();
  k_index[idx * KMAX + 7] = H.getIdx();

  computeSplattingData(8, &k_index[idx * KMAX + 0], g_packedQuadtree, &normals[idx], &splatAxis[idx]);
}


__device__
void heapsort_target8(const unsigned idx, const unsigned validPoints, const unsigned kmin, unsigned* k_index, float* distances)
{
  // build the heap in registers
  sort_str A, B, C, D, E, F, G, H;
  A.setDistance(99999999);
  B.setDistance(99999999);
  C.setDistance(99999999);
  D.setDistance(99999999);
  E.setDistance(99999999);
  F.setDistance(99999999);
  G.setDistance(99999999);
  H.setDistance(99999999);
  A.setIdx(0);
  B.setIdx(0);
  C.setIdx(0);
  D.setIdx(0);
  E.setIdx(0);
  F.setIdx(0);
  G.setIdx(0);
  H.setIdx(0);

  for (int i = 0; i < validPoints; i++)
  {
    if (distances[i] < A.getDistance())
    {
      A.setDistance(distances[i]); 
      A.setIdx(k_index[idx * KMAX + i]);
      heap_drown(A, B, C, D, E, F, G, H); // drown element in heap
    }
  }

  k_index[idx * KMAX + 0] = A.getIdx();
  k_index[idx * KMAX + 1] = B.getIdx();
  k_index[idx * KMAX + 2] = C.getIdx();
  k_index[idx * KMAX + 3] = D.getIdx();
  k_index[idx * KMAX + 4] = E.getIdx();
  k_index[idx * KMAX + 5] = F.getIdx();
  k_index[idx * KMAX + 6] = G.getIdx();
  k_index[idx * KMAX + 7] = H.getIdx();
  
  //remove me
  distances[0] = A.getDistance();
}

__device__
void heapsort_targetn(const unsigned idx, const unsigned validPoints, const unsigned kmin, unsigned* k_index, const float* distances)
{
  // build the heap in registers
  sort_str v[64];
  for (unsigned i = 0; i < 64; ++i)
  {
    v[i].setDistance(99999999);
    v[i].setIdx(0);
  }

  for (int i = 0; i < validPoints; i++)
  {
    if (distances[i] < v[0].getDistance())
    {
      v[0].setDistance(distances[i]);
      v[0].setIdx(k_index[idx * KMAX + i]);
      heap_drown(v, 1, kmin); // drown element in heap
    }
  }

  for (unsigned i = 0; i < kmin; ++i)
    k_index[idx * KMAX + i] = v[i].getIdx();
}


__global__
void knn_kernel_quadtree_k8(
const int n, const unsigned maxIterations, const int kmin, const int traverseLevel, const float initial_radius_multiplier, const float iterative_radius_multiplier,
unsigned* packedCoordinates,
unsigned* k_index,
float3* normals,
float4* splatAxis,
float* perfectRadius, unsigned* validPointsBuf, float* distance_buffer)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  // estimate a minimal radius
  const int coordIdx = packedCoordinates[idx];
  const float3 q = g_packedQuadtree[idx];
  const int x = coordIdx / 1024;
  const int y = coordIdx % 1024;

  int level = findMinLevel(x, y, kmin);
  float kradiusEstimation = estimateKRadius(x, y, kmin, sqrtf(3.f));
  float screenspace_radius = initial_radius_multiplier * kradiusEstimation;// (float)(1<<level) * initial_radius_multiplier;

  float distances[64];
  float viewspace_radius = 0.0f;
  float last_viewspace_radius = 0.0f;
  unsigned validPoints = 0;
  Stack<stack_args, STACKSIZE> stack;
  unsigned run = 0;

  for (; run < maxIterations; run++)
  {
    
    viewspace_radius = screenspace_radius / 512.f * -q.z / 1.5f;

    //bbox<unsigned> outer(x - screenspace_radius, x + screenspace_radius, y - screenspace_radius, y + screenspace_radius);
    stack_args c = { 0, 0, 0 };
    stack.push(c);

    unsigned validOld = validPoints;

    while (!stack.isEmpty())
    {
      stack_args e = stack.pop();
      const unsigned offset = (4 * 1024 * 1024 - 1) / 3 - (4 * (1 << e.level) * (1 << e.level) - 1) / 3;
      const unsigned qidx = offset + e.levelx * (1 << e.level) + e.levely;
      const unsigned s = (1024 >> (e.level + 1));
      const unsigned cx = 2 * e.levelx*s + s;
      const unsigned cy = 2 * e.levely*s + s;
      const unsigned numPoints = g_numBuffer[qidx];

      if (numPoints < 1) // early termination
        continue;

      /*unsigned left = cx - s;
      unsigned right = cx + s - 1;
      unsigned top = cy - s;
      unsigned bottom = cy + s - 1;*/
      /*int dx = max(abs(x - left), abs(right - x));
      int dy = max(abs(y - top), abs(bottom - y));
      float lastScreenspaceRadius = last_viewspace_radius * 1.5f * 512.f / -q.z;

      if (dx*dx + dy*dy < lastScreenspaceRadius*lastScreenspaceRadius)
        continue;*/

      if (e.level < traverseLevel)
      {
        //push_sub_levels(stack, e, outer, cx, cy, s);
        
        push_sub_levelsEx(stack, e, cx, cy, s, x, y, screenspace_radius);
      }
      else
      {
        const unsigned start_idx = g_packedQuadtreeIndices[qidx];
        for (int i = 0; i < numPoints && validPoints < KMAX; i++)
        {
          float3 p = g_packedQuadtree[start_idx + i];
          const float3 diff = p - q;
          const float distance_sq = dot(diff, diff);
          
          if (distance_sq >= last_viewspace_radius*last_viewspace_radius && distance_sq < viewspace_radius*viewspace_radius)
          {
            k_index[idx * KMAX + validPoints] = start_idx + i;
            distances[validPoints] = distance_sq;
            validPoints++;
          }
        }
      }
    }

    if (validPoints >= KMAX)
    {
      validPoints = validOld; // go back to the previous solution
      // we now try to reduce the search radius
      // assuming we have a uniform point distribution and we have found 64 (or more) points, 
      // the number of found points is directly proportional to the covered area with a given radius
      // 
      screenspace_radius *= sqrtf((float)kmin*2.f / (float)KMAX);
      // assume we have 64 points, we possibly have more ..
      // to get a desired number of points we scale the radius down
      // we target kmin points, but we also want to avoid unnecessary iterations, 
      // so we don't target kmin points since we can possible overshoot our target, 
      // get less than kmin points and force another iteration
      // therefore we target kmin*2 points

      //validPoints = 0;
    }
    else if (validPoints < kmin)
    {
//      last_screenspace_radius = screenspace_radius;
      last_viewspace_radius = viewspace_radius;
      screenspace_radius *= sqrtf((float)kmin*2.f / (float)validPoints); // targetting 32 points

      //validPoints = 0;
    }
    else break;
  }



/*  if (run == 0)
    atomicAdd(&doneInFirstRound, 1);

  __syncthreads();

  if (idx == 0)
    printf("done in first iter %d from %d\n", doneInFirstRound, n);*/

/*  if (validPoints < kmin)
    printf("%d ", validPoints);

  if (validPoints >= kmax)
    printf("more ");*/

  if (validPoints >= kmin) // need sorting, but we can also accept less than kmin points
  {
    if (kmin == 8)
    {
      heapsort_target8(idx, validPoints, kmin, k_index, distances);
      validPoints = 8;
    }
    else
    {
      heapsort_targetn(idx, validPoints, kmin, k_index, distances);
      validPoints = kmin;
    }

    validPoints = kmin;

    //    printf("%f\n", kradiusEstimation / (sqrtff(distances[0]) * 1.5f / -q.z * 512.f));
  }
  
  
//  float screenspaceRadius = sqrtf(distances[0]) * 1.5f / -q.z * 512.f;

//  if (validPoints >= kmin && validPoints < KMAX )
//    atomicAdd(&correctNumPoints, 1);
  
  validPointsBuf[idx] = validPoints;

  //if (screenspaceRadius > 0.5f)
  //  computeSplattingData(validPoints, &k_index[idx * KMAX + 0], g_packedQuadtree, &normals[idx], &splatAxis[idx]);
  /*else
  {
    float4* axis = &splatAxis[idx];
    axis->x = 0;
    axis->y = 0;
    axis->z = 0;
    axis->w = 0;
  }*/
}

__global__
void computeSplattingVbos(const int n, const int kmin, const unsigned* k_index, const unsigned* validPoints, float3* normals, float4* splatAxis)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  computeSplattingData(validPoints[idx], &k_index[idx * KMAX + 0], g_packedQuadtree, &normals[idx], &splatAxis[idx]);
}
