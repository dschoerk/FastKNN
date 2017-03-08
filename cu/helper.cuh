#ifndef HELPER_CU
#define HELPER_CU

#include <iostream>
#include <string>

#ifdef MAX
#undef MAX
#endif

#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

void CUDA_SAFE_FUN(cudaError_t e, int line, std::string file)
{
  if(e != cudaSuccess)
  {
    std::cout << "Cuda Error @line: " << line << " in " << file << " " << cudaGetErrorString(e) << std::endl;
  }
}

#ifdef _DEBUG
  #define CUDA_SAFE(arg) CUDA_SAFE_FUN(arg, __LINE__, __FILE__)
  #define CUDA_SAFE_KERNEL CUDA_SAFE_FUN(cudaPeekAtLastError(), __LINE__, __FILE__); CUDA_SAFE_FUN(cudaDeviceSynchronize(), __LINE__, __FILE__);
#else
  #define CUDA_SAFE
  #define CUDA_SAFE_KERNEL
#endif

template<typename T>
T* cudaMallocT(size_t size)
{
  T* ptr;
  CUDA_SAFE(cudaMalloc(&ptr, size * sizeof(T)));
  CUDA_SAFE(cudaMemset(ptr, 0, size * sizeof(T)));
  return ptr;
}

template<typename T>
T* cudaDevice2HostT(T* p_device, size_t size)
{
  T* ptr = new T[size];
  CUDA_SAFE(cudaMemcpy(&ptr[0], p_device, size * sizeof(T), cudaMemcpyDeviceToHost));
  return ptr;
}

void printDeviceInfo()
{
  // Number of CUDA devices
  int devCount = 0;
  cudaGetDeviceCount(&devCount);
  std::cout << "Found " << devCount << " Cuda devices" << std::endl;

  for (int i = 0; i < devCount; ++i)
  {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, i);
    
    std::cout << "# Device " << i << ": " << devProp.name << " Cuda Version " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "# Global: " << devProp.totalGlobalMem << " Shared Memory per block: " << devProp.sharedMemPerBlock/1024 << "kB" << std::endl;
    std::cout << "# Threads per block: " << devProp.maxThreadsPerBlock << std::endl;
	  std::cout << "# Threads per block: " << devProp.maxThreadsDim[0] << "x" << devProp .maxThreadsDim[1] << "x" << devProp .maxThreadsDim[2] << std::endl;
    std::cout << "# Global Memory: " << devProp.totalGlobalMem/1024/1024 << "Mb" << std::endl;
  }
}

template<typename T>
__device__
void swap(T* a, T* b)
{
  T tmp = *a;
  *a = *b;
  *b = tmp;
}

template<typename T>
__device__
void _swap(T& a, T& b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

struct mat3
{
  float m00, m01, m02, 
        m10, m11, m12, 
        m20, m21, m22;
};

__device__
void tsquare(const float3 v, mat3& m)
{
  m.m00 = v.x * v.x;
  m.m01 = v.x * v.y;
  m.m02 = v.x * v.z;
  m.m10 = v.y * v.x;
  m.m11 = v.y * v.y;
  m.m12 = v.y * v.z;
  m.m20 = v.z * v.x;
  m.m21 = v.z * v.y;
  m.m22 = v.z * v.z;
}

__device__
float determinant(const mat3& m)
{
  return m.m00 * m.m11 * m.m22 + 
         m.m01 * m.m12 * m.m20 + 
         m.m02 * m.m10 * m.m21 -
         m.m02 * m.m11 * m.m20 -
         m.m00 * m.m12 * m.m21 -
         m.m01 * m.m10 * m.m22;
}

__device__
void add(const mat3& m, mat3& o)
{
  o.m00 += m.m00;
  o.m01 += m.m01;
  o.m02 += m.m02;
  o.m10 += m.m10;
  o.m11 += m.m11;
  o.m12 += m.m12;
  o.m20 += m.m20;
  o.m21 += m.m21;
  o.m22 += m.m22;
}

__device__
void div(const float& d, mat3& o)
{
  o.m00 /= d;
  o.m01 /= d;
  o.m02 /= d;
  o.m10 /= d;
  o.m11 /= d;
  o.m12 /= d;
  o.m20 /= d;
  o.m21 /= d;
  o.m22 /= d;
}

#define NDIM 3
typedef float VALTYPE;

__device__
VALTYPE hypot2(VALTYPE x, VALTYPE y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

__device__
void tred2(VALTYPE V[NDIM][NDIM], VALTYPE d[NDIM], VALTYPE e[NDIM]) {

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int j = 0; j < NDIM; j++) {
    d[j] = V[NDIM-1][j];
  }

  // Householder reduction to tridiagonal form.

  for (int i = NDIM-1; i > 0; i--) {

    // Scale to avoid under/overflow.

    VALTYPE scale = 0.0;
    VALTYPE h = 0.0;
    for (int k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0) {
      e[i] = d[i-1];
      for (int j = 0; j < i; j++) {
        d[j] = V[i-1][j];
        V[i][j] = 0.0;
        V[j][i] = 0.0;
      }
    } else {

      // Generate Householder vector.

      for (int k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      VALTYPE f = d[i-1];
      VALTYPE g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (int j = 0; j < i; j++) {
        e[j] = 0.0;
      }

      // Apply similarity transformation to remaining columns.

      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        g = e[j] + V[j][j] * f;
        for (int k = j+1; k <= i-1; k++) {
          g += V[k][j] * d[k];
          e[k] += V[k][j] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (int j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      VALTYPE hh = f / (h + h);
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (int j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (int k = j; k <= i-1; k++) {
          V[k][j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1][j];
        V[i][j] = 0.0;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.

  for (int i = 0; i < NDIM-1; i++) {
    V[NDIM-1][i] = V[i][i];
    V[i][i] = 1.0;
    VALTYPE h = d[i+1];
    if (h != 0.0) {
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i+1] / h;
      }
      for (int j = 0; j <= i; j++) {
        VALTYPE g = 0.0;
        for (int k = 0; k <= i; k++) {
          g += V[k][i+1] * V[k][j];
        }
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    for (int k = 0; k <= i; k++) {
      V[k][i+1] = 0.0;
    }
  }
  for (int j = 0; j < NDIM; j++) {
    d[j] = V[NDIM-1][j];
    V[NDIM-1][j] = 0.0;
  }
  V[NDIM-1][NDIM-1] = 1.0;
  e[0] = 0.0;
} 

// Symmetric tridiagonal QL algorithm.

__device__
void tql2(VALTYPE V[NDIM][NDIM], VALTYPE d[NDIM], VALTYPE e[NDIM]) {

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  for (int i = 1; i < NDIM; i++) {
    e[i-1] = e[i];
  }
  e[NDIM-1] = 0.0;

  VALTYPE f = 0.0;
  VALTYPE tst1 = 0.0;
  VALTYPE eps = pow(2.0,-52.0);
  for (int l = 0; l < NDIM; l++) {

    // Find small subdiagonal element

    tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
    int m = l;
    while (m < NDIM) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }

    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.

    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)

        // Compute implicit shift

        VALTYPE g = d[l];
        VALTYPE p = (d[l+1] - g) / (2.0 * e[l]);
        VALTYPE r = hypot2(p,1.0);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        VALTYPE dl1 = d[l+1];
        VALTYPE h = g - d[l];
        for (int i = l+2; i < NDIM; i++) {
          d[i] -= h;
        }
        f = f + h;

        // Implicit QL transformation.

        p = d[m];
        VALTYPE c = 1.0;
        VALTYPE c2 = c;
        VALTYPE c3 = c;
        VALTYPE el1 = e[l+1];
        VALTYPE s = 0.0;
        VALTYPE s2 = 0.0;
        for (int i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p,e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);

          // Accumulate transformation.

          for (int k = 0; k < NDIM; k++) {
            h = V[k][i+1];
            V[k][i+1] = s * V[k][i] + c * h;
            V[k][i] = c * V[k][i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;

        // Check for convergence.

      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0;
  }
  
  // Sort eigenvalues and corresponding vectors.

  for (int i = 0; i < NDIM-1; i++) {
    int k = i;
    VALTYPE p = d[i];
    for (int j = i+1; j < NDIM; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (int j = 0; j < NDIM; j++) {
        p = V[j][i];
        V[j][i] = V[j][k];
        V[j][k] = p;
      }
    }
  }
}

__device__
void eigen_decomposition(VALTYPE A[NDIM][NDIM], VALTYPE V[NDIM][NDIM], VALTYPE d[NDIM]) {
  VALTYPE e[NDIM];
  for (int i = 0; i < NDIM; i++) {
    for (int j = 0; j < NDIM; j++) {
      V[i][j] = A[i][j];
    }
  }
  tred2(V, d, e);
  tql2(V, d, e);
}

__device__
void computeSplattingData(const unsigned numPoint, const unsigned* k_index, const float3* packedQuadtree, float3* normals, float4* splatAxis)
{

  // compute covariance matrix
  float3 mean = { 0, 0, 0 };
  for (int i = 0; i < numPoint; i++)
  {
    int v = k_index[i];
    float3 p = g_packedQuadtree[v];
    mean += p;
  }
  mean /= (float)(numPoint);

  float cov[3][3];
  float diff[64][3];
  for (int i = 0; i < numPoint; i++)
  {
    int v = k_index[i];
    float3 p = g_packedQuadtree[v];
    float3 d = p - mean;
    diff[i][0] = d.x;
    diff[i][1] = d.y;
    diff[i][2] = d.z;
  }

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      cov[i][j] = 0;
      for (int m = 0; m < numPoint; ++m)
      {
        cov[i][j] += diff[m][i] * diff[m][j];
      }
    }
  }

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      cov[i][j] /= (float)(numPoint - 1);
    }
  }

  float eigval[3];
  float eigvec[3][3];
  eigen_decomposition(cov, eigvec, eigval);

  float3 eigVec0 = { eigvec[0][0], eigvec[1][0], eigvec[2][0] };
  float3 eigVec1 = { eigvec[0][1], eigvec[1][1], eigvec[2][1] };
  float3 eigVec2 = { eigvec[0][2], eigvec[1][2], eigvec[2][2] };

  float majorLength = sqrt(eigval[2]);
  float minorLength = sqrt(eigval[1]);
  *normals = normalize(eigVec0);

  eigVec1 = normalize(eigVec1);
  eigVec2 = normalize(eigVec2);

  splatAxis->x = eigVec2.x * majorLength * 1.5f;
  splatAxis->y = eigVec2.y * majorLength * 1.5f;
  splatAxis->z = eigVec2.z * majorLength * 1.5f;
  splatAxis->w = minorLength / majorLength;

//  printf("%f %f %f\n", eigval[0], eigval[1], eigval[2]);

/*  float v = eigval[0] - sqrt(eigval[1] * eigval[1] + eigval[2] * eigval[2]) / 50.f;
  
  if (v > 0)
  {
    splatAxis->w *= 1.f - 10.f * v;
  }
  */
}

#endif