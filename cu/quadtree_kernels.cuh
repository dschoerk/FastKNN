#pragma once

__global__
  void quadtree_kernel(unsigned int* numBuffer, int size, int offset, unsigned int level) // size = 512 offset = 1024*1024
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= size*size)
    return;

  //muh<<<1,1>>>();

  const int x = idx / size;
  const int y = idx % size;
  
  unsigned int n = numBuffer[offset + (x*2) * 2*size + (y*2)];
  n += numBuffer[offset + (x*2) * 2*size + (y*2+1)];
  n += numBuffer[offset + (x*2+1) * 2*size + (y*2)];
  n += numBuffer[offset + (x*2+1) * 2*size + (y*2+1)];

/*  const int px = x * (1024>>(10-level));
  const int py = y * (1024>>(10-level));

  if(n >= 8 && minLevelBuffer[px * 1024 + py] == 0)
  {
    minLevelBuffer[px * 1024 + py] = 10-level;
    //printf("level %d %d @%d %d\n", level, n, px, py);
  }*/

  numBuffer[offset + 4*size*size + x*size+y] = n;
}

__global__
  void packQuadtreeIndices_kernel(unsigned int* numBuffer, unsigned int* packedQuadtreeIndices, int size, int offset)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size*size)
    return;

  const int x = idx / size;
  const int y = idx % size;
  
  const int l_x = x << 1;
  const int l_y = y << 1;
  
  const int l_size = size << 1;
  const int startLowerIdx = offset - size*size*4;

  /*if(idx == 0)
  {
    //printf("offset %d\n", startLowerIdx);
  }*/

  int a = packedQuadtreeIndices[offset + x * size + y];
  packedQuadtreeIndices[startLowerIdx + l_x * l_size + l_y] = a;

  a += numBuffer[startLowerIdx + l_x * l_size + l_y];
  packedQuadtreeIndices[startLowerIdx + (l_x+1) * l_size + l_y] = a;

  a += numBuffer[startLowerIdx + (l_x+1) * l_size + l_y];
  packedQuadtreeIndices[startLowerIdx + l_x * l_size + l_y+1] = a;

  a += numBuffer[startLowerIdx + l_x * l_size + l_y+1];
  packedQuadtreeIndices[startLowerIdx + (l_x+1) * l_size + l_y+1] = a;
}

__global__
  void packQuadtree_kernel(
    unsigned int* numBuffer, 
    unsigned int* packedQuadtreeIndices, 
    float3* pointBuffer, 
    float3* packedQuadtree, 
    float3* packedQuadtreeColors,
    float3* packedQuadtreeNormalsCorrectness,
    unsigned int* packedQuadtreeCoordinateIndex,
    int size)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size*size) // safety check, not necessary if kernel params scale
    return;

  const int x = idx / size;
  const int y = idx % size;


  float4 f2 = tex2D(texRef, x, y);
  if (f2.w > 0)
  //if(numBuffer[x * 1024 + y] == 1)
  {
    unsigned int targetIdx = packedQuadtreeIndices[x * 1024 + y];

    float3 out;
    out.x = f2.x;
    out.y = f2.y;
    out.z = f2.z;

    float4 n = tex2D(texRefNormals, x, y);
    float3 n3;
    n3.x = n.x;
    n3.y = n.y;
    n3.z = n.z;

    float4 c = tex2D(texRefColors, x, y);
    float3 col;
    col.x = c.x;
    col.y = c.y;
    col.z = c.z;

    //printf("%f %f %f\n", n3.x, n3.y, n3.z);

    packedQuadtree[targetIdx] = out;
    packedQuadtreeColors[targetIdx] = col;
    packedQuadtreeNormalsCorrectness[targetIdx] = n3; // real normal for correctness check
    packedQuadtreeCoordinateIndex[targetIdx] = x * 1024 + y;
  }

  /*if(x == 758 && y == 305)
  {
    //const int targetIdx = packedQuadtreeIndices[x * 1024 + y];
    //printf(">>>>> %d    %f %f %f\n", packedQuadtreeIndices[x * 1024 + y], packedQuadtree[targetIdx](0), packedQuadtree[targetIdx](1), packedQuadtree[targetIdx](2));
  }*/
}

__global__
void copyNums(unsigned int* numBuffer, unsigned size) // size = 512 offset = 1024*1024
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size * size)
    return;

  const int x = idx / size;
  const int y = idx % size;

  float4 read = tex2D(texRef, x, y);

  //if (read.w > 0)
  //  printf("%f %f %f\n", read.x, read.y, read.z);

  if (read.w > 0)
    numBuffer[x*size + y] = 1;
  else
    numBuffer[x*size + y] = 0;
}