#pragma once

// https://gist.github.com/mre/1392067
template<typename T, typename K>
__device__ void bitonic_sort_step(T* dev_values, K* val2, int j, int k, int i)
{
  unsigned int ixj; /* Sorting partners: i and ixj */
  ixj = i^j;
 
  /* The threads with the lowest ids sort the array. */
  if ((ixj) > i) 
  {
    if ((i&k) == 0) 
    {
      /* Sort ascending */
      if (dev_values[i] > dev_values[ixj]) 
      {
        swap(&dev_values[i], &dev_values[ixj]);
        swap(&val2[i], &val2[ixj]);
      }
    }
    if ((i&k)!=0) 
    {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) 
      {
        swap(&dev_values[i], &dev_values[ixj]);
        swap(&val2[i], &val2[ixj]);
      }
    }
  }
}