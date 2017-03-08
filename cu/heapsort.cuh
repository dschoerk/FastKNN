#pragma once

#include "sort_str.cuh"
#include "helper.cuh"

__device__
void heap_drown(sort_str* key, int i, int n)
{
  while(2*i <= n)
  {
    int j = 2*i;
    if(j < n && key[j-1] < key[j+1-1])
    {
      j++; // heap[j] is the greater child
    }

    if(key[i-1] < key[j-1])
    {
      swap(&key[i-1], &key[j-1]);
      i = j;
    }
    else
      return;
  }
}

__device__
void heap_drown(sort_str& A, sort_str& B, sort_str& C,sort_str& D,sort_str& E,sort_str& F,sort_str& G,sort_str& H)
{
  if(B > C)
  {
    if(B > A)
    {
      _swap(A,B);
      if(D > E)
      {
        if(D > B)
        {
          _swap(D,B);
          if(H > D)
            _swap(H,D);
        }
      }
      else
      {
        if(E > B)
        {
          _swap(E,B);
        }
      }
    }
  }
  else
  {
    if(C > A)
    {
      _swap(A,C);
      if(F > G)
      {
        if(F > C)
          _swap(F,C);
      }
      else
      {
        if(G > C)
          _swap(G,C);
      }
    }
  }

/*  if(A < B)
  {
    if(B < C)
    {
      swap(C,A);
      if(C < F)
      {
        if(F < G)
        {
          swap(G,C);
        }
        else
        {
          swap(F,C);
        }
      }
    }
    else
    {
      swap(B,A);
      if(B < D)
      {
        if(D < E)
        {
          swap(E,B);
        }
        else
        {
          swap(D,B);
          if(D < H)
          {
            swap(D,H);
          }
        }
      }
    }
  }*/
}