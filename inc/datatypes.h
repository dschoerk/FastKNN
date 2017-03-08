#pragma once


struct TransformationSettings
{
  TransformationSettings() : knn_kmin(8), knn_kmax(16), max_sort_num(64), screen_width(1024), screen_height(1024), screenbuffer_size(1024) {}

  int knn_kmin; // number of neighbors to search
  int knn_kmax;
  int max_sort_num; // maximum sortable number of neighbor candidates
  
  int screen_width; // render target pixel dimension
  int screen_height;
  
  int screenbuffer_size; // sidelength off the allocated quadtree buffer, must be square and sidelengths are power of 2
  int quadtree_size(){return (4*screenbuffer_size*screenbuffer_size-1)/3;}

  float initial_radius_multiplier;
  float iterative_radius_multiplier;

  std::string file;
};

struct LoggingData
{
  float transformationTime;
  float quadtreeTime;
  float candidateTime;
  float copyTime;
  float distanceTime;
  size_t processedPoints;
  size_t processedScreenspacePoints;

  float execTime(){return quadtreeTime+candidateTime+copyTime+distanceTime;}

  void print()
  {
    std::cout << "# of processed points: " << processedPoints << std::endl;
    std::cout << "# of processed points in screenspace: " << processedScreenspacePoints << " ( approx. " << processedScreenspacePoints / ((execTime())/1000.0f) << " points per second)" << std::endl;
    std::cout << "transform points: " << transformationTime << std::endl;
    std::cout << "quadtree        : " << quadtreeTime << std::endl;
    std::cout << "copy            : " << copyTime << std::endl;
    std::cout << "candidate       : " << candidateTime << std::endl;
    std::cout << "distance        : " << distanceTime << std::endl;
    std::cout << "Execution time  : " << execTime() << " (w/o transformation time)" << std::endl;
  }
};