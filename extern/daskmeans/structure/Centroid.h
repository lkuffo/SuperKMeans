// centroid.h
#ifndef CENTROID_H
#define CENTROID_H

#include <vector>
#include "Cluster.h"

class Centroid {
public:
    std::vector<float> coordinate;
    std::vector<float> old_coordinate; // coordinate from the last iteration
    int point_id;       // only used in initialization
    int centroid_id;
    Cluster* cluster;   // remember to release the memory

    // for dask-means
    float drift;
    float max_drift;

public:
    Centroid(int point_id, std::vector<float> coordinate, int centroid_id);

    ~Centroid();

    std::vector<float> getCoordinate();

    std::vector<float> getOldCoordinate();

    Cluster* getCluster();

    void updateCoordinate(std::vector<float> new_coordinate);
};

#endif // CENTROID_H