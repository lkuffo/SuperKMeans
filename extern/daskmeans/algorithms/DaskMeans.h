// DaskMeans.h
#ifndef DASKMEANS_H
#define DASKMEANS_H

#include <vector>
#include "KMeansBase.h"
#include "../utils/Utils.h"
#include "../structure/BallTree.h"
#include "../structure/KnnRes.h"

using namespace Utils;

class DaskMeans : public KMeansBase {
protected:
    BallTree* data_index;
    BallTree* centroid_index;
    std::vector<float> inner_bound;
    std::vector<float> ub;
    int capacity;

    // for debug
    std::vector<float> inner_id;
    float pruned_point = 0.0;

public:
    DaskMeans(int capacity, int max_iterations = MAX_ITERATIONS, float convergence_threshold = 0.001);

    ~DaskMeans() override;

    void run() override;

    void output(const std::string& file_path) override;

protected:
    void buildDataIndex(int capacity = 1);

    void buildCentroidIndex(int capacity = 5);

    void setInnerBound();

    void assignLabels(Node& node, float ub);
    
    void updateCentroids() override;

    void assignToCluster(Node& node, int centroid_id);   
};

#endif