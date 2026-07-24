// Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include "../structure/Centroid.h"
#include "../structure/Node.h"
#include "../structure/KnnRes.h"
#include "../structure/KdTreeNode.h"

namespace Utils {
    // get the distance of two vectors
    float distance1(const std::vector<float>& a, const std::vector<float>& b);

    // get the square of the distance of two vectors
    float distance2(const std::vector<float>& a, const std::vector<float>& b);

    // get the sum of all vectors in dataset
    std::vector<float> sumVectorsInDataset(const std::vector<std::vector<float>>& dataset);
    std::vector<float> sumVectorsInDataset(const std::vector<std::vector<float>>& dataset, std::vector<int>& point_id_list);
    std::vector<float> sumVectorsInDataset(std::vector<Centroid*>& centroid_list);
    std::vector<float> sumVectorsInDataset(std::vector<Centroid*>& centroid_list, std::vector<int>& centroid_id_list);

    // get the sum vector
    std::vector<float> addVector(const std::vector<float>& a, const std::vector<float>& b);

    // get the result of (a - b)
    std::vector<float> subtractVector(const std::vector<float>& a, const std::vector<float>& b);

    // divide a vector by a constant
    std::vector<float> divideVector(const std::vector<float>& v, float c);

    // multiply a vector by a constant
    std::vector<float> multiplyVector(const std::vector<float>& v, float c);

    // find two farthest point to the center in dataset
    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
            const std::vector<std::vector<float>>& dataset, int data_scale);
    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
            const std::vector<std::vector<float>>& dataset, std::vector<int>& point_id_list);
    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
            std::vector<Centroid*>& centroid_list, int data_scale);
    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
            std::vector<Centroid*>& centroid_list, std::vector<int>& centroid_id_list);

    // ball-tree knn
    void ballTree1nn(std::vector<float> point, Node& root, KnnRes& res, 
            const std::vector<std::vector<float>>& dataset);
    void ballTree1nn(std::vector<float> point, Node& root, KnnRes& res, 
            std::vector<Centroid*>& centroid_list);
    void ballTree2nn(std::vector<float> point, Node& root, std::vector<KnnRes*>& res, 
            const std::vector<std::vector<float>>& dataset);
    void ballTree2nn(std::vector<float> point, Node& root, std::vector<KnnRes*>& res, 
            std::vector<Centroid*>& centroid_list);

    // knn that uses simple calculation, storing the result in res
    void calculate1nn(std::vector<float> point, KnnRes& res, 
            const std::vector<std::vector<float>>& dataset);
    void calculate1nn(std::vector<float> point, KnnRes& res, 
            std::vector<Centroid*>& centroid_list);
    void calculate2nn(std::vector<float> point, std::vector<KnnRes*>& res, 
            const std::vector<std::vector<float>>& dataset);
    void calculate2nn(std::vector<float> point, std::vector<KnnRes*>& res, 
            std::vector<Centroid*>& centroid_list);
    
    // kd-tree knn
    void kdTree2nn(std::vector<float> point, KdTreeNode& root, std::vector<KnnRes*>& res, 
            std::vector<Centroid*>& centroid_list);

    // find the dimension with the maximum variance
    int findBestDimension(const std::vector<std::vector<float>>& dataset, 
            const std::vector<int>& point_id_list);
    int findBestDimension(std::vector<Centroid*>& centroid_list, 
            const std::vector<int>& centroid_id_list);

    // manhattan mistance
    float mdistance(const std::vector<float>& a, const std::vector<float>& b);
}

#endif // UTILS_H
