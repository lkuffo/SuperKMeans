// utils.cpp
#include "Utils.h"

namespace Utils {
    float distance1(const std::vector<float>& a, const std::vector<float>& b) {
        return std::sqrt(distance2(a, b));
    }

    float distance2(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension.");
        }
        float sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
    }

    std::vector<float> sumVectorsInDataset(const std::vector<std::vector<float>>& dataset) {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset must not be empty.");
        }
        
        size_t data_scale = dataset.size();
        size_t data_dimension = dataset[0].size();
        std::vector<float> result(data_dimension, 0.0);

        for (size_t i = 0; i < data_scale; ++i) {
            if (dataset[i].size() != data_dimension) {
                throw std::invalid_argument("All vectors in the dataset must have the same dimension.");
            }
            for (size_t j = 0; j < data_dimension; ++j) {
                result[j] += dataset[i][j];
            }
        }
        return result;
    }

    std::vector<float> sumVectorsInDataset(const std::vector<std::vector<float>>& dataset, std::vector<int>& point_id_list) {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset must not be empty.");
        }
        
        size_t data_scale = dataset.size();
        size_t data_dimension = dataset[0].size();
        std::vector<float> result(data_dimension, 0.0);

        for (size_t i : point_id_list) {
            if (dataset[i].size() != data_dimension) {
                throw std::invalid_argument("All vectors in the dataset must have the same dimension.");
            }
            for (size_t j = 0; j < data_dimension; ++j) {
                result[j] += dataset[i][j];
            }
        }
        return result;
    }

    std::vector<float> sumVectorsInDataset(std::vector<Centroid*>& centroid_list) {
        if (centroid_list.empty()) {
            throw std::invalid_argument("Dataset must not be empty.");
        }
        
        size_t data_dimension = centroid_list[0]->coordinate.size();
        std::vector<float> result(data_dimension, 0.0);

        for (auto centroid : centroid_list) {
            if (centroid->coordinate.size() != data_dimension) {
                throw std::invalid_argument("All vectors in the dataset must have the same dimension.");
            }
            for (size_t j = 0; j < data_dimension; ++j) {
                result[j] += centroid->coordinate[j];
            }
        }
        return result;
    }

    std::vector<float> sumVectorsInDataset(std::vector<Centroid*>& centroid_list, std::vector<int>& centroid_id_list) {
        if (centroid_list.empty()) {
            throw std::invalid_argument("Dataset must not be empty.");
        }
        
        size_t data_dimension = centroid_list[0]->coordinate.size();
        std::vector<float> result(data_dimension, 0.0);

        for (int centroid_id : centroid_id_list) {
            if (centroid_list[centroid_id]->coordinate.size() != data_dimension) {
                throw std::invalid_argument("All vectors in the dataset must have the same dimension.");
            }
            for (size_t j = 0; j < data_dimension; ++j) {
                result[j] += centroid_list[centroid_id]->coordinate[j];
            }
        }
        return result;
    }

    std::vector<float> addVector(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension.");
        }
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    std::vector<float> subtractVector(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension.");
        }
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    std::vector<float> divideVector(const std::vector<float>& v, float c) {
        if (c == 0) {
            throw std::invalid_argument("Scaling factor cannot be zero.");
        }
        std::vector<float> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i] / c;
        }
        return result;
    }

    std::vector<float> multiplyVector(const std::vector<float>& v, float c) {
        std::vector<float> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = v[i] * c;
        }
        return result;
    }

    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
        const std::vector<std::vector<float>>& dataset, int data_scale) {

        std::vector<int> point_ids(2, 0);       // [farthest point id, second ... id]
        std::vector<float> distances(2, 0.0);  // [farthest distance, second ...]
        for (int point_id = 0; point_id < data_scale; point_id++) {
            float dis = distance1(center, dataset[point_id]);
            if (dis >= distances[0]) {
                point_ids[0] = point_id;
                distances[0] = dis;
            } 
        }
        for (int point_id = 0; point_id < data_scale; point_id++) {
            if (point_id == point_ids[0])
                continue;
            float dis = distance1(dataset[point_ids[0]], dataset[point_id]);
            if (dis >= distances[1]) {
                point_ids[1] = point_id;
                distances[1] = dis;
            } 
        }
        return point_ids;
    }

    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
        const std::vector<std::vector<float>>& dataset, std::vector<int>& point_id_list) {

        std::vector<int> point_ids(2, 0);       // [farthest point id, second ... id]
        std::vector<float> distances(2, 0.0);  // [farthest distance, second ...]
        for (int point_id : point_id_list) {
            float dis = distance1(center, dataset[point_id]);
            if (dis >= distances[0]) {
                point_ids[0] = point_id;
                distances[0] = dis;
            } 
        }
        for (int point_id : point_id_list) {
            if (point_id == point_ids[0])
                continue;
            float dis = distance1(dataset[point_ids[0]], dataset[point_id]);
            if (dis >= distances[1]) {
                point_ids[1] = point_id;
                distances[1] = dis;
            }
        }
        return point_ids;
    }

    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
        std::vector<Centroid*>& centroid_list, int data_scale) {

        std::vector<int> point_ids(2, 0);       // [farthest point id, second ... id]
        std::vector<float> distances(2, 0.0);  // [farthest distance, second ...]
        for (int point_id = 0; point_id < data_scale; point_id++) {
            float dis = distance1(center, centroid_list[point_id]->coordinate);
            if (dis >= distances[0]) {
                point_ids[0] = point_id;
                distances[0] = dis;
            } 
        }
        for (int point_id = 0; point_id < data_scale; point_id++) {
            if (point_id == point_ids[0])
                continue;
            float dis = distance1(centroid_list[point_ids[0]]->coordinate, centroid_list[point_id]->coordinate);
            if (dis >= distances[1]) {
                point_ids[1] = point_id;
                distances[1] = dis;
            }
        }
        return point_ids;
    }

    std::vector<int> getTwoFarthestPoints(const std::vector<float>& center, 
        std::vector<Centroid*>& centroid_list, std::vector<int>& centroid_id_list) {

        std::vector<int> point_ids(2, 0);       // [farthest point id, second ... id]
        std::vector<float> distances(2, 0.0);  // [farthest distance, second ...]
        for (int point_id : centroid_id_list) {
            float dis = distance1(center, centroid_list[point_id]->coordinate);
            if (dis >= distances[0]) {
                point_ids[0] = point_id;
                distances[0] = dis;
            } 
        }
        for (int point_id : centroid_id_list) {
            if (point_id == point_ids[0])
                continue;
            float dis = distance1(centroid_list[point_ids[0]]->coordinate, centroid_list[point_id]->coordinate);
            if (dis >= distances[1]) {
                point_ids[1] = point_id;
                distances[1] = dis;
            }
        }
        return point_ids;
    }

    void ballTree1nn(std::vector<float> point, Node& root, KnnRes& res, 
        const std::vector<std::vector<float>>& dataset) {
        if (root.leaf) {
            for (int id : root.data_id_list) {
                float distance = distance1(point, dataset[id]);
                if (distance >= res.dis)
                    continue;
                
                res.dis = distance;
                res.id = id;
            }
            return;
        }
        // if is not leaf node, check whether the child nodes in the search area
        float left_dis = distance1(point, root.leftChild->pivot);
        if (res.id == -1 || res.dis > left_dis - root.leftChild->radius) {
            ballTree1nn(point, *(root.leftChild), res, dataset);
        }
        float right_dis = distance1(point, root.rightChild->pivot);
        if (res.id == -1 || res.dis > right_dis - root.rightChild->radius) {
            ballTree1nn(point, *(root.rightChild), res, dataset);
        }
    }

    void ballTree1nn(std::vector<float> point, Node& root, KnnRes& res, 
        std::vector<Centroid*>& centroid_list) {
        if (root.leaf) {
            for (int id : root.data_id_list) {
                float distance = distance1(point, centroid_list[id]->coordinate);
                if (distance >= res.dis)
                    continue;
                
                res.dis = distance;
                res.id = id;
            }
            return;
        }
        // if is not leaf node, check whether the child nodes in the search area
        float left_dis = distance1(point, root.leftChild->pivot);
        if (res.id == -1 || res.dis > left_dis - root.leftChild->radius) {
            ballTree1nn(point, *(root.leftChild), res, centroid_list);
        }
        float right_dis = distance1(point, root.rightChild->pivot);
        if (res.id == -1 || res.dis > right_dis - root.rightChild->radius) {
            ballTree1nn(point, *(root.rightChild), res, centroid_list);
        }
    }

    void ballTree2nn(std::vector<float> point, Node& root, std::vector<KnnRes*>& res, 
        const std::vector<std::vector<float>>& dataset) {
        if (root.leaf) {
            for (int id : root.data_id_list) {
                float distance = distance1(point, dataset[id]);
                if (res[0]->id == -1) {
                    res[0]->dis = distance;
                    res[0]->id = id;
                } else if (distance < res[0]->dis) {
                    res[1]->id = res[0]->id;
                    res[1]->dis = res[0]->dis;
                    res[0]->id = id;
                    res[0]->dis = distance;
                } else if (distance < res[1]->dis) {
                    res[1]->id = id;
                    res[1]->dis = distance;
                }
            }
            return;
        }
        // if is not leaf node, check whether the child nodes in the search area
        float left_dis = distance1(point, root.leftChild->pivot);
        if (res[1]->id == -1 || res[1]->dis > left_dis - root.leftChild->radius) {
            ballTree2nn(point, *(root.leftChild), res, dataset);
        }
        float right_dis = distance1(point, root.rightChild->pivot);
        if (res[1]->id == -1 || res[1]->dis > right_dis - root.rightChild->radius) {
            ballTree2nn(point, *(root.rightChild), res, dataset);
        }
    }

    void ballTree2nn(std::vector<float> point, Node& root, std::vector<KnnRes*>& res, 
        std::vector<Centroid*>& centroid_list) {
        if (root.leaf) {
            for (int id : root.data_id_list) {
                float distance = distance1(point, centroid_list[id]->coordinate);
                if (res[0]->id == -1) {
                    res[0]->dis = distance;
                    res[0]->id = id;
                } else if (distance < res[0]->dis) {
                    res[1]->id = res[0]->id;
                    res[1]->dis = res[0]->dis;
                    res[0]->id = id;
                    res[0]->dis = distance;
                } else if (distance < res[1]->dis) {
                    res[1]->id = id;
                    res[1]->dis = distance;
                }
            }
            return;
        }
        // if is not leaf node, check whether the child nodes in the search area
        float left_dis = distance1(point, root.leftChild->pivot);
        if (res[1]->id == -1 || res[1]->dis > left_dis - root.leftChild->radius) {
            ballTree2nn(point, *(root.leftChild), res, centroid_list);
        }
        float right_dis = distance1(point, root.rightChild->pivot);
        if (res[1]->id == -1 || res[1]->dis > right_dis - root.rightChild->radius) {
            ballTree2nn(point, *(root.rightChild), res, centroid_list);
        }
    }

    void calculate1nn(std::vector<float> point, KnnRes& res, 
        const std::vector<std::vector<float>>& dataset) {
            for (int i = 0; i < dataset.size(); i++) {
                float dis = distance1(point, dataset[i]);
                if (dis <= res.dis) {
                    res.dis = dis;
                    res.id = i;
                }
            }
        }

    void calculate1nn(std::vector<float> point, KnnRes& res, 
        std::vector<Centroid*>& centroid_list) {
            for (int i = 0; i < centroid_list.size(); i++) {
                float dis = distance1(point, centroid_list[i]->coordinate);
                if (dis <= res.dis) {
                    res.dis = dis;
                    res.id = i;
                }
            }
        }

    void calculate2nn(std::vector<float> point, std::vector<KnnRes*>& res, 
        const std::vector<std::vector<float>>& dataset) {
            for (int i = 0; i < dataset.size(); i++) {
                float dis = distance1(point, dataset[i]);
                if (dis <= res[0]->dis) {
                    res[1]->dis = res[0]->dis;
                    res[1]->id = res[0]->id;
                    res[0]->dis = dis;
                    res[0]->id = i;
                } else if (dis < res[1]->dis) {
                    res[1]->dis = dis;
                    res[1]->id = i;
                }
            }
        }

    void calculate2nn(std::vector<float> point, std::vector<KnnRes*>& res, 
        std::vector<Centroid*>& centroid_list) {
            for (int i = 0; i < centroid_list.size(); i++) {
                float dis = distance1(point, centroid_list[i]->coordinate);
                if (dis <= res[0]->dis) {
                    res[1]->dis = res[0]->dis;
                    res[1]->id = res[0]->id;
                    res[0]->dis = dis;
                    res[0]->id = i;
                } else if (dis < res[1]->dis) {
                    res[1]->dis = dis;
                    res[1]->id = i;
                }
            }
        }
    
    void kdTree2nn(std::vector<float> point, KdTreeNode& root, std::vector<KnnRes*>& res, 
            std::vector<Centroid*>& centroid_list) {
        if (root.leaf) {
            for (int id : root.data_id_list) {
                float distance = distance1(point, centroid_list[id]->coordinate);
                if (res[0]->id == -1) {
                    res[0]->dis = distance;
                    res[0]->id = id;
                } else if (distance < res[0]->dis) {
                    res[1]->id = res[0]->id;
                    res[1]->dis = res[0]->dis;
                    res[0]->id = id;
                    res[0]->dis = distance;
                } else if (distance < res[1]->dis) {
                    res[1]->id = id;
                    res[1]->dis = distance;
                }
            }
            return;
        }
        // if is not leaf node, check whether the child nodes in the search area
        int dim = root.current_dimension;
        if (point[dim] <= root.split_point[dim] || res[1]->id == -1) {
            kdTree2nn(point, *(root.leftChild), res, centroid_list);
        }
        if (point[dim] > root.split_point[dim] || res[1]->id == -1) {
            kdTree2nn(point, *(root.rightChild), res, centroid_list);
        }
    }

    int findBestDimension(const std::vector<std::vector<float>>& dataset, 
            const std::vector<int>& point_id_list) {
        int num_dimensions = dataset[0].size();
        std::vector<float> variances(num_dimensions, 0.0);

        // get variance in each dimension
        for (int d = 0; d < num_dimensions; ++d) {
            float mean = 0.0;
            int count = point_id_list.size();

            // mean
            for (int index : point_id_list) {
                mean += dataset[index][d];
            }
            mean /= count;

            // variance
            float variance = 0.0;
            for (int index : point_id_list) {
                variance += (dataset[index][d] - mean) * (dataset[index][d] - mean);
            }
            variance /= count;

            variances[d] = variance;
        }

        int best_dimension = 0;
        float max_variance = variances[0];
        for (int d = 1; d < num_dimensions; ++d) {
            if (variances[d] > max_variance) {
                max_variance = variances[d];
                best_dimension = d;
            }
        }

        return best_dimension;
    }

    int findBestDimension(std::vector<Centroid*>& centroid_list, 
            const std::vector<int>& centroid_id_list) {
        int num_dimensions = centroid_list[0]->coordinate.size();
        std::vector<float> variances(num_dimensions, 0.0);

        // get variance in each dimension
        for (int d = 0; d < num_dimensions; ++d) {
            float mean = 0.0;
            int count = centroid_id_list.size();

            // mean
            for (int index : centroid_id_list) {
                mean += centroid_list[index]->coordinate[d];
            }
            mean /= count;

            // variance
            float variance = 0.0;
            for (int index : centroid_id_list) {
                variance += (centroid_list[index]->coordinate[d] - mean) 
                            * (centroid_list[index]->coordinate[d] - mean);
            }
            variance /= count;

            variances[d] = variance;
        }

        int best_dimension = 0;
        float max_variance = variances[0];
        for (int d = 1; d < num_dimensions; ++d) {
            if (variances[d] > max_variance) {
                max_variance = variances[d];
                best_dimension = d;
            }
        }

        return best_dimension;
    }

    float mdistance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same size.");
        }

        float distance = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            distance += std::abs(a[i] - b[i]);
        }
        return distance;
    }

}
