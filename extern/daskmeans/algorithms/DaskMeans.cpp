#include "DaskMeans.h"
#include "../structure/KnnRes.h"
#include "../structure/Node.h"
#include "../utils/Utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace Utils;

DaskMeans::DaskMeans(int capacity, int max_iterations, float convergence_threshold)
    : KMeansBase(max_iterations, convergence_threshold) {
        this->capacity = capacity;
        data_index = nullptr;
        centroid_index = nullptr;
    }

DaskMeans::~DaskMeans() {
    delete data_index;
    delete centroid_index;
    data_index = nullptr;
    centroid_index = nullptr;
}

void DaskMeans::run() {
    int it = 0;     // iteration
    float start_time, end_time;
    start_time = clock();

    buildDataIndex(this->capacity);
    initializeCentroids();

    end_time = clock();
    init_time = float(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "build index and initialize in " << init_time << " s" << std::endl;

    // main loop
    do {
        start_time = clock();
        buildCentroidIndex();
        if (it == 0) {
            ub = std::vector<float>(k, std::numeric_limits<float>::max());
        } else {
            for (int i = 0; i < k; i++) {
                ub[i] = inner_bound[i] + centroid_list[i]->drift + centroid_list[i]->max_drift;
            }
        }
        setInnerBound();
        #pragma omp parallel
        {
            #pragma omp single
            assignLabels(*data_index->root, std::numeric_limits<float>::max());
        }
        updateCentroids();

        end_time = clock();
        runtime[it] = float(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "iter: " << it << ", runtime: " << runtime[it] << " s" << std::endl;
        it++;
    } while (!hasConverged() && it < max_iterations);

    // show total runtime
    float total_runtime = init_time;
    for (size_t i = 0; i < max_iterations; i++) {
        total_runtime += runtime[i];
    }
    std::cout << "iter: " << it << ", pruned totally: " << pruned_point / 1000000 << std::endl;
    std::cout << "successfully run Dask-means in " << total_runtime << " s" << std::endl;
}

void DaskMeans::output(const std::string& file_path) {
    // std::ofstream file(file_path, std::ios::app);
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    for (auto centroid : centroid_list) {
        Cluster* clus = centroid->getCluster();
        file << "{cluster_id: " << clus->getClusterId() << ", point_id_list: [";

        std::vector<int> data_id_list = clus->getAllDataId();
        for (size_t i = 0; i < data_id_list.size(); ++i) {
            file << data_id_list[i];
            if (i < data_id_list.size() - 1) {
                file << ", ";
            }
        }
        file << "]}" << std::endl;
    }

    file.close();
}

void DaskMeans::buildDataIndex(int capacity) {
    data_index = new BallTree(capacity, data_scale);
    data_index->buildBallTree(dataset, data_scale);
}

void DaskMeans::buildCentroidIndex(int capacity) {
    if (centroid_index != nullptr) {
        delete centroid_index;
        centroid_index = nullptr;
    }
    centroid_index = new BallTree(capacity);
    centroid_index->buildBallTree(centroid_list, k);
}

void DaskMeans::setInnerBound() {
    // using index ball-tree knn to set inner bound
    this->inner_bound = std::vector<float>(k, -1.0);
    this->inner_id = std::vector<float>(k, -1.0);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < k; i++) {
        std::vector<KnnRes*> res(2);
        for (int j = 0; j < 2; j++) {
            res[j] = new KnnRes(ub[i]);
        }
        ballTree2nn(centroid_list[i]->coordinate, *(centroid_index->root), res, centroid_list);
        inner_bound[i] = res[1]->dis;
        inner_id[i] = res[1]->id;
        for (int j = 0; j < 2; j++) {
            delete res[j];
        }
    }
}

void DaskMeans::assignLabels(Node& node, float ub) {
    // 1. if the node is assigned before (pruning 1)
    if (node.centroid_id != -1  && 
        distance1(node.pivot, centroid_list[node.centroid_id]->getCoordinate()) 
        + node.radius < inner_bound[node.centroid_id] / 2) 
        { return; }

    // 2. find two nearest centroid for the node (pruning 2)
    std::vector<KnnRes*> res(2);
    for (int i = 0; i < 2; i++) {
        res[i] = new KnnRes(ub);
    }
    ballTree2nn(node.pivot, *(centroid_index->root), res, centroid_list);
    if (res[1]->dis - res[0]->dis > 2 * node.radius) {
        if (res[0]->id == node.centroid_id) {
            #pragma omp atomic
            pruned_point += node.point_number;
            return;
        }
        assignToCluster(node, res[0]->id);
        Cluster* new_cluster = centroid_list[node.centroid_id]->cluster;
        new_cluster->dataIn(node.point_number, node.sum_vector);
        #pragma omp atomic
        pruned_point += node.point_number;
        return;
    }

    if (!node.leaf) {
        // 3. split the node into two child node
        assignToCluster(node, -1);
        Node* lc = node.leftChild;
        Node* rc = node.rightChild;
        float child_ub = res[1]->dis + node.radius;
        if (lc != nullptr) {
            #pragma omp task firstprivate(lc, child_ub) if (node.point_number > 4096)
            assignLabels(*lc, child_ub);
        }
        if (rc != nullptr) {
            #pragma omp task firstprivate(rc, child_ub) if (node.point_number > 4096)
            assignLabels(*rc, child_ub);
        }
    } else {
        if (node.centroid_id != -1) {
            Cluster* old_cluster = centroid_list[node.centroid_id]->cluster;
            old_cluster->dataOut(node.point_number, node.sum_vector);
        }
        // 4. assign each point in leaf node
        for (int i = 0; i < node.point_number; i++) {
            std::vector<float> data = dataset[node.data_id_list[i]];
            int centroid_id = node.centroid_id_for_data[i];
            // if the point is assigned before
            if (centroid_id != -1 && distance1(data, centroid_list[centroid_id]->coordinate) 
                < inner_bound[centroid_id] / 2 ) {
                    #pragma omp atomic
                    pruned_point += 1;
                    return;
                }

            // use 1nn
            KnnRes* res = new KnnRes(ub);
            ballTree1nn(data, *(centroid_index->root), *res, centroid_list);
            if (centroid_id != -1) {
                Cluster* old_cluster = centroid_list[centroid_id]->cluster;
                old_cluster->dataOut(1, data);
            }
            centroid_id = res[0].id;
            Cluster* new_cluster = centroid_list[centroid_id]->cluster;
            new_cluster->dataIn(1, data);

            delete res;
        }
    }
    for (int i = 0; i < 2; i++) {
        delete res[i];
    }
}

void DaskMeans::updateCentroids() {
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < k; c++) {
        Cluster* cluster = centroid_list[c]->cluster;
        if (cluster->point_number != 0) {
            std::vector<float> new_coordinate =
                divideVector(cluster->sum_vec, cluster->point_number);
            centroid_list[c]->updateCoordinate(new_coordinate);
        }
    }
}

// assign a node and its child node to a centroid,
// and remove all nodes from previous centroids is exist
void DaskMeans::assignToCluster(Node& node, int centroid_id) {
    if (node.centroid_id != -1) {
        Cluster* old_cluster = centroid_list[node.centroid_id]->cluster;
        old_cluster->dataOut(node.point_number, node.sum_vector);
    }

    node.centroid_id = centroid_id;
    if (node.leaf) {
        for (int i = 0; i < node.point_number; i++) {
            int clu_id = node.centroid_id_for_data[i];
            if (clu_id != -1) {
                Cluster* old_cluster = centroid_list[clu_id]->cluster;
                old_cluster->dataOut(1, dataset[node.data_id_list[i]]);
            }
            node.centroid_id_for_data[i] = centroid_id;
        }
    } else {
        if (node.leftChild != nullptr) {
            assignToCluster(*node.leftChild, centroid_id);
        }
        if (node.rightChild != nullptr) {
            assignToCluster(*node.rightChild, centroid_id);
        }
    }
}