#include "Cluster.h"
#include "../utils/Utils.h"
#include <algorithm>
#include <vector>

using namespace Utils;

Cluster::Cluster(int cluster_id, int data_dimension): cluster_id(cluster_id) {
    this->sum_vec = std::vector<float>(data_dimension, 0.0);
    omp_init_lock(&lock);
}

Cluster::~Cluster() { omp_destroy_lock(&lock); }

std::vector<int> Cluster::getDataIdList() {
    return data_id_list;
}

std::vector<int> Cluster::getAllDataId() {
    std::vector<int> all_data_id = data_id_list;
    for (auto node : node_list) {
        std::vector<int> data_in_node = node->getAllDataId();
        all_data_id.insert(all_data_id.end(), data_in_node.begin(), data_in_node.end());
    }
    return all_data_id;
}

void Cluster::addDataId(int data_id) {
    data_id_list.push_back(data_id);
}

int Cluster::getClusterId() {
    return cluster_id;
}

void Cluster::clear() {
    data_id_list.clear();
}

void Cluster::dataIn(std::vector<float> data_in, int data_id) {
    omp_set_lock(&lock);
    sum_vec = addVector(sum_vec, data_in);
    data_id_list.push_back(data_id);
    point_number += 1;
    omp_unset_lock(&lock);
}

void Cluster::dataIn(std::vector<float> data_in, Node* node) {
    omp_set_lock(&lock);
    sum_vec = addVector(sum_vec, node->sum_vector);
    node_list.push_back(node);
    point_number += node->point_number;
    omp_unset_lock(&lock);
}

void Cluster::dataIn(int point_num, std::vector<float> data_in) {
    omp_set_lock(&lock);
    sum_vec = addVector(sum_vec, data_in);
    point_number += point_num;
    omp_unset_lock(&lock);
}

void Cluster::dataOut(std::vector<float> data_out, int data_id) {
    omp_set_lock(&lock);
    auto it = std::find(data_id_list.begin(), data_id_list.end(), data_id);
    if (it != data_id_list.end()) {
        auto new_end = std::remove(data_id_list.begin(), data_id_list.end(), data_id);
        data_id_list.erase(new_end, data_id_list.end());

        sum_vec = subtractVector(sum_vec, data_out);
        point_number -= 1;
    }
    omp_unset_lock(&lock);
}

void Cluster::dataOut(std::vector<float> data_out, Node* node) {
    omp_set_lock(&lock);
    auto it = std::find(node_list.begin(), node_list.end(), node);
    if (it != node_list.end()) {
        auto new_end = std::remove(node_list.begin(), node_list.end(), node);
        node_list.erase(new_end, node_list.end());

        sum_vec = subtractVector(sum_vec, data_out);
        point_number -= node->point_number;
    }
    omp_unset_lock(&lock);
}

void Cluster::dataOut(int point_num, std::vector<float> data_out) {
    omp_set_lock(&lock);
    sum_vec = subtractVector(sum_vec, data_out);
    point_number -= point_num;
    omp_unset_lock(&lock);
}
