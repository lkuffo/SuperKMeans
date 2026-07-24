#include "Node.h"
#include "../utils/Utils.h"

Node::Node() {
    leftChild = nullptr;
    rightChild = nullptr;
}

Node::Node(std::vector<float> pivot, float radius)
    : pivot(pivot), radius(radius) {
    leftChild = nullptr;
    rightChild = nullptr;
}

Node::Node(std::vector<float> pivot, float radius, bool isLeaf)
    : pivot(pivot), radius(radius), leaf(leaf) {
    leftChild = nullptr;
    rightChild = nullptr;
}

Node::~Node() {
    if (!isLeaf()) {
        delete leftChild;
        delete rightChild;
        leftChild = nullptr;
        rightChild = nullptr;
    }
}

// only called by leaf node
void Node::initLeafNode(std::vector<int> data_id_list, int size) {
    this->leaf = true;
    this->data_id_list = data_id_list;
    this->point_number = size;
    this->sum_vector = Utils::multiplyVector(pivot, point_number);
    centroid_id_for_data = std::vector(size, -1);
}

// set the sum_vector and point_number after building the ball-tree
void Node::setSum() {    
    if (this->isLeaf())
        return;

    int size = pivot.size();
    sum_vector = std::vector<float>(size, 0.0f);

    if (leftChild != nullptr) {
        leftChild->setSum();
        sum_vector = Utils::addVector(sum_vector, leftChild->sum_vector);
        point_number += leftChild->point_number;
    }
    if (rightChild != nullptr) {
        rightChild->setSum();
        sum_vector = Utils::addVector(leftChild->sum_vector, rightChild->sum_vector);
        point_number += rightChild->point_number;
    }
}

std::vector<int> Node::getAllDataId() {
    if (isLeaf()) {
        return data_id_list;
    }

    std::vector<int> data_in_left_child = leftChild->getAllDataId();
    std::vector<int> data_in_right_child = rightChild->getAllDataId();
    data_in_left_child.insert(data_in_left_child.end(), 
        data_in_right_child.begin(), data_in_right_child.end());
    return data_in_left_child;
}

bool Node::operator==(const Node& other) const {
    return (this->pivot == other.pivot) && (this->radius == other.radius);
}