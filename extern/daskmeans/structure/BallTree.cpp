#include <BallTree.h>
#include <../utils/Utils.h>

using namespace Utils;


BallTree::BallTree() { root = new Node(); }

BallTree::BallTree(int capacity): capacity(capacity) { root = new Node(); }

BallTree::BallTree(int capacity, int data_scale): capacity(capacity) { 
    root = new Node();
    height = static_cast<int>(std::round(std::log2(data_scale / capacity))) + 1;
}

BallTree::~BallTree() {
    if (this != nullptr && root != nullptr) {
        delete root;
        root = nullptr;
    }
}

void BallTree::buildBallTree(std::vector<std::vector<float>>& dataset, int data_scale) {
    createNode(dataset, data_scale, *root);
    std::vector<int> point_id_list(data_scale);
    for (int i = 0; i < data_scale; i++) {
        point_id_list[i] = i;
    }
    // buildBallTree1(dataset, *root, point_id_list, 4);
    buildBallTree1(dataset, *root, point_id_list);
    initBallTree();
}

void BallTree::buildBallTree(std::vector<Centroid*>& centroid_list, int k) {
    createNode(centroid_list, k, *root);
    std::vector<int> centroid_id_list(k);
    for (int i = 0; i < k; i++) {
        centroid_id_list[i] = i;
    }
    buildBallTree1(centroid_list, *root, centroid_id_list);
}


/**
 * @brief build a ball-tree recursively
 * @param dataset the full dataset loaded by the algorithm
 * @param node current ball-tree node that will be constructed
 * @param point_id_list the list of points' ids covered by current node
 */
void BallTree::buildBallTree1(std::vector<std::vector<float>>& dataset, 
    Node& node, std::vector<int>& point_id_list, int h) {

    int size = point_id_list.size();
    if (size <= this->capacity || h >= height) {
        node.initLeafNode(point_id_list, size);
        return;
    }
    // is not leaf node
    // 1. split the point_id_list into two list
    std::vector<float> vec1 = dataset[node.farthest_point_id[0]];
    std::vector<float> vec2 = dataset[node.farthest_point_id[1]];
    std::vector<int> point_id_list1;
    std::vector<int> point_id_list2;

    for (int point_id : point_id_list) {
        std::vector<float> point = dataset[point_id];
        if (distance2(vec1, point) <= distance2(vec2, point)) {
            point_id_list1.push_back(point_id);
        } else {
            point_id_list2.push_back(point_id);
        }
    }

    // 2. create two child nodes
    int size1 = point_id_list1.size();
    if (size1 != 0) {
        node.leftChild = new Node();
        createNode(dataset, point_id_list1, size1, *(node.leftChild));
        buildBallTree1(dataset, *(node.leftChild), point_id_list1, h + 1);
    }

    int size2 = point_id_list2.size();
    if (size2 != 0) {
        node.rightChild = new Node();
        createNode(dataset, point_id_list2, size2, *(node.rightChild));
        buildBallTree1(dataset, *(node.rightChild), point_id_list2, h + 1);
    }
}

void BallTree::buildBallTree1(std::vector<std::vector<float>>& dataset, 
    Node& node, std::vector<int>& point_id_list) {

    int size = point_id_list.size();
    if (size <= this->capacity) {
        node.initLeafNode(point_id_list, size);
        return;
    }
    // is not leaf node
    // 1. split the point_id_list into two list
    std::vector<float> vec1 = dataset[node.farthest_point_id[0]];
    std::vector<float> vec2 = dataset[node.farthest_point_id[1]];
    std::vector<int> point_id_list1;
    std::vector<int> point_id_list2;
    int cnt = 0;

    for (int point_id : point_id_list) {
        std::vector<float> point = dataset[point_id];
        if (distance2(vec1, point) <= distance2(vec2, point) && cnt <= size / 2) {
        // if (distance2(vec1, point) <= distance2(vec2, point)) {
            point_id_list1.push_back(point_id);
            cnt += 1;
        } else {
            point_id_list2.push_back(point_id);
        }
    }

    // 2. create two child nodes
    int size1 = point_id_list1.size();
    if (size1 != 0) {
        node.leftChild = new Node();
        createNode(dataset, point_id_list1, size1, *(node.leftChild));
        buildBallTree1(dataset, *(node.leftChild), point_id_list1);
    }
    
    int size2 = point_id_list2.size();
        if (size2 != 0) {
        node.rightChild = new Node();
        createNode(dataset, point_id_list2, size2, *(node.rightChild));
        buildBallTree1(dataset, *(node.rightChild), point_id_list2);
    }
}

void BallTree::buildBallTree1(std::vector<Centroid*>& centroid_list, 
    Node& node, std::vector<int>& centroid_id_list) {

    int size = centroid_id_list.size();
    if (size <= this->capacity) {
        node.initLeafNode(centroid_id_list, size);
        return;
    }
    // is not leaf node
    // 1. split the point_id_list into two list
    std::vector<float> vec1 = centroid_list[node.farthest_point_id[0]]->coordinate;
    std::vector<float> vec2 = centroid_list[node.farthest_point_id[1]]->coordinate;
    std::vector<int> centroid_id_list1;
    std::vector<int> centroid_id_list2;
    int cnt = 0;

    for (int centroid_id : centroid_id_list) {
        std::vector<float> coordinate = centroid_list[centroid_id]->coordinate;
        // if (distance2(vec1, coordinate) <= distance2(vec2, coordinate)) {
        if (distance2(vec1, coordinate) <= distance2(vec2, coordinate) && cnt <= size / 2) {
            centroid_id_list1.push_back(centroid_id);
            cnt += 1;
        } else {
            centroid_id_list2.push_back(centroid_id);
        }
    }

    // 2. create two child nodes
    int size1 = centroid_id_list1.size();
    int size2 = centroid_id_list2.size();
    if (size1 != 0) {
        node.leftChild = new Node();
        createNode(centroid_list, centroid_id_list1, size1, *(node.leftChild));
        buildBallTree1(centroid_list, *(node.leftChild), centroid_id_list1);
    }
    if (size != 0) {
        node.rightChild = new Node();
        createNode(centroid_list, centroid_id_list2, size2, *(node.rightChild));
        buildBallTree1(centroid_list, *(node.rightChild), centroid_id_list2);
    }
}

void BallTree::buildBalancedBallTree(std::vector<std::vector<float>>& dataset, 
    int data_scale, int data_dimension) {
    
}

void BallTree::initBallTree() {
    root->setSum();
}

/*function*/

// giving a dataset and point_id_list, return a ball-tree node with pivot and radius
void BallTree::createNode(std::vector<std::vector<float>>& dataset, int data_scale, Node& node) {
    std::vector<float> pivot = sumVectorsInDataset(dataset);
    pivot = divideVector(pivot, data_scale);
    std::vector<int> farthest_point_id = getTwoFarthestPoints(pivot, dataset, data_scale);
    float radius = distance1(pivot, dataset[farthest_point_id[0]]);

    node.pivot = pivot;
    node.radius = radius;
    node.farthest_point_id = farthest_point_id;
}

void BallTree::createNode(std::vector<std::vector<float>>& dataset, 
    std::vector<int>& point_id_list, int size, Node& node) {

    std::vector<float> pivot = sumVectorsInDataset(dataset, point_id_list);
    pivot = divideVector(pivot, size);
    std::vector<int> farthest_point_id = getTwoFarthestPoints(pivot, dataset, point_id_list);
    float radius = distance1(pivot, dataset[farthest_point_id[0]]);

    node.pivot = pivot;
    node.radius = radius;
    node.farthest_point_id = farthest_point_id;
}

void BallTree::createNode(std::vector<Centroid*>& centroid_list, int data_scale, Node& node) {
    std::vector<float> pivot = sumVectorsInDataset(centroid_list);
    pivot = divideVector(pivot, data_scale);
    std::vector<int> farthest_point_id = getTwoFarthestPoints(pivot, centroid_list, data_scale);
    float radius = distance1(pivot, centroid_list[farthest_point_id[0]]->getCoordinate());

    node.pivot = pivot;
    node.radius = radius;
    node.farthest_point_id = farthest_point_id;
}

void BallTree::createNode(std::vector<Centroid*>& centroid_list, 
    std::vector<int>& centroid_id_list, int size, Node& node) {

    std::vector<float> pivot = sumVectorsInDataset(centroid_list, centroid_id_list);
    pivot = divideVector(pivot, size);
    std::vector<int> farthest_point_id = getTwoFarthestPoints(pivot, centroid_list, centroid_id_list);
    float radius = distance1(pivot, centroid_list[farthest_point_id[0]]->getCoordinate());

    node.pivot = pivot;
    node.radius = radius;
    node.farthest_point_id = farthest_point_id;
}
