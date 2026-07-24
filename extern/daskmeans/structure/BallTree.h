// BallTree.h
#ifndef BALLTREE_H
#define BALLTREE_H

#include <vector>
#include "Centroid.h"
#include "Node.h"

class BallTree {
public:
    Node* root;
    int capacity = 1;   // leaf node capatity
    int height;
    
public:
    BallTree();

    BallTree(int capacity);

    BallTree(int capacity, int data_scale);

    ~BallTree();

    void buildBallTree(std::vector<std::vector<float>>& dataset, int data_scale);

    void buildBallTree(std::vector<Centroid*>& centroid_list, int k);

    void buildBallTree1(std::vector<std::vector<float>>& dataset, 
        Node& node, std::vector<int>& point_id_list, int height);
    
    void buildBallTree1(std::vector<std::vector<float>>& dataset, 
        Node& node, std::vector<int>& point_id_list);

    void buildBallTree1(std::vector<Centroid*>& centroid_list, 
        Node& node, std::vector<int>& centroid_id_list);

    void buildBalancedBallTree(std::vector<std::vector<float>>& dataset,
        int data_scale, int data_dimension);

    void initBallTree();

    /*function*/
    void createNode(std::vector<std::vector<float>>& dataset,
        int data_scale, Node& node);

    void createNode(std::vector<std::vector<float>>& dataset, 
        std::vector<int>& point_id_list, int size, Node& node);

    void createNode(std::vector<Centroid*>& centroid_list, 
        int data_scale, Node& node);

    void createNode(std::vector<Centroid*>& centroid_list, 
        std::vector<int>& centroid_id_list, int size, Node& node);

};

#endif