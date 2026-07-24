#ifndef KNNRES_H
#define KNNRES_H

#include <limits>

class KnnRes {
public:
    float dis;
    int id;

    KnnRes() {
        dis = std::numeric_limits<float>::max();
        id = -1;
    }
    KnnRes(float dis): dis(dis) {}
    KnnRes(float dis, int id): dis(dis), id(id) {}
    ~KnnRes() {}
};

#endif