#include "Centroid.h"
#include "../utils/Utils.h"


Centroid::Centroid(int point_id, std::vector<float> coordinate, int centroid_id)
    : point_id(point_id), coordinate(coordinate), centroid_id(centroid_id) {
        cluster = new Cluster(centroid_id, coordinate.size());
        max_drift = 0;
        drift = 0;
    }

Centroid::~Centroid() { delete cluster; }

std::vector<float> Centroid::getCoordinate() { return coordinate; }

std::vector<float> Centroid::getOldCoordinate() { return old_coordinate; }

Cluster* Centroid::getCluster() { return cluster; }

void Centroid::updateCoordinate(std::vector<float> new_coordinate) {
    this->old_coordinate = this->coordinate;
    this->coordinate = new_coordinate;
    this->drift = Utils::distance1(this->coordinate, this->old_coordinate);
    this->max_drift = std::max(drift, max_drift);
}