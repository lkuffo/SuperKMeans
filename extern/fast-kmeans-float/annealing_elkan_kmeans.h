#ifndef ANNEALING_ELKAN_KMEANS_H
#define ANNEALING_ELKAN_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations.
 */


#include "triangle_inequality_base_kmeans.h"

class AnnealingElkanKmeans : public TriangleInequalityBaseKmeans {
public:
    AnnealingElkanKmeans() : centerCenterDistDiv2(NULL) {}
    virtual ~AnnealingElkanKmeans() { free(); }
    virtual void free();
    virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
    virtual std::string getName() const { return "an_elkan"; }

protected:
    virtual int runThread(int threadId, int maxIterations);

    // Update the distances between each pair of centers.
    void update_center_dists(int threadId);

    // Update the upper and lower bounds for the range of points given.
    void update_bounds(int startNdx, int endNdx);

    // Keep track of the distance (divided by 2) between each pair of
    // points.
    float *centerCenterDistDiv2;

    int *lastChanged;
};

#endif

