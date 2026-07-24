#ifndef BETA_HAMERLY_KMEANS_H
#define BETA_HAMERLY_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * HamerlyKmeans implements Hamerly's k-means algorithm that uses one lower
 * bound per point.
 */

#include "triangle_inequality_base_kmeans.h"

class BetaHamerlyKmeans : public TriangleInequalityBaseKmeans {
public:
    BetaHamerlyKmeans() { numLowerBounds = 1; }
    virtual ~BetaHamerlyKmeans() { free(); }
    virtual std::string getName() const { return "hamerly"; }

protected:
    // Update the upper and lower bounds for the given range of points.
    float *pointNorms;
    int *bins;
    int *beta;
    int *alpha;
    float *probabilities;

    void update_bounds(int startNdx, int endNdx);

    virtual int runThread(int threadId, int maxIterations);
};

#endif