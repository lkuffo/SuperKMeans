#ifndef ANNULUS_KMEANS_H
#define ANNULUS_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * The Annulus K-means algorithm is based on Hamerly's algorithm, but also sorts
 * the centers by their norms (distances from the origin). Doing this allows
 * searching the centers using the norm of the point to exclude centers that
 * cannot be close.
 */

#include "hamerly_kmeans.h"
#include <utility>

class AnnulusKmeans;

class AnnulusKmeans : public HamerlyKmeans {
    public:
        AnnulusKmeans() : xNorm(NULL), cOrder(NULL), guard(NULL) {}
        virtual ~AnnulusKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "annulus"; }

    protected:
        virtual int runThread(int threadId, int maxIterations);
        void sort_means_by_norm();

        // The norm of each point.
        float *xNorm;

        // The order of the centers (first is the norm, second is the center
        // index).
        std::pair<float, int> *cOrder;

        // Guard is an index, for each point, of what is (or might be) the
        // second-closest center. When we tighten the bound, it is the
        // second-closest; however, this is not guaranteed to hold as the bounds
        // change. It is still useful as an index of a close center that is not
        // the closest.
        unsigned short *guard;
};

#endif

