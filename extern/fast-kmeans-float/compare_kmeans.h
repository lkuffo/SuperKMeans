#ifndef COMPARE_KMEANS_H
#define COMPARE_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * The CompareKmeans algorithm is an implementation of Phillip's Compare-Means
 * algorithm.
 */

#include "original_space_kmeans.h"

class CompareKmeans : public OriginalSpaceKmeans {
    public:
        CompareKmeans() : centersDist2div4(NULL) {}
        virtual ~CompareKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "compare"; }
    
    private:
        virtual int runThread(int threadId, int maxIterations);
        void update_center_dists(int threadId);

        // A matrix (implemented in a single array) of center-center squared
        // distances, divided by 4. By dividing by 4 early, we save divisions
        // later during the comparison of this value.
        float *centersDist2div4;
};

#endif

