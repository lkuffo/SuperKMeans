#ifndef SORT_KMEANS_H
#define SORT_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * The SortKmeans algorithm is an implementation of Phillip's Sort-Means
 * algorithm.
 */

#include "original_space_kmeans.h"

class SortKmeans : public OriginalSpaceKmeans {
    public:
        SortKmeans() : sortedCenters(NULL) {}
        virtual ~SortKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "sort"; }

    private:
        virtual int runThread(int threadId, int maxIterations);
        void sort_centers();

        // float is center-center distance squared, divided by 4;
        // short is center index
        std::pair<float, unsigned short> *sortedCenters;
};

#endif

