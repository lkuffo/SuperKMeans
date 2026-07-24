#ifndef HEAP_KMEANS_H
#define HEAP_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * HeapKmeans keeps k heaps, one for each cluster. The points for each cluster
 * are kept in the respective heaps, ordered by the bound difference
 * (lower-upper), where lower is the same as Hamerly's lower bound. For details
 * on this algorithm, please see the forthcoming book chapter (Partitional
 * Clustering Algorithms, Springer 2014).
 */

#include "triangle_inequality_base_kmeans.h"
#include <vector>

typedef std::vector<std::pair<float, int> > Heap;

class HeapKmeans : public TriangleInequalityBaseKmeans {
    public:
        HeapKmeans() : heaps(NULL), heapBounds(NULL) {}
        virtual ~HeapKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "heap"; }

    protected:
        virtual int runThread(int threadId, int maxIterations);
        void update_bounds();

        // Each thread has k heaps, so we have numThreads * k heaps.
        Heap **heaps;

        // The heapBounds essentially accumulate the total distance traveled by
        // each center over the iterations of k-means. This value is used to
        // compare with the heap priority to determine if a point's bounds
        // (lower-upper) are violated (i.e. < 0).
        float *heapBounds;
};

#endif

