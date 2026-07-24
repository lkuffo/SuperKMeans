/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "sample_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cstring>
#include <random>
#include <algorithm>
/* The classic algorithm of assign, move, repeat. No optimizations that prune
 * the search.
 *
 * Return value: the number of iterations performed (always at least 1)
 */

int SampleKmeans::runThread(int threadId, int maxIterations) {
    // track the number of iterations the algorithm performs
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    float samplePercentage;
    std::cin >> samplePercentage;
    int sampleSize;
    sampleSize = samplePercentage * x->n;
    std::random_device rd;
    std::mt19937 gen(rd());
    // Create a dynamic array for indices
    int* indices = new int[x->n];
    for (int i = 0; i < x->n; ++i) {
        indices[i] = i;
    }
    // Shuffle the indices randomly using std::shuffle
    std::shuffle(indices, indices + x->n, gen);
    int i;
    while ((iterations < maxIterations) && (! converged)) {
        ++iterations;
        // loop over all examples
        for (int t = startNdx; t < startNdx + sampleSize; ++t) {
            i = indices[t];
            // look for the closest center to this example
            int closest = 0;
            float closestDist2 = std::numeric_limits<float>::max();
            for (int j = 0; j < k; ++j) {
                float d2 = pointCenterDist2(i, j);
                if (d2 < closestDist2) {
                    closest = j;
                    closestDist2 = d2;
                }
            }
            if (assignment[i] != closest) {
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        synchronizeAllThreads();

        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();
    }

    return iterations;
}

