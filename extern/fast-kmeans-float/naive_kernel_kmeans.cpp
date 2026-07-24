/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "naive_kernel_kmeans.h"
#include <algorithm>
#include <iterator>

int NaiveKernelKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        bool membershipChanged = false;

        // precompute the (kernelized) inner product of each center with itself
        computeMemberships(threadId, &memberships, &cc);

        // we have converged... until we find out we haven't
        if (threadId == 0) {
            converged = true;
        }
        synchronizeAllThreads();

        // loop over all records
        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            float currentDist2 = pointCenterDist2(i, closest);

            // now update the lower bound by looking at all other centers
            for (int j = 0; j < k; ++j) {
                if (j == closest) {
                    continue;
                }

                float dist2 = pointCenterDist2(i, j);

                if (dist2 < currentDist2) {
                    closest = j;
                    currentDist2 = dist2;
                }
            }

            // if the assignment for i has changed, then adjust the counts and
            // locations of each center's accumulated mass
            if (assignment[i] != closest) {
                assignment[i] = closest;
                membershipChanged = true;
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        if (membershipChanged) {
            setConverged(false);
        }

        synchronizeAllThreads();
    }

    return iterations;
}

