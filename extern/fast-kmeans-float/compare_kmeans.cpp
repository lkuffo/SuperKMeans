/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "compare_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cmath>
#include <algorithm>

void CompareKmeans::free() {
    OriginalSpaceKmeans::free();
    delete [] centersDist2div4;
    centersDist2div4 = NULL;
}

void CompareKmeans::update_center_dists(int threadId) {
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        #ifdef USE_THREADS
        if (c1 % numThreads != threadId) {
            continue;
        }
        #endif

        centersDist2div4[c1 * k + c1] = std::numeric_limits<float>::max();

        for (int c2 = c1 + 1; c2 < k; ++c2) {
            centersDist2div4[c1 * k + c2] = centersDist2div4[c2 * k + c1] = centerCenterDist2(c1, c2) / 4.0;
        }
    }
}

void CompareKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    OriginalSpaceKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    centersDist2div4 = new float[k * k];
    std::fill(centersDist2div4, centersDist2div4 + k * k, 0.0);
}

int CompareKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        update_center_dists(threadId);
        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            int minClass = assignment[i];
            float minDist2 = pointCenterDist2(i, minClass);

            for (int j = 0; j < k; ++j) {
                // center-center squared distances are already divided by 4.0
                if (centersDist2div4[j * k + minClass] > minDist2) continue;

                if (j == minClass) continue;

                const float dist2 = pointCenterDist2(i, j);
        
                if (dist2 < minDist2) {
                    minDist2 = dist2;
                    minClass = j;
                } else if (dist2 == minDist2) {
                    if (j < minClass) {
                        minClass = j;
                    }
                }
            }
            
            if (assignment[i] != minClass) {
                changeAssignment(i, minClass, threadId);
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


