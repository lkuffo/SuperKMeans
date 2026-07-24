/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "sort_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cmath>
#include <algorithm>

void SortKmeans::free() {
    OriginalSpaceKmeans::free();
    delete [] sortedCenters;
    sortedCenters = NULL;
}


int SortKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;
        
        if (threadId == 0) {
            sort_centers();
        }
        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short initial = assignment[i];
            unsigned short closest = initial;
            
            float minDistance = pointCenterDist2(i, initial);

            for (int o = 1; o < k; ++o) {
                if (minDistance < sortedCenters[initial * k + o].first) break;

                const unsigned short j = sortedCenters[initial * k + o].second;
            
                const float distance = pointCenterDist2(i, j);
                if (distance < minDistance) {
                    minDistance = distance;
                    closest = j;
                    o = 0;
                }
                else if (j < closest) {
                    if (distance == minDistance) {
                        closest = j;
                        o = 0;
                    }
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

/* This function initializes the upper/lower bounds, assignment, centerCounts,
 * and sumNewCenters. It sets the bounds to invalid values which will force the
 * first iteration of k-means to set them correctly.
 *
 * Parameters: none
 *
 * Return value: none
 */
void SortKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    OriginalSpaceKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    sortedCenters = new std::pair<float, unsigned short>[k * k];
}

void SortKmeans::sort_centers() {
    // Compute inter-center distances
    for (int j = 0; j < k; ++j) {
        sortedCenters[j * k + j].first = 0.0;
        sortedCenters[j * k + j].second = j;
        for (int p = j + 1; p < k; ++p) {
            sortedCenters[p * k + j].first = sortedCenters[j * k + p].first = centerCenterDist2(j, p) / 4.0;
            sortedCenters[j * k + p].second = p;
            sortedCenters[p * k + j].second = j;
        }
    }

    // Sort centers by distance and record the range
    for (int j = 0; j < k; ++j) {
        std::sort(sortedCenters + j * k, sortedCenters + (j + 1) * k);
    }
}


