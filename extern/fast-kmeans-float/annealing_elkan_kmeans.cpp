/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "annealing_elkan_kmeans.h"
#include "general_functions.h"
#include <cmath>

void AnnealingElkanKmeans::update_center_dists(int threadId) {
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<float>::max();

            for (int c2 = 0; c2 < k; ++c2) {
                // we do not need to consider the case when c1 == c2 as centerCenterDistDiv2[c1*k+c1]
                // is equal to zero from initialization, also this distance should not be used for s[c1]
                if (c1 != c2) {
                    // divide by 2 here since we always use the inter-center
                    // distances divided by 2
                    centerCenterDistDiv2[c1 * k + c2] = sqrt(centerCenterDist2(c1, c2)) / 2.0;

                    if (centerCenterDistDiv2[c1 * k + c2] < s[c1]) {
                        s[c1] = centerCenterDistDiv2[c1 * k + c2];
                    }
                }
            }
        }
    }
}

int AnnealingElkanKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);
    float changedLastIter = 0;
    float modified_upper;
    float atr = 0;
    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        update_center_dists(threadId);
        synchronizeAllThreads();
        if(iterations <= 2){
            atr = 1 + changedLastIter/x->n;
        }else{
            atr = 0.5 * atr + 0.5 * (1 + changedLastIter/x->n);
        }
        //std::cout << atr << std::endl;
        changedLastIter = 0;

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;
            if (upper[i] <= s[closest]) {
                continue;
            }
            modified_upper = upper[i] / atr;
            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                if (modified_upper <= lower[i * k + j]) {
                    continue;
                }
                if (modified_upper <= centerCenterDistDiv2[closest * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));
                    modified_upper = upper[i];
                    lower[i * k + closest] = upper[i];
                    r = false;
                    if ((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lower[i * k + j] = sqrt(pointCenterDist2(i, j));
                if (lower[i * k + j] < upper[i]) {
                    closest = j;
                    upper[i] = lower[i * k + j];
                    modified_upper = upper[i];
                }
            }

            if (assignment[i] != closest) {
                changeAssignment(i, closest, threadId);
                changedLastIter = changedLastIter + 1;
            }
        }
        //std::cout << fraction << std::endl;
        verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        synchronizeAllThreads();
        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();
        if (! converged) {
            update_bounds(startNdx, endNdx);
        }
        synchronizeAllThreads();
    }
    return iterations;}

void AnnealingElkanKmeans::update_bounds(int startNdx, int endNdx) {
    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * numLowerBounds + j] -= centerMovement[j];
        }
    }
}

void AnnealingElkanKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    numLowerBounds = aK;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    centerCenterDistDiv2 = new float[k * k];
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
}

void AnnealingElkanKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    delete [] centerCenterDistDiv2;
    centerCenterDistDiv2 = NULL;
}