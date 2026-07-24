/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "elkan_kernel_kmeans.h"
#include <algorithm>
#include <iterator>

void ElkanKernelKmeans::update_center_dists(int threadId) {
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        #ifdef USE_THREADS
        if (c1 % numThreads != threadId) {
            continue;
        }
        #endif

        centerCenterDistDiv2[c1 * k + c1] = s[c1] = std::numeric_limits<float>::max();

        for (int c2 = 0; c2 < k; ++c2) {
            if (c2 > c1) {
                // divide by 2 here since we always use the inter-center
                // distances divided by 2
                centerCenterDistDiv2[c1 * k + c2] = centerCenterDistDiv2[c2 * k + c1] = sqrt(centerCenterDist2(c1, c2)) / 2.0;
            }

            if (centerCenterDistDiv2[c1 * k + c2] < s[c1]) {
                s[c1] = centerCenterDistDiv2[c1 * k + c2];
            }
        }
    }
}

int ElkanKernelKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // precompute the (kernelized) inner product of each center with itself
    computeMemberships(threadId, &memberships, &cc);
    synchronizeAllThreads();

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        bool membershipChanged = false;

        // we have converged... until we find out we haven't
        synchronizeAllThreads();
        if (threadId == 0) {
            setConverged(true);
        }
        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;

            if (upper[i] <= s[closest]) {
                continue;
            }

            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                if (upper[i] <= lower[i * k + j]) { continue; }
                if (upper[i] <= centerCenterDistDiv2[closest * k + j]) { continue; }

                // ELKAN 3(a)
                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));
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
                }
            }
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

        if (converged) {
            break;
        }

        // compute center movements and update upper and lower bounds
        computeMemberships(threadId, &newMemberships, &newCc);
        synchronizeAllThreads();
        computeCenterMovement(threadId);
        synchronizeAllThreads();

        if (threadId == 0) {
            memberships.swap(newMemberships);
            cc.swap(newCc);
        }
        synchronizeAllThreads();
        update_center_dists(threadId);
        synchronizeAllThreads();

        update_bounds(startNdx, endNdx);
        synchronizeAllThreads();
    }

    return iterations;
}

void ElkanKernelKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    KernelKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    centerCenterDistDiv2 = new float[k * k];
    s = new float[k];
    upper = new float[n];
    lower = new float[n * k];

    // start with invalid bounds and assignments which will force the first
    // iteration of k-means to do all its standard work 
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
    std::fill(s, s + k, 0.0);
    std::fill(upper, upper + n, std::numeric_limits<float>::max());
    std::fill(lower, lower + n * k, 0.0);

    newMemberships.clear();
    newMemberships.resize(k);
    newCc.resize(k);
    std::fill(newCc.begin(), newCc.end(), 0.0);
}


void ElkanKernelKmeans::free() {
    KernelKmeans::free();
    delete [] centerCenterDistDiv2;
    delete [] s;
    delete [] upper;
    delete [] lower;
    centerCenterDistDiv2 = NULL;
    s = NULL;
    upper = NULL;
    lower = NULL;

    newMemberships.clear();
    newCc.clear();
}

void ElkanKernelKmeans::update_bounds(int startNdx, int endNdx) {
    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
        for (int j = 0; j < k; ++j) {
            lower[i * k + j] -= centerMovement[j];
        }
    }
}


void ElkanKernelKmeans::computeCenterMovement(int threadId) {
    for (int j = 0; j < k; ++j) {
        #ifdef USE_THREADS
        if (j % numThreads != threadId) {
            continue;
        }
        #endif
        centerMovement[j] = sqrt(cc[j] - 2.0 * centerCenterInnerProductGeneral(memberships[j], newMemberships[j]) + newCc[j]);
    }
}
