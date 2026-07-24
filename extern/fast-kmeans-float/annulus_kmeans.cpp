/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "annulus_kmeans.h"
#include "general_functions.h"
#include <cmath>
#include <numeric>
#include <algorithm>

void AnnulusKmeans::free() {
    HamerlyKmeans::free();

    delete [] guard;
    delete [] xNorm;
    delete [] cOrder;
    guard = NULL;
    xNorm = NULL;
    cOrder = NULL;
}

int AnnulusKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        update_s(threadId);
        synchronizeAllThreads();

        if (threadId == 0) {
            // compute the inter-center distances, keeping only the closest distances
            sort_means_by_norm();
        }
        synchronizeAllThreads();

        // loop over all records
        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];

            // if upper[i] is less than the greater of these two, then we can
            // ignore record i
            float upper_comparison_bound = std::max(s[closest], lower[i]);

            // first check: if u(x) <= s(c(x)) or u(x) <= lower(x), then ignore
            // x, because its closest center must still be closest
            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            // otherwise, compute the real distance between this record and its
            // closest center, and update upper
            float u2 = pointCenterDist2(i, closest);
            upper[i] = sqrt(u2);

            // if (u(x) <= s(c(x))) or (u(x) <= lower(x)), then ignore x
            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            float l2 = pointCenterDist2(i, guard[i]);
            lower[i] = sqrt(l2);

            float beta = std::max(lower[i], upper[i]);

            std::pair<float, int>* begin = std::lower_bound(cOrder, cOrder + k, std::make_pair(xNorm[i] - beta, k));
            std::pair<float, int>* end = std::lower_bound(begin, cOrder + k, std::make_pair(xNorm[i] + beta, k));

            for (std::pair<float, int>* jp = begin; jp != end; ++jp) {
                if (jp->second == closest) continue;

                float dist2 = pointCenterDist2(i, jp->second);
                if (dist2 <= u2) {
                    if (dist2 == u2) {
                        if (jp->second < closest) closest = jp->second;
                    } else {
                        l2 = u2;
                        u2 = dist2;
                        guard[i] = closest;
                        closest = jp->second;
                    }
                } else if (dist2 < l2) {
                    // we must reduce the lower bound on the distance to the
                    // *second* closest center to x[i]
                    l2 = dist2;
                    guard[i] = jp->second;
                }
            }

            // we have been dealing in squared distances; need to convert
            lower[i] = sqrt(l2);

            // if the assignment for i has changed, then adjust the counts and
            // locations of each center's accumulated mass
            if (assignment[i] != closest) {
                upper[i] = sqrt(u2);
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        // calculate the new center locations
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

    return iterations;
}

void AnnulusKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    HamerlyKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    guard = new unsigned short[n]; 
    xNorm = new float[n];
    cOrder = new std::pair<float, int>[k];

    for (int i = 0; i < k; ++i) {
        cOrder[i].first = 0.0;
        cOrder[i].second = i;
    }

    std::fill(guard, guard + n, 1);
    for (int i = 0; i < n; ++i) {
        xNorm[i] = sqrt(pointPointInnerProduct(i, i));
    }
}

void AnnulusKmeans::sort_means_by_norm() {
    // sort the centers by their norms
    for (int c1 = 0; c1 < k; ++c1) {
        cOrder[c1].first = sqrt(centerCenterInnerProduct(c1, c1));
        cOrder[c1].second = c1;
    }
    std::sort(cOrder, cOrder + k);
}


