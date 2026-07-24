/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "drake_kmeans.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>
#include <cassert>

DrakeKmeans::DrakeKmeans(int aB) : closestOtherCenters(NULL) {
    numLowerBounds = aB;
}

void DrakeKmeans::free() {
    for (int i = 0; i < n; ++i) {
        delete [] closestOtherCenters[i];
    }
    TriangleInequalityBaseKmeans::free();
    delete [] closestOtherCenters;
    closestOtherCenters = NULL;
}

int DrakeKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    int numLowerBoundsRemaining = numLowerBounds;

    std::pair<float, int> *centerOrder = new std::pair<float, int>[k];

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;
        
        // Reset the catcher statistic
        int maxCatcher = 0;

        // Find nearest center for each point
        for (int i = startNdx; i < endNdx; ++i) {
            // The index of the lower bound that caught us (hopefully)
            int catcher;
            
            // Check bounds, widening the check outward as necessary
            // to determine if we must recalculate all the bounds.
            bool mustRecalculate = true;
            for (catcher = 0; mustRecalculate && (catcher < numLowerBoundsRemaining); ++catcher) {
                // Check if the upper bound is within this lower bound              
                // Used to be <=, as that's theoretically equivalent,
                // but I'm trying to match naive EXACTLY, i.e. stable_sort, etc.
                if (upper[i] < lower[i * numLowerBounds + catcher]) {                   
                    // We've been caught by this lower bound, so we
                    // don't need to recalculate everything!
                    mustRecalculate = false;
                    
                    // If this is the first lower bound, then the assigned
                    // center cannot possibly have changed, which is great
                    if (catcher != 0) {
                        // Otherwise, we only have to reorder the (hopefully
                        // few) centers within the lower bound that caught us.
                        reorder_near_centers(i, catcher, centerOrder, threadId);
                    }
                }
            }
            
            // Keep track of the catches
            if ((maxCatcher < catcher) && (catcher < numLowerBoundsRemaining)) {
                maxCatcher = catcher;
            }

            // If none of the bounds held, then recalculate everything.
            if (mustRecalculate) {
                // Sort the centers by increasing distance
                find_near_centers(i, numLowerBoundsRemaining, centerOrder, threadId);
                //find_near_centers_general(i, k);
            }       
        }
        
        verifyAssignment(iterations, startNdx, endNdx);
        
        synchronizeAllThreads();
        // Adjust the centers based on their new point memberships
        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            
            // If nothing happened when we tried to move centers, we've converged!
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        // Otherwise, release tension in the bounds caused by centers' movement
        synchronizeAllThreads();
        update_bounds(startNdx, endNdx, numLowerBoundsRemaining);
        
        // Adjust the number of lower bounds being used
        if ((10 < iterations) && ((k >> 3) <= maxCatcher)) {
            numLowerBoundsRemaining = std::max(maxCatcher, 1);
        }
    }

    delete [] centerOrder;

    return iterations;
}

void DrakeKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    assert(0 < numLowerBounds);
    assert(numLowerBounds < k);
    closestOtherCenters = new unsigned short*[n];

    for (int i = 0; i < n; ++i) {
        closestOtherCenters[i] = new unsigned short[numLowerBounds];
        for (int j = 0; j < numLowerBounds; ++j) {
            closestOtherCenters[i][j] = j + 1;
        }
    }
}

void DrakeKmeans::update_bounds(int startNdx, int endNdx, int numLowerBoundsRemaining) {
    int furthestMovingCenter = (int)(std::max_element(centerMovement, centerMovement + k) - centerMovement);

    // Update the upper and lower bounds for each point
    for (int i = startNdx; i < endNdx; ++i) {
        // Widen the upper bound based on the closest center's movement
        upper[i] += centerMovement[assignment[i]];
        
        // Update all but the outermost lower bound
        for (int j = 0; j < numLowerBoundsRemaining - 1; ++j) {
            // Shrink the lower bound by the distance its center has moved
            lower[i * numLowerBounds + j] -= centerMovement[closestOtherCenters[i][j]];
        }
        
        // Shrink the outermost lower bound by maximum distance moved by any
        // center; of course this is not as tight as possible, but it's cheap!
        lower[i * numLowerBounds + numLowerBoundsRemaining - 1] -= centerMovement[furthestMovingCenter];
        // TODO try the tighter version again. no idea why it didn't work.
    
        // Force lower bounds to stay in order by collapsing
        // the circles from the outside inward
        for (int j = numLowerBoundsRemaining - 2; j >= 0; --j) {
            if (lower[i * numLowerBounds + j + 1] < lower[i * numLowerBounds + j]) {
                lower[i * numLowerBounds + j] = lower[i * numLowerBounds + j + 1];
            }
        }
    }
}


void DrakeKmeans::find_near_centers(int i, int numLowerBoundsRemaining, std::pair<float, int> *order, int threadId) {
    // Sort all centers by increasing distance from this point
    for (int j = 0; j < k; ++j) {
        // Record the squared distances
        order[j].first = pointCenterDist2(i, j);
        order[j].second = j;
    }
    std::partial_sort(order, order + numLowerBoundsRemaining + 1, order + k);

    // Reassign the center (incremental)
    if (assignment[i] != order[0].second) {
        changeAssignment(i, order[0].second, threadId);
    }

    // Record the indices of the near centers,
    // and update their lower bounds
    upper[i] = sqrt(order[0].first);
    for (int j = 0; j < numLowerBoundsRemaining; ++j) {
        closestOtherCenters[i][j] = order[j + 1].second;
        lower[i * numLowerBounds + j] = sqrt(order[j + 1].first);
    }
}

void DrakeKmeans::reorder_near_centers(int i, int catcher, std::pair<float, int> *order, int threadId) {
    // Sort the centers we caught, by increasing distance from this point
    order[0].first = pointCenterDist2(i, assignment[i]);
    order[0].second = assignment[i];
    for (int j = 0; j < catcher; ++j) {
        // Record the squared distances
        order[j + 1].second = closestOtherCenters[i][j];
        order[j + 1].first = pointCenterDist2(i, order[j + 1].second);
    }
    std::partial_sort(order, order + catcher + 1, order + catcher + 1);
    
    // Reassign the center (incremental)
    if (assignment[i] != order[0].second) {
        changeAssignment(i, order[0].second, threadId);
    }

    // Record the indices of the caught centers,
    // and update their lower bounds
    upper[i] = sqrt(order[0].first);
    for (int j = 0; j < catcher; ++j) {
        closestOtherCenters[i][j] = order[j + 1].second;
        lower[i * numLowerBounds + j] = sqrt(order[j + 1].first);
    }
}

