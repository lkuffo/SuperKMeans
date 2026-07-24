/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "heap_kmeans.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>

void HeapKmeans::free() {
    for (int t = 0; t < numThreads; ++t) {
        delete [] heaps[t];
    }
    TriangleInequalityBaseKmeans::free();
    delete [] heaps;
    delete [] heapBounds;
    heaps = NULL;
    heapBounds = NULL;
}

int HeapKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    std::greater<std::pair<float, int> > heapComp;

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        for (int h = 0; h < k; ++h) {
            Heap &heap = heaps[threadId][h];
            while (heap.size() > 0) {
                 if (heapBounds[h] <= heap[0].first)
                     break;

                int i = heap[0].second;
                
                std::pop_heap(heap.begin(), heap.end(), heapComp);
                heap.pop_back();

                unsigned short closest = assignment[i];
                unsigned short nextClosest = 0;
            
                float u2 = pointCenterDist2(i, closest);
                float l2 = std::numeric_limits<float>::max();
    
                for (unsigned short j = 0; j < k; ++j) {
                    if (j == closest) continue;

                    float dist2 = pointCenterDist2(i, j);
                    if (dist2 < u2) {
                        l2 = u2;
                        u2 = dist2;
                        nextClosest = closest;
                        closest = j;
                    } else if (dist2 < l2) {
                        l2 = dist2;
                        nextClosest = j;
                    }
                }

                const float bound = sqrt(l2) - sqrt(u2);

                // Break ties consistently with Lloyd (also prevents infinite cycle)
                if ((bound == 0.0) && (nextClosest < closest)) {
                    closest = nextClosest;
                }

                if (closest != assignment[i]) {
                    changeAssignment(i, closest, threadId);
                }

                Heap &newHeap = heaps[threadId][closest];
                newHeap.push_back(std::make_pair(heapBounds[closest] + bound, i));
                std::push_heap(newHeap.begin(), newHeap.end(), heapComp);
            }
        }

        verifyAssignment(iterations, start(threadId), end(threadId));

        synchronizeAllThreads();
        if (threadId == 0) {
            int furthestMoving = move_centers();
            converged = (0.0 == centerMovement[furthestMoving]);
            update_bounds();
        }

        synchronizeAllThreads();
    }

    return iterations;
}

void HeapKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    heaps = new Heap*[numThreads];
    heapBounds = new float[k];

    for (int t = 0; t < numThreads; ++t) {
        heaps[t] = new Heap[k];
        int startNdx = start(t);
        int endNdx = end(t);
        heaps[t][0].resize(endNdx - startNdx, std::make_pair(-1.0, 0));
        for (int i = 0; i < endNdx - startNdx; ++i) {
            heaps[t][0][i].second = i + startNdx;
        }
    }

    std::fill(heapBounds, heapBounds + k, 0.0);
}

void HeapKmeans::update_bounds() {
    int furthestMovingCenter = 0;
    float longest = centerMovement[furthestMovingCenter];
    float secondLongest = 0.0;
    for (int j = 0; j < k; ++j) {
        if (longest < centerMovement[j]) {
            secondLongest = longest;
            longest = centerMovement[j];
            furthestMovingCenter = j;
        } else if (secondLongest < centerMovement[j]) {
            secondLongest = centerMovement[j];
        }
    }

    for (int j = 0; j < k; ++j) {
        heapBounds[j] += centerMovement[j];
        if (j == furthestMovingCenter) {
            heapBounds[j] += secondLongest;
        } else {
            heapBounds[j] += longest;
        }
    }
}

