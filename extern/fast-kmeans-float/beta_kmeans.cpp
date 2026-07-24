/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "beta_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>

/* The classic algorithm of assign, move, repeat. No optimizations that prune
 * the search.
 *
 * Return value: the number of iterations performed (always at least 1)
 */

float distanceCalculation(float* p1, float* p2, float* p3){
    float temp = 0;
    while(p1 != p2){
        float t = *p1++ - *p3++;
        temp = temp + t * t;
    }
    return temp;

}


int BetaKmeans::runThread(int threadId, int maxIterations) {
    // track the number of iterations the algorithm performs
    //long calculated = 0;
    //long skipped = 0;
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);
    int* listOfEligibleCenters = new int[k];

    pointNorms = new float[n];
    for (int i = startNdx; i < endNdx; ++i) {
        float sumOfSquares = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            float coord = x->data[i*d + dim];
            sumOfSquares += coord * coord;
        }
        pointNorms[i] = sqrt(sumOfSquares);
    }

    std::vector<std::pair<float, int>> normIndexPairs;
    for (int i = 0; i < n; ++i) {
        normIndexPairs.emplace_back(pointNorms[i], i);
    }

    // Sort the vector by the norm
    std::sort(normIndexPairs.begin(), normIndexPairs.end());

    int numberOfBins = 10; // Define the number of bins
    //bins = new int[n];
    //std::fill(bins, bins + n, 0);

    // Distribute points into bins
    int pointsPerBin = n / numberOfBins;

    float threshold = (float) 1.0 / pointsPerBin;

    for (int bin = 0; bin < numberOfBins; ++bin) {
        for (int i = 0; i < pointsPerBin; ++i) {
            int index = bin * pointsPerBin + i;
            if (index < n) {
                bins[bin].push_back(normIndexPairs[index].second);
            }
        }
    }
    // Handle the last bin separately if n is not divisible by numberOfBins
    for (int i = numberOfBins * pointsPerBin; i < n; ++i) {
        bins[numberOfBins - 1].push_back(normIndexPairs[i].second);
    }


    beta = new int[k * numberOfBins];
    alpha = new int[k * numberOfBins];
    probabilities = new float [k * numberOfBins];

    std::fill(beta, beta + k * numberOfBins, 2.0);
    std::fill(alpha, alpha + k * numberOfBins, 2.0);
    std::fill(probabilities, probabilities + k * numberOfBins, 0.5);


    while ((iterations < maxIterations) && (! converged)) {
        ++iterations;


        for (const auto& pair : bins) {
            int bin = pair.first;
            int count = 0;

            for (int j = 0; j < k; ++j) {
                if (bin == 0) {
                    if ((probabilities[bin * k + j] < threshold) && (probabilities[(bin + 1) * k + j] < threshold)) {
                        continue;
                    }
                } else if (bin == numberOfBins - 1) {
                    if ((probabilities[bin * k + j] < threshold) && (probabilities[(bin - 1) * k + j] < threshold)) {
                        continue;
                    }
                } else {
                    if ((probabilities[bin * k + j] < threshold) && (probabilities[(bin - 1) * k + j] < threshold) &&
                        (probabilities[(bin + 1) * k + j] < threshold)) {
                        continue;
                    }
                }
                listOfEligibleCenters[count++] = j;
            }


            for (int i : pair.second) {
                int closest = assignment[i];
                float closestDist2 = sqrt(distanceCalculation(x->data + i * x->d, x->data + (i + 1) * x->d, centers->data + assignment[i] * x->d));
                for (int j = 0; j < count; ++j) {
                    if (listOfEligibleCenters[j] == assignment[i]){
                        continue;
                    }
                    float d2 = sqrt(distanceCalculation(x->data + i * x->d, x->data + (i + 1) * x->d, centers->data + listOfEligibleCenters[j] * x->d));
                    #ifdef COUNT_DISTANCES
                    numDistances += 1;
                    #endif
                    if (d2 < closestDist2) {
                        closest = listOfEligibleCenters[j];
                        closestDist2 = d2;
                    }
                }
                for (int j = 0; j < k; ++j) {
                    if (closest == j) {
                        alpha[bin * k + j]++;
                    }else{
                        beta[bin * k + j]++;
                    }
                }

                if (assignment[i] != closest) {
                    changeAssignment(i, closest, threadId);
                }
            }

        }

        for (int i = 0; i < k * numberOfBins; i++){
            probabilities[i] = (float)(alpha[i] - 1) / (alpha[i] + beta[i] - 2);
        }

        verifyAssignment(iterations, startNdx, endNdx);

        synchronizeAllThreads();

        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();
    }
    //std::cout << "Calculated " << calculated << ", Skipped " << skipped << " Ratio " << (float) calculated / (calculated + skipped) << std::endl;
    return iterations;
}

