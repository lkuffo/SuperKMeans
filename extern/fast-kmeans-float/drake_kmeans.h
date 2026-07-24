#ifndef DRAKE_KMEANS_H
#define DRAKE_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * DrakeKmeans implements Drake's k-means algorithm that uses a user-defined
 * number of lower bounds.
 */

#include "triangle_inequality_base_kmeans.h"
#include <utility>
#include <sstream>

class DrakeKmeans : public TriangleInequalityBaseKmeans {
    public:
        DrakeKmeans(int aNumBounds);
        virtual ~DrakeKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "drake"; }
    
    protected:
        virtual int runThread(int threadId, int maxIterations);

        // For each point, the indexes of the closest centers other than the
        // assigned center. Size is n * numLowerBounds.
        unsigned short **closestOtherCenters;
    
        // Find the centers that are nearest point i. The parameter "order" is
        // an array of pairs that will contain the closest
        // numLowerBoundsRemaining centers and their distances. This method is
        // called when all bounds failed to prune distance calculations, and the
        // algorithm must look at all centers.
        void find_near_centers(int i, int numLowerBoundsRemaining, std::pair<float, int> *order, int threadId);

        // If the nearest centers need reordering, this method handles that.
        // This method is called when one of the bounds succeeded in pruning a
        // distance calculation, but we need to discover the true order of the
        // nearest centers.
        void reorder_near_centers(int i, int nearIndex, std::pair<float, int> *order, int threadId);

        // Update the upper and lower bounds for the given points.  Assumes that
        // centerMovement has been computed.
        void update_bounds(int startNdx, int endNdx, int numLowerBoundsRemaining);
};

#endif
