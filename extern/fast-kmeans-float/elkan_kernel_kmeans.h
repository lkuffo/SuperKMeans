#ifndef ELKAN_KERNEL_KMEANS_H
#define ELKAN_KERNEL_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * A kernelized version of Elkan's algorithm. It would be nice to make this
 * inherit from both KernelKmeans and ElkanKmeans, but for now we basically
 * replicate code.
 */

#include "kernel_kmeans.h"

class ElkanKernelKmeans : public KernelKmeans {
    public:
        ElkanKernelKmeans(Kernel const *k) : KernelKmeans(k), centerCenterDistDiv2(NULL), s(NULL), upper(NULL), lower(NULL) {}
        virtual ~ElkanKernelKmeans() { free(); }
        virtual std::string getName() const {
            std::ostringstream out;
            out << "elkan_kernel(" << kernel.getName() << ")";
            return out.str();
        }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);

    protected:
        int runThread(int threadId, int maxIterations);
        void update_bounds(int startNdx, int endNdx);
        void update_center_dists(int threadId);
        void computeCenterMovement(int threadId);

        // Matrix in an array of distance between each center, divided by 2.
        float *centerCenterDistDiv2;

        // Distance between center j and its closest other center, divided by 2.
        float *s;

        // Upper bound for each point.
        float *upper;

        // Matrix in an array of k lower bounds for each point.
        float *lower;

        // These are used to hold the new memberships and new squared center
        // norms (Cc means <c,c>) from one iteration to the next.
        std::vector<std::vector<unsigned int> > newMemberships;
        std::vector<float> newCc;
};

#endif

