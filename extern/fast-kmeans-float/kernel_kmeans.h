#ifndef KERNEL_KMEANS_H
#define KERNEL_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * KernelKmeans is a base class for all k-means algorithms that use kernels.
 * Kernel-based algorithms don't represent the centers explicitly, but instead
 * implicitly by the memberships in each cluster.
 */

#include "kmeans.h"
#include "general_functions.h"
#include <cmath>
#include <vector>
#include <cassert>
#include <sstream>
#include <numeric>

class Kernel {
    public:
        virtual ~Kernel() {}
        virtual float operator()(float const *, float const *, int) const = 0;
        virtual std::string getName() const = 0;
};

class LinearKernel : public Kernel {
    public:
        virtual float operator()(float const *a, float const *b, int dimension) const { 
            return std::inner_product(a, a + dimension, b, 0.0);
        }
        virtual std::string getName() const { return "linear"; }
};

class PolynomialKernel : public Kernel {
    public:
        PolynomialKernel(float cc, float p) : c(cc), power(p) {}
        virtual float operator()(float const *a, float const *b, int dimension) const { 
            return pow(std::inner_product(a, a + dimension, b, 0.0) + c, power);
        }
        virtual std::string getName() const {
            std::ostringstream out;
            out << "poly[" << c << "," << power << "]";
            return out.str();
        }
    protected:
        float c, power;
};

class GaussianKernel : public Kernel {
    public:
        GaussianKernel(float t) : tau(t), twoTau2(2.0 * t * t) {}
        virtual float operator()(float const *a, float const *b, int dimension) const { 
            float d2 = std::inner_product(a, a + dimension, a, 0.0)
                        - 2 * std::inner_product(a, a + dimension, b, 0.0)
                        + std::inner_product(b, b + dimension, b, 0.0);
            return exp(-d2 / twoTau2);
        }
        virtual std::string getName() const {
            std::ostringstream out;
            out << "gaussian[" << tau << "]";
            return out.str();
        }
    protected:
        float tau;
        float twoTau2;
};

class KernelKmeans : public Kmeans {
    public:
        KernelKmeans(Kernel const *k);
        virtual ~KernelKmeans() { free(); delete &kernel; }
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual void free();

    protected:
        // Functions for computing inner products with kernels.
        float centerCenterInnerProductGeneral(std::vector<unsigned int> const &members1, std::vector<unsigned int> const &members2) const;
        float pointCenterInnerProductGeneral(int xndx, std::vector<unsigned int> const &members) const;

        virtual float centerCenterInnerProduct(unsigned short c1, unsigned short c2) const {
            return centerCenterInnerProductGeneral(memberships[c1], memberships[c2]);
        }
        virtual float pointCenterInnerProduct(int xndx, unsigned short cluster) const {
            return pointCenterInnerProductGeneral(xndx, memberships[cluster]);
        }
        virtual float pointPointInnerProduct(int x1, int x2) const {
            return kernel(x->data + x1 * d, x->data + x2 * d, d);
        }

        // Compute the memberships and center inner product for the points
        // assigned to this thread. 
        void computeMemberships(int threadId, std::vector<std::vector<unsigned int> > *membershipResult, std::vector<float> *ccResult);

        // Convenience function to determine convergence across multiple
        // threads.
        void setConverged(bool aConverged) {
            #ifdef USE_THREADS
            pthread_mutex_lock(convergenceLock);
            #endif
            converged = aConverged;
            #ifdef USE_THREADS
            pthread_mutex_unlock(convergenceLock);
            #endif
        }

        // Convenience function for locking/unlocking a cluster.
        void lockCluster(unsigned short j) {
            #ifdef USE_THREADS
            pthread_mutex_lock(&clusterLocks[j]);
            #endif
        }

        void unlockCluster(unsigned short j) {
            #ifdef USE_THREADS
            pthread_mutex_unlock(&clusterLocks[j]);
            #endif
        }

        // A reference to the kernel this algorithm is using.
        Kernel const &kernel;

        #ifdef USE_THREADS
        // Method of detecting convergence across multiple threads.
        pthread_mutex_t *convergenceLock;
        std::vector<pthread_mutex_t> clusterLocks;
        #endif

        // For each cluster, a list of the indexes of the points that are
        // members in that cluster.
        std::vector<std::vector<unsigned int> > memberships;

        // The inner product for each center with itself.
        std::vector<float> cc;
};


#endif

