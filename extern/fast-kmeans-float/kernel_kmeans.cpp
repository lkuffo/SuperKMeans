/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "kernel_kmeans.h"
#include <cassert>

KernelKmeans::KernelKmeans(Kernel const *k) : kernel(*k) {
    #ifdef USE_THREADS
    convergenceLock = NULL;
    #endif
}

void KernelKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    Kmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    cc.resize(k);
    memberships.clear();
    memberships.resize(k);

    #ifdef USE_THREADS
    convergenceLock = new pthread_mutex_t;
    pthread_mutex_init(convergenceLock, NULL);

    clusterLocks.resize(k);
    for (int j = 0; j < k; ++j) {
        pthread_mutex_init(&clusterLocks[j], NULL);
    }
    #endif
}

void KernelKmeans::free() {
    Kmeans::free();
    cc.clear();
    memberships.clear();

    #ifdef USE_THREADS
    if (convergenceLock) {
        pthread_mutex_destroy(convergenceLock);
        delete convergenceLock;
        convergenceLock = NULL;
    }
    #endif
}

float KernelKmeans::centerCenterInnerProductGeneral(std::vector<unsigned int> const &members1, std::vector<unsigned int> const &members2) const {
    float s = 0.0;
    std::vector<unsigned int>::const_iterator i, j;
    if (&members1 == &members2) {
        for (i = members1.begin(); i != members1.end(); ++i) {
            s += kernel(x->data + *i * d, x->data + *i * d, d);
            for (j = i + 1; j != members1.end(); ++j) {
                s += 2.0 * kernel(x->data + *i * d, x->data + *j * d, d);
            }
        }
    } else {
        for (i = members1.begin(); i != members1.end(); ++i) {
            for (j = members2.begin(); j != members2.end(); ++j) {
                s += kernel(x->data + *i * d, x->data + *j * d, d);
            }
        }
    }

    size_t n = members1.size() * members2.size();
    if (n > 0) { s /= n; }

    return s;
}

float KernelKmeans::pointCenterInnerProductGeneral(int i, std::vector<unsigned int> const &members) const {
    float s = 0.0;
    std::vector<unsigned int>::const_iterator j;
    for (j = members.begin(); j != members.end(); ++j) {
        s += kernel(x->data + i * d, x->data + *j * d, d);
    }

    size_t n = members.size();
    if (n > 0) { s /= n; }

    return s;
}

void KernelKmeans::computeMemberships(int threadId, std::vector<std::vector<unsigned int> > *membershipResult, std::vector<float> *ccResult) {
    std::vector<std::vector<unsigned int> > threadMemberships(k);

    if (threadId == 0) {
        for (int j = 0; j < k; ++j) {
            membershipResult->at(j).clear();
        }
    }
    synchronizeAllThreads();

    for (int i = start(threadId); i < end(threadId); ++i) {
        threadMemberships[assignment[i]].push_back(i);
    }
    synchronizeAllThreads();

    for (int j = 0; j < k; ++j) {
        lockCluster(j);
        std::copy(threadMemberships[j].begin(), threadMemberships[j].end(), std::back_insert_iterator<std::vector<unsigned int> >(membershipResult->at(j)));
        unlockCluster(j);
    }
    synchronizeAllThreads();

    for (int j = 0; j < k; ++j) {
        if (j % numThreads == threadId) {
            ccResult->at(j) = centerCenterInnerProductGeneral(membershipResult->at(j), membershipResult->at(j));
        }
    }
}

