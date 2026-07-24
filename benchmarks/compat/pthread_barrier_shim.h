#pragma once
// macOS compatibility shim: Apple's libpthread does not implement the optional
// POSIX pthread_barrier API that fast-kmeans (extern/fast-kmeans) relies on under
// USE_THREADS. Force-included (via -include) into those translation units on Apple
// platforms only; a no-op elsewhere (Linux, where the paper's results were produced).
#if defined(__APPLE__)

#include <errno.h>
#include <pthread.h>

#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD 1

typedef int pthread_barrierattr_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned int count;   // threads required to trip the barrier
    unsigned int waiting; // threads currently waiting
    unsigned int phase;   // generation counter to avoid spurious wakeups
} pthread_barrier_t;

static inline int
pthread_barrier_init(pthread_barrier_t* b, const pthread_barrierattr_t* attr, unsigned int count) {
    (void) attr;
    if (count == 0) {
        errno = EINVAL;
        return -1;
    }
    if (pthread_mutex_init(&b->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&b->cond, NULL) != 0) {
        pthread_mutex_destroy(&b->mutex);
        return -1;
    }
    b->count = count;
    b->waiting = 0;
    b->phase = 0;
    return 0;
}

static inline int pthread_barrier_wait(pthread_barrier_t* b) {
    pthread_mutex_lock(&b->mutex);
    unsigned int phase = b->phase;
    if (++b->waiting == b->count) {
        b->phase++;
        b->waiting = 0;
        pthread_cond_broadcast(&b->cond);
        pthread_mutex_unlock(&b->mutex);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    while (phase == b->phase) {
        pthread_cond_wait(&b->cond, &b->mutex);
    }
    pthread_mutex_unlock(&b->mutex);
    return 0;
}

static inline int pthread_barrier_destroy(pthread_barrier_t* b) {
    pthread_mutex_destroy(&b->mutex);
    pthread_cond_destroy(&b->cond);
    return 0;
}

#endif // PTHREAD_BARRIER_SERIAL_THREAD
#endif // __APPLE__
