// Copyright (c) 2017 Facebook Inc.
// Copyright (c) 2015-2017 Georgia Institute of Technology
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Needed for syscall.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif  // _GNU_SOURCE

/* Standard C headers */
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Configuration header */
#include <fxdiv.h>
#include "threadpool-common.h"

/* POSIX headers */
#if PTHREADPOOL_USE_PTHREADS
#include <pthread.h>
#include <unistd.h>
#else
#include <threads.h>
#endif  // PTHREADPOOL_USE_PTHREADS

/* Futex-specific headers */
#if PTHREADPOOL_USE_FUTEX
#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>

/* Old Android NDKs do not define SYS_futex and FUTEX_PRIVATE_FLAG */
#ifndef SYS_futex
#define SYS_futex __NR_futex
#endif  // SYS_futex

#ifndef FUTEX_PRIVATE_FLAG
#define FUTEX_PRIVATE_FLAG 128
#endif  // FUTEX_PRIVATE_FLAG

#elif defined(__EMSCRIPTEN__)
/* math.h for INFINITY constant */
#include <emscripten/threading.h>
#include <math.h>

#else
#error \
    "Platform-specific implementation of futex_wait and futex_wake_all required"
#endif  // defined(__linux__)
#endif  // PTHREADPOOL_USE_FUTEX

/* Windows-specific headers */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sysinfoapi.h>
#endif

/* Dependencies */
#if PTHREADPOOL_USE_CPUINFO
#include <cpuinfo.h>
#endif

/* Alloca */
#if defined(_MSC_VER)
#include <malloc.h>
#define alloca _alloca
#elif !defined(alloca) && defined(__GNUC__)
#define alloca(s) __builtin_alloca(s)
#else
#include <alloca.h>
#endif

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-atomics.h"
#include "threadpool-object.h"
#include "threadpool-utils.h"

/* Logging-related headers */
#ifndef PTHREADPOOL_DEBUG_LOGGING
// Default value if not specified at compile time.
#define PTHREADPOOL_DEBUG_LOGGING 0
#endif  // PTHREADPOOL_DEBUG_LOGGING

#if PTHREADPOOL_DEBUG_LOGGING
#include <stdio.h>

#if defined(__ARM_ARCH)
static uint64_t __rdtsc() {
  uint64_t val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}
#endif  // defined(__ARM_ARCH)
static size_t pthreadpool_get_ticks(size_t since) { return __rdtsc() - since; }

#define pthreadpool_log_debug(format, ...)                                    \
  fprintf(stderr, "[%zu] %s (%s:%i): " format "\n", pthreadpool_get_ticks(0), \
          __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);
#else
#define pthreadpool_log_debug(format, ...)
#endif  // PTHREADPOOL_DEBUG_LOGGING

#if PTHREADPOOL_USE_FUTEX
#if defined(__linux__)
static int futex_wait(pthreadpool_atomic_uint32_t* address, uint32_t value) {
  return syscall(SYS_futex, address, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, value,
                 NULL);
}

static int futex_wake_n(pthreadpool_atomic_uint32_t* address, uint32_t n) {
  return syscall(SYS_futex, address, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, n);
}

static int futex_wake_all(pthreadpool_atomic_uint32_t* address) {
  return futex_wake_n(address, /*n=*/INT_MAX);
}
#elif defined(__EMSCRIPTEN__)
static int futex_wait(pthreadpool_atomic_uint32_t* address, uint32_t value) {
  return emscripten_futex_wait((volatile void*)address, value, INFINITY);
}

static int futex_wake_n(pthreadpool_atomic_uint32_t* address, uint32_t n) {
  return emscripten_futex_wake((volatile void*)address, n);
}

static int futex_wake_all(pthreadpool_atomic_uint32_t* address) {
  return emscripten_futex_wake((volatile void*)address, INT_MAX);
}
#else
#error \
    "Platform-specific implementation of futex_wait and futex_wake_all required"
#endif
#endif

size_t pthreadpool_set_threads_count(struct pthreadpool* threadpool,
                                     size_t num_threads) {
  if (threadpool == NULL) {
    return 1;
  }
  /* We shouldn't change this while a parallel computation is running. */
  pthreadpool_mutex_lock(&threadpool->execution_mutex);

  // Adjust `num_threads` to the feasible limits.
  if (num_threads == 0) {
    // Set to the maximum number of threads.
    num_threads = threadpool->max_num_threads;
  } else {
    num_threads = max(num_threads, 1);
    num_threads = min(num_threads, threadpool->max_num_threads);
  }

  // Check whether this is really a change.
  if (num_threads != threadpool->threads_count) {
    pthreadpool_store_release_size_t(&threadpool->threads_count, num_threads);
  }
  pthreadpool_log_debug("setting max_num_threads to %zu.", num_threads);

  pthreadpool_mutex_unlock(&threadpool->execution_mutex);

  return num_threads;
}

static void wait_on_num_recruited_threads(pthreadpool_t threadpool,
                                          uint32_t expected) {
  uint32_t num_recruited_threads =
      pthreadpool_load_acquire_uint32_t(&threadpool->num_recruited_threads);

#if !PTHREADPOOL_USE_FUTEX
  if (num_recruited_threads != expected) {
    pthreadpool_mutex_lock(&threadpool->completion_mutex);
#endif  // !PTHREADPOOL_USE_FUTEX

    for (size_t iter = 0; num_recruited_threads != expected; iter++) {
      // Just spin for the first few iterations.
      if (iter < PTHREADPOOL_SPIN_WAIT_ITERATIONS) {
        pthreadpool_yield(iter);
      } else {
        pthreadpool_log_debug("waiting on %u recruited threads...",
                              num_recruited_threads - expected);
#if PTHREADPOOL_USE_FUTEX
        futex_wait(&threadpool->num_recruited_threads, num_recruited_threads);
#else
        pthreadpool_cond_wait(&threadpool->completion_condvar,
                              &threadpool->completion_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
      }
      num_recruited_threads =
          pthreadpool_load_acquire_uint32_t(&threadpool->num_recruited_threads);
    }

#if !PTHREADPOOL_USE_FUTEX
    pthreadpool_mutex_unlock(&threadpool->completion_mutex);
  }
#endif  // !PTHREADPOOL_USE_FUTEX
}

static int32_t wait_on_num_active_threads(pthreadpool_t threadpool,
                                          uint32_t thread_id) {
  int32_t curr_active_threads =
      pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);

  if (curr_active_threads <= 0) {
#if !PTHREADPOOL_USE_FUTEX
    pthreadpool_mutex_lock(&threadpool->num_active_threads_mutex);
#endif  // !PTHREADPOOL_USE_FUTEX

    for (size_t iter = 0; curr_active_threads <= 0; iter++) {
      // Just spin for the first few iterations.
      if (!(threadpool->flags & PTHREADPOOL_FLAG_DONT_SPIN_WORKERS) &&
          iter < PTHREADPOOL_SPIN_WAIT_ITERATIONS) {
        pthreadpool_yield(iter);
      }

      // If we've borrowed this thread from an executor, then we should return
      // it to the executor instead of blocking it indefinitely.
      else if (threadpool->executor.num_threads) {
        return PTHREADPOOL_NUM_ACTIVE_THREADS_DONE;
      }

      // Otherwise, put this thread to sleep until `num_waiting_threads` is
      // larger than zero.
      else {
#if PTHREADPOOL_USE_FUTEX
        // First increase the `num_waiting_threads` counter and re-check
        // `num_active_threads` thereafter to avoid slipping past calls to
        // `signal_num_active_threads`.
        pthreadpool_fetch_add_acquire_release_uint32_t(
            &threadpool->num_waiting_threads, 1);
        if ((curr_active_threads = pthreadpool_load_consume_int32_t(
                 &threadpool->num_active_threads)) <= 0) {
          // Use futex/condition signaling.
          pthreadpool_log_debug(
              "thread %u waiting on change in num active threads (curr=%i)...",
              thread_id, curr_active_threads);
          futex_wait(
              (pthreadpool_atomic_uint32_t*)&threadpool->num_active_threads,
              curr_active_threads);
        }
        pthreadpool_decrement_fetch_acquire_release_uint32_t(
            &threadpool->num_waiting_threads);
#else
        pthreadpool_log_debug(
            "thread %u waiting on change in num active threads (curr=%i)...",
            thread_id, curr_active_threads);
        pthreadpool_cond_wait(&threadpool->num_active_threads_condvar,
                              &threadpool->num_active_threads_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
      }

      curr_active_threads =
          pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);
    }

#if !PTHREADPOOL_USE_FUTEX
    pthreadpool_mutex_unlock(&threadpool->num_active_threads_mutex);
#endif  // !PTHREADPOOL_USE_FUTEX
  }

  return curr_active_threads;
}

static void wait_on_work_is_done(pthreadpool_t threadpool) {
  int32_t work_is_done =
      pthreadpool_exchange_acquire_uint32_t(&threadpool->work_is_done, 0);

#if !PTHREADPOOL_USE_FUTEX
  if (!work_is_done) {
    pthreadpool_mutex_lock(&threadpool->completion_mutex);
#endif  // !PTHREADPOOL_USE_FUTEX

    for (size_t iter = 0; !work_is_done; iter++) {
      // Just spin for the first few iterations.
      if (iter < PTHREADPOOL_SPIN_WAIT_ITERATIONS) {
        pthreadpool_yield(iter);
      } else {
        // Use futex/condition signaling.
        pthreadpool_log_debug("thread waiting on work_is_done...");
#if PTHREADPOOL_USE_FUTEX
        futex_wait((pthreadpool_atomic_uint32_t*)&threadpool->work_is_done,
                   work_is_done);
#else
        pthreadpool_cond_wait(&threadpool->completion_condvar,
                              &threadpool->completion_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
      }

      work_is_done =
          pthreadpool_exchange_acquire_uint32_t(&threadpool->work_is_done, 0);
    }

#if !PTHREADPOOL_USE_FUTEX
    pthreadpool_mutex_unlock(&threadpool->completion_mutex);
  }
#endif  // !PTHREADPOOL_USE_FUTEX
}

static void signal_num_recruited_threads(pthreadpool_t threadpool) {
#if PTHREADPOOL_USE_FUTEX
  futex_wake_all(&threadpool->num_recruited_threads);
#else
  pthreadpool_mutex_lock(&threadpool->completion_mutex);
  pthreadpool_cond_signal(&threadpool->completion_condvar);
  pthreadpool_mutex_unlock(&threadpool->completion_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
}

static void signal_num_active_threads(pthreadpool_t threadpool,
                                      uint32_t max_num_waiting) {
#if PTHREADPOOL_USE_FUTEX
  const uint32_t num_waiting_threads =
      pthreadpool_load_consume_uint32_t(&threadpool->num_waiting_threads);
  if (num_waiting_threads > max_num_waiting) {
    futex_wake_n((pthreadpool_atomic_uint32_t*)&threadpool->num_active_threads,
                 num_waiting_threads - max_num_waiting);
  }
#else
  pthreadpool_mutex_lock(&threadpool->num_active_threads_mutex);
  pthreadpool_cond_broadcast(&threadpool->num_active_threads_condvar);
  pthreadpool_mutex_unlock(&threadpool->num_active_threads_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
}

static void signal_work_is_done(pthreadpool_t threadpool) {
  uint32_t prev_value = pthreadpool_exchange_acquire_release_uint32_t(
      &threadpool->work_is_done, 1);
  assert(prev_value == 0);
#if PTHREADPOOL_USE_FUTEX
  futex_wake_all(&threadpool->work_is_done);
#else
  pthreadpool_mutex_lock(&threadpool->completion_mutex);
  pthreadpool_cond_signal(&threadpool->completion_condvar);
  pthreadpool_mutex_unlock(&threadpool->completion_mutex);
#endif  // PTHREADPOOL_USE_FUTEX
}

static void pthreadpool_register_threads(pthreadpool_t threadpool,
                                         uint32_t num_threads) {
  pthreadpool_fetch_add_acquire_release_uint32_t(
      &threadpool->num_recruited_threads, num_threads);
}

static void pthreadpool_release(pthreadpool_t threadpool) {
  // If we're the last pending thread of a "done" threadpool, signal for
  // any waiting cleanup and bail before anything else can go wrong (e.g.
  // the `threadpool` might get cleaned up).
  if (pthreadpool_decrement_fetch_acquire_release_uint32_t(
          &threadpool->num_recruited_threads) == 0) {
    signal_num_recruited_threads(threadpool);
  }
}

static void run_thread_function(struct pthreadpool* threadpool,
                                uint32_t thread_id) {
  // Save the current FPU state, if requested.
  const uint32_t flags = pthreadpool_load_relaxed_uint32_t(&threadpool->flags);
  struct fpu_state saved_fpu_state = {0};
  if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
    saved_fpu_state = get_fpu_state();
    disable_fpu_denormals();
  }

  // Call the job function.
  const thread_function_t thread_function =
      pthreadpool_load_relaxed_void_p(&threadpool->thread_function);
  thread_function(threadpool, &threadpool->threads[thread_id]);

  // Restore the original FPU state in case we clobbered it.
  if (flags & PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
    set_fpu_state(saved_fpu_state);
  }

  pthreadpool_log_debug("thread %u done working on job %u.", thread_id,
                        threadpool->job_id);
}

static uint32_t thread_wrap_up(struct pthreadpool* threadpool,
                               uint32_t thread_id) {
  // Get the current state.
  int32_t curr_active_threads =
      pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);
  assert(curr_active_threads != 0);
  assert(curr_active_threads != PTHREADPOOL_NUM_ACTIVE_THREADS_DONE);

  // If we are the first thread to finish work, flip the state from
  // "running" to "wrapping_up".
  bool first_past_the_post = false;
  while (curr_active_threads > 0 &&
         !(first_past_the_post =
               pthreadpool_compare_exchange_sequentially_consistent_int32_t(
                   &threadpool->num_active_threads, &curr_active_threads,
                   -(curr_active_threads - 1)))) {
  }

  if (first_past_the_post) {
    pthreadpool_log_debug(
        "thread %u switched num_active_threads from `%i' to '%i'.", thread_id,
        curr_active_threads, -(curr_active_threads - 1));
    curr_active_threads = -(curr_active_threads - 1);
  } else {
    curr_active_threads = pthreadpool_fetch_add_acquire_release_int32_t(
                              &threadpool->num_active_threads, 1) +
                          1;
  }

  // If we are the last active thread, let the calling thread know (unless we
  // are the calling thread).
  if (thread_id != 0 && curr_active_threads == 0) {
    pthreadpool_log_debug("thread %u switched num_active_threads to 0.",
                          thread_id);
    signal_work_is_done(threadpool);
  }

  return curr_active_threads;
}

static pthreadpool_thread_return_t thread_main(void* arg);

static void ensure_num_threads(struct pthreadpool* threadpool,
                               uint32_t thread_id) {
  struct pthreadpool_executor* executor = &threadpool->executor;

  /* If we're not using an executor, do nothing. */
  if (!executor->num_threads) {
    return;
  }

  // Get the number of required threads.
  const uint32_t max_num_threads =
      pthreadpool_load_acquire_size_t(&threadpool->threads_count);

  // Start up to two other threads so that we fan out exponentially.
  uint32_t num_threads_to_start = 2;

  /* Schedule any missing threads for this threadpool. */
  for (uint32_t tid = thread_id + 1;
       num_threads_to_start && tid < max_num_threads; tid++) {
    // Check whether this thread was active, and if not, schedule it.
    struct thread_info* thread = &threadpool->threads[tid];
    if (!pthreadpool_load_relaxed_uint32_t(&thread->is_active)) {
      // Make sure there is still ongoing work (unless we're the main thread).
      const int32_t curr_active_threads =
          thread_id ? pthreadpool_load_consume_int32_t(
                          &threadpool->num_active_threads)
                    : 1;
      if (curr_active_threads < 0 ||
          curr_active_threads == PTHREADPOOL_NUM_ACTIVE_THREADS_DONE) {
        return;
      }

      // Schedule this worker thread.
      if (!pthreadpool_exchange_sequentially_consistent_uint32_t(
              &thread->is_active, 1)) {
        // Note that `threadpool->num_recruited_threads` is always non-zero
        // because this function is only ever called by the main thread or
        // another active thread, so we don't have to worry about activating a
        // thread on an already stopped threadpool.
        pthreadpool_register_threads(threadpool, 1);
        executor->schedule(threadpool->executor_context, thread,
                           (void (*)(void*))thread_main);
        num_threads_to_start--;
      }
    }
  }
}

static pthreadpool_thread_return_t thread_main(void* arg) {
  // Unpack the argument, i.e. extract the pointer to the `pthreadpool` from the
  // provided pointer to this thread's `thread_info`.
  struct thread_info* thread = (struct thread_info*)arg;
  const uint32_t thread_id = thread->thread_number;
  struct pthreadpool* threadpool =
      (struct pthreadpool*)((uintptr_t)thread -
                            thread_id * sizeof(struct thread_info) -
                            offsetof(struct pthreadpool, threads));
  uint32_t last_job_id = 0;

  // Get the current threadpool state.
  int32_t curr_active_threads =
      pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);

  // Check whether we have to wake up any other threads.
  if (curr_active_threads > 0 &&
      curr_active_threads != PTHREADPOOL_NUM_ACTIVE_THREADS_DONE) {
    ensure_num_threads(threadpool, thread_id);
    curr_active_threads =
        pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);
  }

  pthreadpool_log_debug("thread %u entering main loop.", thread_id);

  // Main loop.
  while (true) {
    if (curr_active_threads == PTHREADPOOL_NUM_ACTIVE_THREADS_DONE) {
      // Signal our intent to leave the party.
      pthreadpool_store_release_uint32_t(&thread->is_active, 0);

      // Double-check that it really was time to go.
      if (!((curr_active_threads =
                 pthreadpool_load_sequentially_consistent_int32_t(
                     &threadpool->num_active_threads)) > 0 &&
            curr_active_threads != PTHREADPOOL_NUM_ACTIVE_THREADS_DONE) ||
          pthreadpool_exchange_sequentially_consistent_uint32_t(
              &thread->is_active, 1)) {
        // We're done here.
        pthreadpool_log_debug("thread %u leaving main loop.", thread_id);
        break;
      }

    } else if (curr_active_threads <= 0) {
      // If the state is `idle` or `wrapping_up`, wait for a state change to
      // "running".
      curr_active_threads = wait_on_num_active_threads(threadpool, thread_id);

    } else {
      // If the threadpool is currently running a job, try to join in on the
      // work.
      bool got_work = false;
      while (!got_work && curr_active_threads > 0 &&
             curr_active_threads != PTHREADPOOL_NUM_ACTIVE_THREADS_DONE) {
        got_work = pthreadpool_compare_exchange_sequentially_consistent_int32_t(
            &threadpool->num_active_threads, &curr_active_threads,
            curr_active_threads + 1);
      }

      // Did we get a foot in?
      if (got_work) {
        assert(last_job_id < threadpool->job_id);
        last_job_id = threadpool->job_id;

        // Do we already have too many threads working on this?
        const uint32_t max_active_threads =
            pthreadpool_load_acquire_size_t(&threadpool->threads_count);
        if (curr_active_threads < max_active_threads) {
          // Do the needful.
          pthreadpool_log_debug("thread %u working on job %u.", thread_id,
                                threadpool->job_id);
          run_thread_function(threadpool, thread_id);
        }

        // Ring the bell on the way out.
        curr_active_threads = thread_wrap_up(threadpool, thread_id);
      }
    }
  }

  // Release our hold on the threadpool.
  pthreadpool_release(threadpool);

  return 0;
}

static size_t get_num_cpus(void) {
#if PTHREADPOOL_USE_CPUINFO
  return cpuinfo_get_processors_count();
#elif defined(_SC_NPROCESSORS_ONLN)
  size_t num_cpus = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
#if defined(__EMSCRIPTEN_PTHREADS__)
  /* Limit the number of threads to 8 to match link-time PTHREAD_POOL_SIZE
   * option */
  if (num_cpus >= 8) {
    num_cpus = 8;
  }
#endif
  return num_cpus;
#elif defined(_WIN32)
  SYSTEM_INFO system_info;
  ZeroMemory(&system_info, sizeof(system_info));
  GetSystemInfo(&system_info);
  return (size_t)system_info.dwNumberOfProcessors;
#else
#error \
    "Platform-specific implementation of sysconf(_SC_NPROCESSORS_ONLN) required"
#endif
}

struct pthreadpool* pthreadpool_create(size_t threads_count) {
  return pthreadpool_create_v2(/*executor=*/NULL, /*executor_context=*/NULL,
                               threads_count);
}

struct pthreadpool* pthreadpool_create_v2(struct pthreadpool_executor* executor,
                                          void* executor_context,
                                          size_t max_num_threads) {
#if PTHREADPOOL_USE_CPUINFO
  if (!cpuinfo_initialize()) {
    return NULL;
  }
#endif

  if (max_num_threads == 0) {
    max_num_threads = get_num_cpus();
  }

  const uint32_t num_threads =
      executor
          ? min(max_num_threads, executor->num_threads(executor_context) + 1)
          : max_num_threads;
  struct pthreadpool* threadpool = pthreadpool_allocate(num_threads);
  if (threadpool == NULL) {
    return NULL;
  }
  if (executor) {
    threadpool->executor = *executor;
    threadpool->executor_context = executor_context;
  }
  threadpool->max_num_threads = num_threads;
  threadpool->threads_count = num_threads;
  for (size_t tid = 0; tid < num_threads; tid++) {
    threadpool->threads[tid].thread_number = tid;
    threadpool->threads[tid].threadpool = threadpool;
  }
  threadpool->num_active_threads = 0;

  /* Initialize the execution mutex. */
  pthreadpool_mutex_init(&threadpool->execution_mutex);

  if (num_threads > 1) {
#if !PTHREADPOOL_USE_FUTEX
    /* Initialize the condition variables and mutexes. */
    pthreadpool_mutex_init(&threadpool->completion_mutex);
    pthreadpool_cond_init(&threadpool->completion_condvar);
    pthreadpool_mutex_init(&threadpool->num_active_threads_mutex);
    pthreadpool_cond_init(&threadpool->num_active_threads_condvar);
#endif

    /* If we weren't given an executor, start our own threads. */
    if (!executor) {
      /* Caller thread serves as worker #0. Thus, we create system threads
       * starting with worker #1. */
      pthreadpool_register_threads(threadpool, num_threads - 1);
      for (size_t tid = 1; tid < num_threads; tid++) {
        pthreadpool_thread_create(&threadpool->threads[tid].thread_object,
                                  &thread_main, &threadpool->threads[tid]);
      }
    }
  }

  return threadpool;
}

PTHREADPOOL_INTERNAL void pthreadpool_parallelize(
    struct pthreadpool* threadpool, thread_function_t thread_function,
    const void* params, size_t params_size, void* task, void* context,
    size_t linear_range, uint32_t flags) {
  assert(threadpool != NULL);
  assert(thread_function != NULL);
  assert(task != NULL);
  assert(linear_range > 1);

  /* Protect the global threadpool structures */
  pthreadpool_mutex_lock(&threadpool->execution_mutex);

  /* Make changes by other threads visible to this thread. */
  pthreadpool_fence_acquire();

  /* Make sure the threadpool is idle or done. */
  const int32_t num_active_threads =
      pthreadpool_load_consume_int32_t(&threadpool->num_active_threads);
  assert(num_active_threads == 0 ||
         num_active_threads == PTHREADPOOL_NUM_ACTIVE_THREADS_DONE);

  /* Setup global arguments */
  pthreadpool_store_relaxed_void_p(&threadpool->thread_function,
                                   thread_function);
  pthreadpool_store_relaxed_void_p(&threadpool->task, task);
  pthreadpool_store_relaxed_void_p(&threadpool->argument, context);
  pthreadpool_store_relaxed_uint32_t(&threadpool->flags, flags);
  threadpool->job_id += 1;
  if (params_size != 0) {
    memcpy(&threadpool->params, params, params_size);
  }

  // How many threads should we parallelize over?
  const uint32_t prev_num_threads = threadpool->threads_count;
  const uint32_t num_threads = min(linear_range, prev_num_threads);
  pthreadpool_store_relaxed_size_t(&threadpool->threads_count, num_threads);

  pthreadpool_log_debug("main thread starting job %u with %u threads.",
                        (uint32_t)threadpool->job_id, num_threads);

  /* Populate a `thread_info` struct for each thread */
  const struct fxdiv_result_size_t range_params =
      fxdiv_divide_size_t(linear_range, fxdiv_init_size_t(num_threads));
  size_t range_start = 0;
  for (size_t tid = 0; tid < num_threads; tid++) {
    struct thread_info* thread = &threadpool->threads[tid];
    const size_t range_length =
        range_params.quotient + (size_t)(tid < range_params.remainder);
    const size_t range_end = range_start + range_length;
    pthreadpool_store_relaxed_size_t(&thread->range_start, range_start);
    pthreadpool_store_relaxed_size_t(&thread->range_end, range_end);
    pthreadpool_store_relaxed_size_t(&thread->range_length, range_length);

    /* The next subrange starts where the previous ended */
    range_start = range_end;
  }

  // Populate sentinels.
  for (size_t tid = num_threads; tid < threadpool->max_num_threads; tid++) {
    struct thread_info* thread = &threadpool->threads[tid];
    pthreadpool_store_relaxed_size_t(&thread->range_start, range_start);
    pthreadpool_store_relaxed_size_t(&thread->range_end, range_start);
    pthreadpool_store_relaxed_size_t(&thread->range_length, 0);
  }

  /* Make changes by this thread visible to other threads. */
  pthreadpool_fence_release();

  /* Set the number of active threads for this job (currently just this thread).
   */
  pthreadpool_store_sequentially_consistent_int32_t(
      &threadpool->num_active_threads, 1);

  /* Wake up any thread waiting on a change of state. */
  signal_num_active_threads(threadpool,
                            threadpool->max_num_threads - num_threads);

  /* Make sure we have enough threads running. */
  ensure_num_threads(threadpool, /*thread_id=*/0);

  /* Do a bit of work ourselves, as thread zero. */
  run_thread_function(threadpool, /*thread_id=*/0);

  /* If we weren't the last one out, wait for any other threads to finish. */
  if (thread_wrap_up(threadpool, /*thread_id=*/0)) {
    /* Wait for any other threads to finish. */
    wait_on_work_is_done(threadpool);
  }

  /* Make changes by other threads visible to this thread. */
  pthreadpool_fence_acquire();

  /* Re-set the number of threads in case it was reduced for this task. */
  if (prev_num_threads != num_threads) {
    pthreadpool_store_relaxed_size_t(&threadpool->threads_count,
                                     prev_num_threads);
  }

  /* Unprotect the global threadpool structures now that we're done. */
  pthreadpool_mutex_unlock(&threadpool->execution_mutex);
}

static void pthreadpool_release_all_threads(struct pthreadpool* threadpool) {
  if (threadpool != NULL) {
    assert(threadpool->num_active_threads == 0 ||
           threadpool->num_active_threads ==
               PTHREADPOOL_NUM_ACTIVE_THREADS_DONE);

    // Set the state to "done".
    pthreadpool_store_sequentially_consistent_int32_t(
        &threadpool->num_active_threads, PTHREADPOOL_NUM_ACTIVE_THREADS_DONE);
    pthreadpool_log_debug(
        "main thread switching num_active_threads from %i to %i.", 0,
        PTHREADPOOL_NUM_ACTIVE_THREADS_DONE);

    /* Wake up any thread waiting on a change of state. */
    signal_num_active_threads(threadpool, 0);
  }
}

void pthreadpool_release_executor_threads(struct pthreadpool* threadpool) {
  if (threadpool && threadpool->executor.num_threads) {
    pthreadpool_release_all_threads(threadpool);
  }
}

bool pthreadpool_update_executor(pthreadpool_t threadpool,
                                 struct pthreadpool_executor* executor,
                                 void* executor_context) {
  /* Protect the global threadpool structures */
  pthreadpool_mutex_lock(&threadpool->execution_mutex);

  const bool res = threadpool->executor.num_threads != executor->num_threads ||
                   threadpool->executor.schedule != executor->schedule ||
                   threadpool->executor_context != executor_context;
  if (res) {
    pthreadpool_release_executor_threads(threadpool);
    threadpool->executor = *executor;
    threadpool->executor_context = executor_context;
  }

  /* Unprotect the global threadpool structures now that we're done. */
  pthreadpool_mutex_unlock(&threadpool->execution_mutex);

  return res;
}

void pthreadpool_destroy(struct pthreadpool* threadpool) {
  if (threadpool != NULL) {
    /* Tell all threads to stop. */
    pthreadpool_release_all_threads(threadpool);

    // Wait for any recruited threads to leave.
    wait_on_num_recruited_threads(threadpool, 0);

    if (!threadpool->executor.num_threads) {
      /* Wait until all threads return */
      for (size_t thread = 1; thread < threadpool->max_num_threads; thread++) {
        pthreadpool_thread_join(threadpool->threads[thread].thread_object,
                                NULL);
      }
    }

    /* Release resources */
    pthreadpool_mutex_destroy(&threadpool->execution_mutex);
#if !PTHREADPOOL_USE_FUTEX
    pthreadpool_mutex_destroy(&threadpool->num_active_threads_mutex);
    pthreadpool_cond_destroy(&threadpool->num_active_threads_condvar);
    pthreadpool_mutex_destroy(&threadpool->completion_mutex);
    pthreadpool_cond_destroy(&threadpool->completion_condvar);
#endif

#if PTHREADPOOL_USE_CPUINFO
    cpuinfo_deinitialize();
#endif

    pthreadpool_log_debug("destroying threadpool at %p.", threadpool);
    pthreadpool_deallocate(threadpool);
  }
}
