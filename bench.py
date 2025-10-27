import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import numpy as np
import resource
from threadpoolctl import threadpool_info, threadpool_limits

if __name__ == '__main__':
    n = 4096
    d = 768

    X = np.random.randn(n, d).astype(np.float32)
    print(X.shape)

    start = time.time()
    for j in range(5000):
        Y = X.transpose().copy()
    end = time.time()
    print(f"Took: {(end - start):.2f} s")

    R, C = X.shape
    nblocks = R // 64
    Y = X.transpose().copy()
    start = time.time()
    for j in range(5000):
        for x in range(0, n, 64):
            Y[:, x:x+64] = X[x:x+64, :].transpose().copy()
    end = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_time = usage.ru_utime + usage.ru_stime
    print(f"Took: {(end - start):.2f} s")

    # reshape into blocks, swap axes inside each block, then stack
    start = time.time()
    for j in range(5000):
        Y = X.reshape(nblocks, 64, C).transpose(0, 2, 1).reshape(nblocks * C, 64)
    end = time.time()
    print(f"Took: {(end - start):.2f} s")


