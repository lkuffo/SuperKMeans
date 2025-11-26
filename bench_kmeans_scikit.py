import os
threads = "10"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["BLIS_NUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads


from sklearn.cluster import KMeans
import numpy as np
import os
import time
import math
import sys
import fastkmeans
from fastkmeans import FastKMeans

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "glove100"
    dataset_params = {
        "mxbai": (769_382, 1024),
        "openai": (999_000, 1536),
        "arxiv": (2_253_000, 768),
        "sift": (1_000_000, 128),
        "fmnist": (60_000, 784),
        "glove100": (1_183_514, 100),
        "glove50": (1_183_514, 50)
    }

    if dataset not in dataset_params:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from {list(dataset_params.keys())}"
        )
    num_vectors, num_dimensions = dataset_params[dataset]
    num_centroids = max(1, int(math.sqrt(num_vectors) * 4))
    threads = 10
    n_iter = 25

    print(f"Dataset: {dataset}")
    print(f"num_vectors={num_vectors}, num_dimensions={num_dimensions}")
    print(f"num_centroids={num_centroids}, threads={threads}, n_iter={n_iter}")

    # Load data file (expects float32, row-major, n*d values)
    filename = f"data_{dataset}.bin"
    data = np.fromfile(filename, dtype=np.float32)
    if data.size != num_vectors * num_dimensions:
        raise ValueError(
            f"File size mismatch: got {data.size} floats, "
            f"expected {num_vectors * num_dimensions}"
        )
    data = data.reshape(num_vectors, num_dimensions)

    start = time.time()
    km = KMeans(
        n_clusters=num_centroids,
        init='random',
        n_init=1,
        max_iter=n_iter,
        verbose=1,
        random_state=42,
        copy_x=True
    )
    km.fit(data)
    end = time.time()
    print(f"ScikitLearn Took: {(end - start):.2f} s")

    start = time.time()
    kmeans = FastKMeans(
        d=num_dimensions,
        k=num_centroids,
        niter=n_iter,
        tol=-math.inf,
        device='cpu',
        seed=42,
        max_points_per_centroid=256,
        verbose=True,
        use_triton=False,
    )
    kmeans.train(data)
    end = time.time()
    print(f"FastKMeans Took: {(end - start):.2f} s")
