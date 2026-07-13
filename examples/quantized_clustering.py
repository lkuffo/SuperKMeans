import numpy as np
import sys
import time
from sklearn.datasets import make_blobs
from superkmeans import SuperKMeans


def print_usage(program_name):
    print(f"Usage: {program_name} [quantizer] [n] [d] [k]")
    print("  quantizer: f32, sq8, lvq4, or rabitq (default: sq8)")
    print("  n: Number of vectors (default: 1000000)")
    print("  d: Dimensionality (default: 768)")
    print("  k: Number of clusters (default: 1000)")
    print("Example:")
    print(f"  {program_name} rabitq 500000 512 100")


def main():
    quantizer = "sq8"
    n = 1000000
    d = 768
    k = 1000

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage(sys.argv[0])
            return 0
        quantizer = sys.argv[1]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        d = int(sys.argv[3])
    if len(sys.argv) > 4:
        k = int(sys.argv[4])

    print(f"Parameters: quantizer={quantizer}, n={n}, d={d}, k={k}")
    print(f"Generating {n} vectors with {d} dimensions...")
    X, _ = make_blobs(n_samples=n, n_features=d, centers=100, random_state=42)
    data = X.astype(np.float32)

    kmeans = SuperKMeans(
        n_clusters=k,
        dimensionality=d,
        quantizer=quantizer,
    )

    print("Generating centroids...")
    start_time = time.time_ns()
    centroids = kmeans.train(data)
    end_time = time.time_ns()
    print(f"Index built in {(end_time - start_time) / 1000000:.2f} milliseconds")

    print("Assigning points to clusters...")
    assignments = kmeans.quantized_assign(data, centroids)

    print("Done!")



if __name__ == "__main__":
    main()
