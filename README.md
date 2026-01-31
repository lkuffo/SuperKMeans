<h1 align="center">
  Super K-Means
<!-- <div align="center"> -->
    <!-- <a href="https://arxiv.org/pdf/2503.04422"><img src="https://img.shields.io/badge/Paper-SIGMOD'25%3A_PDX-blue" alt="Paper" /></a> -->
    <!-- <a href="https://github.com/cwida/PDX/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cwida/PDX?cacheSeconds=3600" alt="License" /></a>
    <a href="https://github.com/cwida/PDX/stargazers"><img src="https://img.shields.io/github/stars/cwida/PDX" alt="GitHub stars" /></a> -->
<!-- </div> -->
</h1>
<h3 align="center">
  The fastest clustering library for high-dimensional vector embeddings
</h3>

<p align="center">
        <img src="./benchmarks/results/plots/github_1.png" alt="SuperKMeans vs FAISS and Scikit Learn" style="{max-height: 150px}">
</p>

> [!WARNING]
> WiP (under submission for VLDB'26). 
> The library is already usable! But, if you stumble upon this repository, 
> contact lxkr@cwi.nl for more information!

High number of clusters? No problem! SuperKMeans scales like charm:

<p align="center">
        <img src="./benchmarks/results/plots/github_2.png" alt="SuperKMeans vs FAISS and Scikit Learn" style="{max-height: 150px}">
</p>

## Why Super K-Means?
- You can save up to 10x time when indexing large-scale high-dimensional vector embeddings (Cohere, OpenAI, Contriever, MXBAI, CLIP, MiniLM, GIST).
- You need a lightweight and faster alternative to FAISS clustering.
- We are efficient in both ARM and x86 machines.

## Usage
```py
from superkmeans import SuperKMeans

data = ...
k = 1000
d = 768

kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d
)

# Run the clustering
centroids = kmeans.train(data) # 2D array with centroids (k x d) 

# Get assignments
assignments = kmeans._assignments

# Or, assign new points:
new_data = ...
new_assignments = kmeans.assign(new_data)
```

Then, you can use the `centroids` to create an IVF index for Vector Search, for example, in FAISS.

## Installation
We provide Python bindings for ease of use. 

### Prerequisites
- C++17, CMake 3.26
- OpenMP
- A BLAS implementation 

```sh
git clone https://github.com/lkuffo/SuperKMeans.git
git submodule update --init
pip install . 
```

For a more comprehensive installation guide, check [INSTALL.md](./INSTALL.md).

## Getting the Best Performance
Check [INSTALL.md](./INSTALL.md).

## Roadmap
We are actively developing Super K-Means and accepting contributions! These are our current priorities:
- Hierarchical K-Means 
- 64-bit `double` support
- 16-bit `half` support
- 8-bit `uint8` support (experimental)

## Benchmarking
To run our benchmark suite in C++, refer to [BENCHMARKING.md](./BENCHMARKING.md).