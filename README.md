<h1 align="center">
  Super K-Means
<!-- <div align="center"> -->
    <!-- <a href="https://arxiv.org/pdf/2503.04422"><img src="https://img.shields.io/badge/Paper-SIGMOD'25%3A_PDX-blue" alt="Paper" /></a> -->
    <!-- <a href="https://github.com/cwida/PDX/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cwida/PDX?cacheSeconds=3600" alt="License" /></a>
    <a href="https://github.com/cwida/PDX/stargazers"><img src="https://img.shields.io/github/stars/cwida/PDX" alt="GitHub stars" /></a> -->
<!-- </div> -->
</h1>
<h3 align="center">
  A super fast clustering library for high-dimensional vector embeddings
</h3>

<p align="center">
        <img src="./benchmarks/results/plots/github_1.png" height=220 alt="SuperKMeans vs FAISS and Scikit Learn" style="{max-height: 100px}">
</p>

<h4 align="center">
High number of clusters? No problem! SuperKMeans scales like charm:
</h4>

<p align="center">
        <img src="./benchmarks/results/plots/github_2.png" height=230 alt="SuperKMeans vs FAISS and Scikit Learn" style="{max-height: 100px}">
</p>

> [!IMPORTANT]
> **VLDB'26 reviewers**: For reproducibility of our results, check [BENCHMARKING.md](./BENCHMARKING.md).
> **For other people**: This library is already usable! But, if you stumble upon this repository, contact lxkr@cwi.nl for more information! **Important:** We will soon move this repository to https://github.com/cwida/SuperKMeans.

## Why Super K-Means?
- Up to **10x faster clustering** than FAISS of large-scale high-dimensional vector embeddings (Cohere, OpenAI, Contriever, MXBAI, CLIP, MiniLM, GIST).
- Faster **without compromising clustering quality**.
- Efficient in **CPUs** (ARM and x86) and **GPUs**.

## Our secret sauce
- Reliable and efficient **pruning of dimensions**.
- We will release our paper with all the details soon!
- In the benchmarks you see in the cover image, **all algorithms are clustering the same data**: No dimensionality reduction, no sampling, same number of iterations, no early-termination.

## Usage
```py
from superkmeans import SuperKMeans

data = ... # Numpy 2D matrix
k = 1000
d = 768

kmeans = SuperKMeans(
    n_clusters=k,
    dimensionality=d
)

# Run the clustering
centroids = kmeans.train(data) # 2D array with centroids (k x d) 

# Get assignments
assignments = kmeans.assign(data)
```

Then, you can use the `centroids` to create an IVF index for Vector Search, for example, in FAISS.

<details>

<summary>Usage in C++</summary>

```c++
#include <vector>
#include <cstddef>
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    std::vector<float> data; // Fill
    size_t k = 1000;
    size_t d = 768;
    size_t n = 1000000;
    auto kmeans = skmeans::SuperKMeans(k, d);
    
    // Run the clustering
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    
    // Assign points
    std::vector<uint32_t> assignments = kmeans.Assign(data.data(), centroids.data(), n, k);
}

```

Check our [examples](./examples/) for a fully working C++ example.

</details>

## Installation
We provide Python bindings for ease of use. 

### Prerequisites
- Clang 17, CMake 3.26
- OpenMP
- A BLAS implementation
- Python 3 (only for Python bindings)

```bash
git clone https://github.com/lkuffo/SuperKMeans.git
cd SuperKMeans
git submodule update --init
pip install .

# Run plug-and-play example
python ./examples/simple_clustering.py

# Set a value for n, d and k
python ./examples/simple_clustering.py 100000 1536 1000
```

<details>

<summary>Compilation in C++</summary>

```bash
git clone https://github.com/lkuffo/SuperKMeans.git
cd SuperKMeans
git submodule update --init

# Set proper path to clang if needed
export CXX="/usr/bin/clang++-18" 

# Compile
cmake .
make examples

# Run plug-and-play example
cd examples
./simple_clustering.out

# Set a value for n, d and k
./simple_clustering.out 100000 1536 1000
```
</details>

For a more comprehensive installation and compilation guide, check [INSTALL.md](./INSTALL.md).

## Getting the Best Performance
Check [INSTALL.md](./INSTALL.md).

## Roadmap
We are actively developing Super K-Means and accepting contributions! Check [CONTRIBUTING.md](./CONTRIBUTING.md)

## Benchmarking
To run our benchmark suite in C++, refer to [BENCHMARKING.md](./BENCHMARKING.md).