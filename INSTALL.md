# Installation

Looking for installation on GPU? We have an implementation (see `gpu_optimized` branch)! But the installation instructions are still WiP

### Super K-Means on CPU needs:
- C++17, CMake 3.26
- OpenMP
- A BLAS implementation

> [!WARNING]
> A proper BLAS implementation is **EXTREMELY** important for performance. The pre-installed BLAS in your Linux distribution or installing via `apt` is **SLOW**.

* [Installing OpenMP](#installing-openmp)
* [Installing BLAS](#installing-blas)
* [Troubleshooting](#troubleshooting)

Once you have these requirements, you can install Python Bindings or compile our C++ example code.

### Python Bindings
```sh
git clone https://github.com/lkuffo/SuperKMeans.git
git submodule update --init
pip install . 
```

### C++ Example
```sh
git clone https://github.com/lkuffo/SuperKMeans.git
git submodule update --init

# Compile
cmake . 
make simple_example.out

# Run
cd examples
./simple_example
```


## Installing OpenMP

### Linux
Most distributions come with OpenMP, or you can install it with:
```sh
sudo apt-get install libomp-dev
```

### MacOS
```sh 
brew install libomp
```


## Installing BLAS

BLAS is extremely important to achieve high performance. We recommend [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). 

### Linux
Most distributions come with [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), or you may have already installed OpenBLAS via `apt`. **THIS IS SLOW**. We recommend installing OpenBLAS from source.

```sh
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make DYNAMIC_ARCH=1 USE_OPENMP=1 NUM_THREADS=128
make PREFIX=/usr/local install
ldconfig
```

### MacOS
**Silicon Chips (M1 to M5)**: You don't need to do anything special. We automatically detect [Apple Accelerate](https://developer.apple.com/documentation/accelerate) that uses the [AMX](https://github.com/corsix/amx) unit. 

**Intel Chips (older Macs)**: Install OpenBLAS as detailed above.


## Troubleshooting

### SuperKMeans is slow on my Apple Silicon
If you previously installed OpenBLAS in your machine, the installation may be linking to OpenBLAS instead of Apple Accelerate. You can try forcing the linking of Apple Accelerate: 

```sh
# C++
cmake . -DBLAS_LIBRARIES=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework

# Python Installation
pip install . -C cmake.args="-DBLAS_LIBRARIES=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework"
```

### SuperKMeans is slow even after installing OpenBlas from source in Linux
Try forcing the linking to the OpenBlas you just installed. For example:
```sh
# C++
cmake . -DBLAS_LIBRARIES=/usr/local/lib/libopenblas_neoversev2p-r0.3.31.dev.so

# Python Installation
pip install . -C cmake.args="-DBLAS_LIBRARIES=/usr/local/lib/libopenblas_neoversev2p-r0.3.31.dev.so"
```

Note that the name of the `.so` library changes if you target a specific CPU or use `DYNAMIC_ARCH=1`. You might want to verify its name by looking into the `/usr/local/lib` directory.

### Can I use IntelMKL or AMD AOCL BLIS instead of OpenBLAS?
We detect IntelMKL automatically. For AMD AOCL BLIS, you can force the linking by doing:

```sh
# C++
cmake . -DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis-mt.so

# Python Installation
pip install . -C cmake.args="-DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis-mt.so"
```

### OpenBLAS binaries are too large. What can I do?
You can install OpenBLAS for a specific CPU instead of using `DYNAMIC_ARCH=1`. This will reduce the binary sizes. You can check the available CPU targets [here](https://github.com/OpenMathLib/OpenBLAS/blob/develop/TargetList.txt).

These are some commands for common CPUs:
```sh
# Intel Sapphire Rapids, Emerald Rapids, Granite Rapids
# Code-named in AWS as r8i, r7i
make TARGET=SAPPHIRERAPIDS DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# Intel Skylake, Icelake
# Code-named in AWS as r6i
make TARGET=SKYLAKEX DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AMD Zen 5, Zen 4, Zen 3
# Code-named in AWS as r8a, r7a, r6a
make TARGET=ZEN DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AWS Graviton 4
# Code-named in AWS as r8g
make TARGET=NEOVERSEV2 DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128

# AWS Graviton 3
# Code-named in AWS as r7g
make TARGET=NEOVERSEV1 DYNAMIC_ARCH=0 USE_OPENMP=1 NUM_THREADS=128
```

### Does Super K-Means use SIMD?
Yes. We have optimizations for AVX512, AVX2, and NEON. You don't need to do anything special to activate these. If your machine doesn't have any of these, we rely on scalar code. 

