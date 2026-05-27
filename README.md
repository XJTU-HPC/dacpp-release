# DACPP on LLVM/Clang

This repository contains the DACPP source-to-source translation framework built
on top of LLVM/Clang. The `multi-node` branch provides the multi-node release
with MPI + SYCL lowering in addition to the standard SYCL buffer workflow.

## Overview

DACPP extends Clang with a custom translator located at
`clang/tools/translator`. The translator parses DACPP constructs, performs
source analysis, and rewrites input programs into executable C++/SYCL code.
With `--mpi`, it emits MPI + SYCL code for distributed multi-node execution.

This repository keeps the LLVM monorepo layout aligned with the public `main`
branch so that the translator can be built as part of the Clang toolchain.

## Upstream Base

This release is based on the LLVM/Clang 19.1.4 source tree.

Main project-specific changes are located under `clang/tools/translator`.

## Repository Layout

- `clang/tools/translator`: DACPP translator implementation, runtime headers,
  helper scripts, examples, and tests.
- `clang/`, `llvm/`, `cmake/`, and other top-level project directories:
  upstream LLVM/Clang monorepo components kept in the public release layout.
- `README.llvm.md`: the original LLVM top-level README.

Generated build trees and machine-local artifacts are not part of the release.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler supported by LLVM
- Ninja or another CMake generator
- MPI implementation with `mpicxx` and `mpirun`
- AdaptiveCpp for compiling generated SYCL programs

Optional environment variables:

```bash
export ACPP_ROOT=/path/to/adaptivecpp
export CUDA_PATH=/path/to/cuda
export ONEAPI_SETVARS=/path/to/setvars.sh
```

## Build

Build the translator from the repository root:

```bash
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build translator
```

After a successful build, the translator executable is available at:

```text
build/bin/translator/translator
```

## Usage

Detailed translator usage, helper environment setup, and test layout are
documented in `clang/tools/translator/README.md`.

## Branches

The public release is organized by execution mode:

- `main`: single-node release
- `multi-node`: multi-node MPI + SYCL release

## License

This repository includes LLVM/Clang source code and follows the upstream
license terms in `LICENSE.TXT`. DACPP-specific source files are distributed
together with the modified LLVM/Clang code in this repository.
