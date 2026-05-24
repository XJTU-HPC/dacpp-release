# DACPP on LLVM/Clang

This repository contains the DACPP source-to-source translation framework built
on top of LLVM/Clang. The current branch is the single-node public release and
focuses on translating DACPP programs into SYCL-based C++ code.

## Overview

DACPP extends Clang with a custom translator located at
`clang/tools/translator`. The translator parses DACPP constructs, performs
source analysis, and rewrites input programs into executable SYCL-oriented C++
code together with the supporting runtime headers in the translator tree.

This repository keeps the LLVM monorepo layout so that the translator can be
built as part of the Clang toolchain.

## Upstream Base

This release is based on the LLVM/Clang 19.1.4 source tree.

Main project-specific changes are located under `clang/tools/translator`.

## Repository Layout

- `clang/tools/translator`: DACPP translator implementation, support libraries,
  helper scripts, and functional tests.
- `clang/`, `llvm/`, `cmake/`: upstream LLVM/Clang source tree required to
  build the translator.
- `build/`: local build output generated during compilation. This directory is
  not part of the public source release.

## Build

Build the translator from the repository root with CMake and Ninja:

```bash
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build
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
- `multi-node`: multi-node release

## Upstream Reference

The original LLVM top-level README has been preserved in `README.llvm.md` for
reference.

## License

This repository includes LLVM/Clang source code and follows the upstream
license terms in `LICENSE.TXT`. DACPP-specific source files are distributed
together with the modified LLVM/Clang code in this repository.
