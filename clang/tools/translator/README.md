# DACPP Translator

`clang/tools/translator` contains the DACPP source-to-source translator and
runtime headers used by generated C++/SYCL and MPI + SYCL programs.

## What It Translates

DACPP programs describe data association with `shell` functions and local
computation with `calc` functions:

```cpp
VADD(a_tensor, b_tensor, out_tensor) <-> vadd;
```

The translator parses the DACPP expression with Clang AST matchers, lowers the
`shell`/`calc` pair, and rewrites the source into ordinary C++ code using SYCL
buffers. With `--mpi`, it emits MPI + SYCL wrappers for multi-node execution.

## Helper Environment

From the repository root:

```bash
source clang/tools/translator/env.sh
```

Useful helper functions:

- `dacpp`: invoke the translator with the required DACPP include paths.
- `acpp-compile`: compile generated SYCL C++ with AdaptiveCpp.
- `dacpp-translate-and-build`: translate with MPI and compile the result.

`ACPP_ROOT` should point to an AdaptiveCpp installation. If `ACPP_ROOT` is not
set, `env.sh` tries to infer it from an `acpp` executable already on `PATH`.

## Translation Commands

Standard SYCL buffer translation:

```bash
dacpp path/to/file.dac.cpp --mode=buffer
```

MPI + SYCL translation:

```bash
dacpp path/to/file.dac.cpp --mode=buffer --mpi
```

MPI output synchronization policy:

```bash
dacpp path/to/file.dac.cpp --mode=buffer --mpi --mpi-output-sync=all-ranks
dacpp path/to/file.dac.cpp --mode=buffer --mpi --mpi-output-sync=root-only
```

`all-ranks` is the default. It gathers root-visible outputs and broadcasts
updated values when later code needs non-root ranks to see them. `root-only`
skips the final broadcast and is only suitable when later code does not need
consistent non-root copies.

Generated files usually use one of these names:

```text
*.dac_sycl_buffer.cpp
*.large_dac_sycl_buffer.cpp
```

## Build Generated Code

```bash
acpp-compile path/to/file.dac_sycl_buffer.cpp /tmp/my_program
```

Run a local SYCL program:

```bash
/tmp/my_program
```

Run a multi-node MPI program with your site launcher:

```bash
mpirun -np 4 /tmp/my_program
```

On clusters, replace `mpirun` with the scheduler-provided launcher when needed
and export the runtime library paths required by your MPI and SYCL stack.

## Examples And Tests

The `examples/` directory is kept from the public single-node release layout.
The `tests/` directory contains the current translator test set, including
multi-node MPI structure and execution cases.

Local buffer tests:

```bash
cd clang/tools/translator
bash test_local.sh
```

MPI + SYCL tests:

```bash
cd clang/tools/translator
bash test_mpi.sh
```

Run selected cases:

```bash
bash test_mpi.sh vectorAddCombo stencil1.0 waveEquation1.0
```

Use larger inputs when a case provides `*.large_dac.cpp`:

```bash
bash test_mpi.sh --large stencil1.0 waveEquation1.0
```

Temporary directories default to `/tmp/dacpp_local_tmp` and
`/tmp/dacpp_mpi_tmp`. Override them with:

```bash
export DACPP_LOCAL_TEST_TMP_DIR=/path/to/local/tmp
export DACPP_MPI_TEST_TMP_DIR=/path/to/mpi/tmp
```

## Source Map

Core entry points:

- `translator.cpp`: CLI options, AST matchers, and rewrite dispatch.
- `parser/include/DacppStructure.h`: in-memory DACPP file model.
- `parser/lib/Shell.cpp`: shell split/binding analysis.
- `parser/lib/Calc.cpp`: calc body extraction and rewriting support.
- `rewriter/lib/Rewriter_Buffer_new.cpp`: standard SYCL buffer lowering.
- `rewriter/lib/Rewriter_MPI.cpp`: MPI lowering orchestration.
- `rewriter/lib/mpi/shared/MpiPlanBuilder.cpp`: MPI plan routing.

MPI runtime headers:

- `dpcppLib/include/MPIPlanner.h`: generated-code include facade.
- `dpcppLib/include/mpi/common`: shared MPI/SYCL view and type helpers.
- `dpcppLib/include/mpi/legacy_access_pattern`: fallback wrapper runtime.
- `dpcppLib/include/mpi/stencil`: stencil exchange runtime.
- `dpcppLib/include/mpi/operator_resident`: operator-resident runtime helpers.

The MPI lowering order is:

1. shell-derived/operator-resident analysis
2. stencil Phase-C lowering when needed
3. legacy AccessPattern wrapper fallback

Unsupported or unproven patterns are intentionally routed to conservative
fallbacks instead of silently using an unsafe optimized path.
