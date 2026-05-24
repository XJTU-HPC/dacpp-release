#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: dacpp_build.sh <input.dac_sycl_buffer.cpp> [-o output] [--cpu|--gpu] [-- <extra icpx args>]

Build a translated DACPP SYCL source file.

Examples:
  dacpp_build.sh /path/to/my_case.dac_sycl_buffer.cpp
  dacpp_build.sh /path/to/my_case.dac_sycl_buffer.cpp --gpu
  dacpp_build.sh /path/to/my_case.dac_sycl_buffer.cpp -o /tmp/my_case

Environment overrides:
  ICPX        Path to the icpx compiler
  CUDA_HOME   CUDA toolkit path used for --gpu builds
  ONEAPI_ROOT oneAPI installation root used when icpx is not already on PATH
EOF
}

INPUT_FILE=""
OUTPUT_FILE=""
TARGET="cpu"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -o)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Missing value for -o." >&2
                exit 1
            fi
            OUTPUT_FILE="$1"
            shift
            ;;
        --cpu)
            TARGET="cpu"
            shift
            ;;
        --gpu)
            TARGET="gpu"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$INPUT_FILE" ]]; then
                echo "Only one input SYCL source file is supported." >&2
                usage >&2
                exit 1
            fi
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    usage >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input SYCL source file not found: $INPUT_FILE" >&2
    exit 1
fi

if ! command -v icpx >/dev/null 2>&1; then
    if [[ -n "${ONEAPI_ROOT:-}" && -f "$ONEAPI_ROOT/setvars.sh" ]]; then
        # shellcheck disable=SC1090
        source "$ONEAPI_ROOT/setvars.sh" >/dev/null 2>&1
    fi
fi

ICPX_BIN=${ICPX:-$(command -v icpx || true)}
if [[ -z "$ICPX_BIN" ]]; then
    echo "icpx compiler not found. Set ONEAPI_ROOT or ICPX first." >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRANSLATOR_ROOT=$(realpath "$SCRIPT_DIR/..")
INPUT_FILE=$(realpath "$INPUT_FILE")

if [[ -z "$OUTPUT_FILE" ]]; then
    INPUT_DIR=$(dirname "$INPUT_FILE")
    INPUT_BASE=$(basename "$INPUT_FILE")
    if [[ "$INPUT_BASE" == *_sycl_buffer.cpp ]]; then
        INPUT_BASE="${INPUT_BASE%_sycl_buffer.cpp}"
    else
        INPUT_BASE="${INPUT_BASE%.cpp}"
    fi
    OUTPUT_FILE="$INPUT_DIR/$INPUT_BASE"
fi

CMD=(
    "$ICPX_BIN"
    -std=c++17
    -fsycl
    "$INPUT_FILE"
    -o "$OUTPUT_FILE"
    -I"$TRANSLATOR_ROOT/dpcppLib/include"
    -I"$TRANSLATOR_ROOT/dacppLib/include"
    -I"$TRANSLATOR_ROOT/rewriter/include"
    -I"$TRANSLATOR_ROOT/std_lib/include"
)

if [[ "$TARGET" == "gpu" ]]; then
    CUDA_PATH="${CUDA_HOME:-}"
    if [[ -z "$CUDA_PATH" ]]; then
        echo "CUDA_HOME is not set. Set CUDA_HOME or use --cpu." >&2
        exit 1
    fi

    CMD+=(
        -fsycl-targets=nvptx64-nvidia-cuda
        "--cuda-path=$CUDA_PATH"
    )
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"

echo "Generated executable: $OUTPUT_FILE"
