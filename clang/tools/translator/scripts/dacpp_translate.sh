#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: dacpp_translate.sh <input.dac.cpp> [-- <extra clang args>]

Translate a DACPP source file into a buffer-mode SYCL source file.

Examples:
  dacpp_translate.sh /path/to/my_case.dac.cpp
  dacpp_translate.sh /path/to/my_case.dac.cpp -- -I/path/to/my/includes

Environment overrides:
  BUILD_DIR       Build directory containing bin/translator/translator
  TRANSLATOR_BIN  Path to the translator executable
EOF
}

INPUT_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
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
                echo "Only one input source file is supported." >&2
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
    echo "Input source file not found: $INPUT_FILE" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRANSLATOR_ROOT=$(realpath "$SCRIPT_DIR/..")
REPO_ROOT=$(realpath "$TRANSLATOR_ROOT/../../..")
BUILD_DIR=${BUILD_DIR:-"$REPO_ROOT/build"}
TRANSLATOR_BIN=${TRANSLATOR_BIN:-"$BUILD_DIR/bin/translator/translator"}
INPUT_FILE=$(realpath "$INPUT_FILE")

if [[ ! -x "$TRANSLATOR_BIN" ]]; then
    echo "Translator executable not found: $TRANSLATOR_BIN" >&2
    echo "Build the translator first, or set TRANSLATOR_BIN explicitly." >&2
    exit 1
fi

OUTPUT_FILE="$INPUT_FILE"
if [[ "$OUTPUT_FILE" == *.cpp ]]; then
    OUTPUT_FILE="${OUTPUT_FILE%.cpp}_sycl_buffer.cpp"
else
    OUTPUT_FILE="${OUTPUT_FILE}_sycl_buffer.cpp"
fi

"$TRANSLATOR_BIN" \
    "$INPUT_FILE" \
    -extra-arg=-std=c++17 \
    -extra-arg=-Wno-unused-value \
    -- \
    -I"$TRANSLATOR_ROOT/std_lib/include" \
    -I"$TRANSLATOR_ROOT/dacppLib/include" \
    "${EXTRA_ARGS[@]}"

echo "Generated SYCL source: $OUTPUT_FILE"
