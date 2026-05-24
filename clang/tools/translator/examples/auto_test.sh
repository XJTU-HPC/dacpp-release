#!/usr/bin/env bash

usage() {
    cat <<'EOF'
Usage: auto_test.sh [--small|--large] [--cpu|--gpu]

Run the built-in DACPP example regression tests.

Options:
  --small   Test the default small example inputs
  --large   Test the large example inputs
  --cpu     Compile generated SYCL code for CPU targets (default)
  --gpu     Compile generated SYCL code for NVIDIA GPU targets

Environment:
  ONEAPI_ROOT  Optional path to oneAPI. Used when icpx is not already on PATH.
  CUDA_HOME    Required for --gpu runs.
EOF
}

SCALE="small"
TARGET="cpu"

for arg in "$@"; do
    case "$arg" in
        --small) SCALE="small" ;;
        --large) SCALE="large" ;;
        --cpu) TARGET="cpu" ;;
        --gpu) TARGET="gpu" ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
TEST_DIR=$(dirname "$SCRIPT_PATH")
TRANSLATOR_DIR=$(realpath "$TEST_DIR/..")
BUILD_DIR=$(realpath "$TRANSLATOR_DIR/../../../build")
TRANSLATOR_BIN="$BUILD_DIR/bin/translator/translator"

if ! command -v icpx >/dev/null 2>&1; then
    if [[ -n "${ONEAPI_ROOT:-}" && -f "$ONEAPI_ROOT/setvars.sh" ]]; then
        # shellcheck disable=SC1090
        source "$ONEAPI_ROOT/setvars.sh" >/dev/null 2>&1
    else
        echo "icpx not found. Please source oneAPI first or set ONEAPI_ROOT." >&2
        exit 1
    fi
fi

INCLUDE_DIRS=(
    "$TRANSLATOR_DIR/dpcppLib/include/"
    "$TRANSLATOR_DIR/dacppLib/include/"
    "$TRANSLATOR_DIR/rewriter/include/"
    "$TRANSLATOR_DIR/std_lib/include/"
)

ICPX=$(command -v icpx)
if [[ -z "$ICPX" ]]; then
    echo "icpx not found after environment setup." >&2
    exit 1
fi

if [[ "$TARGET" == "gpu" && -z "${CUDA_HOME:-}" ]]; then
    echo "CUDA_HOME is not set. Please set CUDA_HOME before running --gpu tests." >&2
    exit 1
fi

dacpp() {
    if [[ ! -x "$TRANSLATOR_BIN" ]]; then
        echo "Translator executable not found: $TRANSLATOR_BIN" >&2
        return 1
    fi

    "$TRANSLATOR_BIN" "$@" \
        -extra-arg=-std=c++17 \
        -extra-arg=-Wno-unused-value \
        -- \
        -I"$TRANSLATOR_DIR/std_lib/include" \
        -I"$TRANSLATOR_DIR/dacppLib/include"
}

icpx-cpu() {
    "$ICPX" -fsycl \
        "$@" \
        "${INCLUDE_DIRS[@]/#/-I}"
}

icpx-gpu() {
    "$ICPX" -fsycl \
        -fsycl-targets=nvptx64-nvidia-cuda \
        --cuda-path="$CUDA_HOME" \
        "$@" \
        "${INCLUDE_DIRS[@]/#/-I}"
}

compile_sycl() {
    if [[ "$TARGET" == "gpu" ]]; then
        icpx-gpu "$@"
    else
        icpx-cpu "$@"
    fi
}

prepare_example_dir() {
    local example_name="$1"

    mkdir -p "$TMP_DIR/$example_name"
    cp -a "$TEST_DIR/$example_name/." "$TMP_DIR/$example_name/"
    find "$TMP_DIR/$example_name" -maxdepth 1 -type f -name "*_sycl_*.cpp" -delete
}

TMP_DIR="$TEST_DIR/tmp"
rm -rf $TMP_DIR
mkdir $TMP_DIR

examples=(
    "matMul1.0"
    "waveEquation1.0"
    "stencil1.0"
    "jacobi1.0"
    "FOuLa1.0"
    "decay1.0"
    "DFT1.0"
    "liuliang1.0"
    "MDP1.0"
    "mandel1.0"
    "gradientSum"
    "matMul_file"
    "matMul3D"
    "stencil3D"
)


echo "------------------------------------------------------------------------------------------"
echo "DACPP to SYCL transpilation test"
echo
echo "Mode: buffer"
echo "Scale: $SCALE"
echo "Device target: $TARGET"
echo

for dir in ${examples[@]}; do
    if [ "$SCALE" != "large" ]; then
        dacpp_file=$(find "$TEST_DIR/$dir/" -type f -name "*.dac.cpp" | head -n 1)
    else
        dacpp_file=$(find "$TEST_DIR/$dir/" -type f -name "*.large_dac.cpp" | head -n 1)
    fi
    if [ -z "$dacpp_file" ]; then
        echo "Example $dir: DACPP source file not found"
        continue
    fi
    prepare_example_dir "$dir"
    if [ "$SCALE" != "large" ]; then
        new_dacpp_file=$(find "$TMP_DIR/$dir/" -type f -name "*.dac.cpp" | head -n 1)
    else
        new_dacpp_file=$(find "$TMP_DIR/$dir/" -type f -name "*.large_dac.cpp" | head -n 1)
    fi
    if ! dacpp "$new_dacpp_file" >"$TMP_DIR/$dir/translate.log" 2>&1; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
        echo "  See $TMP_DIR/$dir/translate.log"
        continue
    fi
    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*_sycl_buffer.cpp")
    if [ -z "$sycl_file" ]; then
        echo "Example $dir: DACPP to SYCL transpilation failed"
        echo "  See $TMP_DIR/$dir/translate.log"
    else
        echo "Example $dir: DACPP to SYCL transpilation succeeded"
    fi
done

echo "------------------------------------------------------------------------------------------"
echo "Compile standard sycl files"
echo

for dir in ${examples[@]}; do
    if [ "$SCALE" != "large" ]; then
        std_sycl_file=$(find "$TMP_DIR/$dir/" -type f -name "*.StandardSycl.cpp" | head -n 1)
    else
        std_sycl_file=$(find "$TMP_DIR/$dir/" -type f -name "*.large_StandardSycl.cpp" | head -n 1)
    fi
    if [ -z "$std_sycl_file" ]; then
        if [ "$SCALE" != "large" ]; then
            std_file=$(find "$TMP_DIR/$dir/" -type f -name "*.serial.cpp" | head -n 1)
        else
            std_file=$(find "$TMP_DIR/$dir/" -type f -name "*.large_serial.cpp" | head -n 1)
        fi
        if [ "$std_file" ]; then
            echo "Example $dir: Standard SYCL file does not exist but serial C++ file exists"
            if ! g++ "$std_file" -o "$TMP_DIR/$dir/std_$dir" >"$TMP_DIR/$dir/std_compile.log" 2>&1; then
                echo "Example $dir: Serial C++ file compilation failed"
                echo "  See $TMP_DIR/$dir/std_compile.log"
                continue
            fi
            exe_file=$(find "$TMP_DIR/$dir/" -type f -name "std_$dir")
            if [ -z "$exe_file" ]; then
                echo "Example $dir: Serial C++ file compilation failed"
                echo "  See $TMP_DIR/$dir/std_compile.log"
            else
                exe_name=$(basename "$exe_file")
                if ! (cd "$TMP_DIR/$dir" && "./$exe_name" > "${exe_name}.std.out" 2>std_run.log); then
                    echo "Example $dir: Serial baseline execution failed"
                    echo "  See $TMP_DIR/$dir/std_run.log"
                fi
            fi
        else
            echo "Example $dir: No standard SYCL file or serial C++ file found"
        fi
    else
        if ! compile_sycl "$std_sycl_file" -o "$TMP_DIR/$dir/std_$dir" >"$TMP_DIR/$dir/std_compile.log" 2>&1; then
            echo "Example $dir: Standard SYCL file compilation failed"
            echo "  See $TMP_DIR/$dir/std_compile.log"
            continue
        fi
        exe_file=$(find "$TMP_DIR/$dir/" -type f -name "std_$dir")
        if [ -z "$exe_file" ]; then
            echo "Example $dir: Standard SYCL file compilation failed"
            echo "  See $TMP_DIR/$dir/std_compile.log"
        else
            exe_name=$(basename "$exe_file")
            if ! (cd "$TMP_DIR/$dir" && "./$exe_name" > "${exe_name}.std.out" 2>std_run.log); then
                echo "Example $dir: Standard SYCL execution failed"
                echo "  See $TMP_DIR/$dir/std_run.log"
            fi
        fi
    fi 
done

echo "------------------------------------------------------------------------------------------"
echo "Generated SYCL files compilation test"
echo

for dir in ${examples[@]}; do
    sycl_file=$(find "$TMP_DIR/$dir" -type f -name "*_sycl_buffer.cpp")
    if [ -z "$sycl_file" ]; then
        continue
    fi
    if ! compile_sycl -fp-model strict -fno-fast-math "$sycl_file" -o "$TMP_DIR/$dir/$dir" >"$TMP_DIR/$dir/generated_compile.log" 2>&1; then
        echo "Example $dir: SYCL compilation failed"
        echo "  See $TMP_DIR/$dir/generated_compile.log"
        continue
    fi
    exe_file=$(find "$TMP_DIR/$dir/" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        echo "Example $dir: SYCL compilation failed"
        echo "  See $TMP_DIR/$dir/generated_compile.log"
    else
        echo "Example $dir: SYCL compilation succeeded"
    fi
done
echo "------------------------------------------------------------------------------------------"
echo "SYCL files ouput result test"
echo

for dir in ${examples[@]}; do
    exe_file=$(find "$TMP_DIR/$dir/" -type f -name "$dir")
    if [ -z "$exe_file" ]; then
        continue
    fi
    exe_name=$(basename "$exe_file")
    if ! (cd "$TMP_DIR/$dir" && "./$exe_name" > "${exe_name}.out" 2>generated_run.log); then
        echo "Example $dir: generated SYCL execution failed"
        echo "  See $TMP_DIR/$dir/generated_run.log"
        continue
    fi

    std_res=$(find "$TMP_DIR/$dir/" -type f -name "*.std.out" | head -n 1)
    if [ -z "$std_res" ]; then
        echo "Example $dir: no standard result found for diff"
    elif diff -y --suppress-common-lines "$TMP_DIR/$dir/${exe_name}.out" "$std_res"; then
        echo "Example $dir: execution test succeeded"
    else
        echo
        echo "Example $dir: execution test failed with some different lines listed above"
    fi
done


echo "------------------------------------------------------------------------------------------"
