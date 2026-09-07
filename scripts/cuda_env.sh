#!/usr/bin/env bash
# scripts/cuda_env.sh
# Dynamically configures LD_LIBRARY_PATH for pip-installed NVIDIA CUDA / cuDNN libraries.
# Usage:
#   source .venv/bin/activate
#   source scripts/cuda_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve active virtualenv
TARGET_VENV=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "${VIRTUAL_ENV}" ]; then
    TARGET_VENV="${VIRTUAL_ENV}"
elif [ -d "${PROJECT_ROOT}/.venv" ]; then
    TARGET_VENV="${PROJECT_ROOT}/.venv"
elif [ -d "${PWD}/.venv" ]; then
    TARGET_VENV="${PWD}/.venv"
fi

if [ -z "${TARGET_VENV}" ]; then
    echo "[cuda_env] WARNING: No virtual environment found. Please activate your virtualenv first (e.g. 'source .venv/bin/activate')."
    return 1 2>/dev/null || exit 1
fi

ADDED_PATHS=()

# Search for nvidia cublas and cudnn lib directories in site-packages
for lib_dir in $(find "${TARGET_VENV}" -type d \( -path "*/nvidia/cublas/lib" -o -path "*/nvidia/cudnn/lib" -o -path "*/nvidia/cudart/lib" \) 2>/dev/null); do
    if [ -d "${lib_dir}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${lib_dir}:"*) ;;
            *)
                export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
                ADDED_PATHS+=("${lib_dir}")
                ;;
        esac
    fi
done

if [ ${#ADDED_PATHS[@]} -gt 0 ]; then
    echo "[cuda_env] Added to LD_LIBRARY_PATH:"
    for p in "${ADDED_PATHS[@]}"; do
        echo "  - ${p}"
    done
else
    echo "[cuda_env] Note: No pip-installed nvidia/cublas or nvidia/cudnn lib dirs found in ${TARGET_VENV}."
    echo "[cuda_env] If you are on Linux/WSL with CUDA pip wheels installed, ensure nvidia-cublas-cu12 / nvidia-cudnn-cu12 are installed."
fi
