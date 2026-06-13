#!/bin/bash
set -e

source ./env_vars.sh

if [ ! -x "$PYTHON_INTERPRETER_PATH" ]; then
    echo "Error: PYTHON_INTERPRETER_PATH is not executable: $PYTHON_INTERPRETER_PATH"
    exit 1
fi

if ! command -v ninja &> /dev/null; then
    echo "Error: Ninja is not installed. Please install Ninja before proceeding."
    exit 1
fi

CLEAN_BUILD=true

if $CLEAN_BUILD; then
    echo "Cleaning the build directory..."
    rm -rf build
fi

echo "Creating the build directory..."
mkdir build && cd build || { echo "Error changing to the build directory"; exit 1; }

echo "Configuring the project with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON_INTERPRETER_PATH" -G Ninja .. || { echo "Error running CMake"; exit 1; }

echo "Building the project..."
ninja -j$(nproc) || { echo "Error building the project"; exit 1; }

echo "Build completed successfully!"
