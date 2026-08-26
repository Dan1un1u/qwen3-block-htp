#!/usr/bin/env bash

# Isolated toolchain environment for EXP-0001. Source this file from scripts;
# do not install or replace system-wide SDK components.

export EXP0001_SDK_ROOT="${EXP0001_SDK_ROOT:-/home/daniuniu/toolchains/hexagon-sdk-6.6.0.0-exp0001/Hexagon_SDK/6.6.0.0}"
export EXP0001_NDK_ROOT="${EXP0001_NDK_ROOT:-/home/daniuniu/toolchains/android-ndk-r26c}"
export EXP0001_CMAKE_ROOT="${EXP0001_CMAKE_ROOT:-/home/daniuniu/toolchains/cmake-3.28.6-exp0001}"

export PATH="/home/daniuniu/toolchains/exp0001-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export ANDROID_ROOT_DIR="${EXP0001_NDK_ROOT}"

# SDK setup expects HEXAGON_SDK_ROOT to be unset on entry.
unset HEXAGON_SDK_ROOT
source "${EXP0001_SDK_ROOT}/setup_sdk_env.source"
export CMAKE_ROOT_PATH="${EXP0001_CMAKE_ROOT}"
