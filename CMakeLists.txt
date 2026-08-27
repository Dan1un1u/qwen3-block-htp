cmake_minimum_required(VERSION 3.14.3)

project(qwen3_probe C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0115 OLD)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/rtos/qurt
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    add_library(qwen3_probe SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/qwen3_probe_stub
    )
    build_idl(include/qwen3_probe.idl qwen3_probe)

    add_executable(qwen3_probe_cli
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/main
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/session
    )

    set_common_compile_and_link_options(qwen3_probe_cli)
    target_compile_options(qwen3_probe_cli PRIVATE -Wall -Wextra -Werror)

    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(qwen3_probe ${dsprpc})

    add_dependencies(qwen3_probe_cli qwen3_probe)
    target_link_libraries(qwen3_probe_cli qwen3_probe)

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(qwen3_probe_cli PRIVATE -llog -ldl)
    endif()

    copy_binaries(qwen3_probe_cli qwen3_probe)
else()
    add_library(qwen3_probe_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/qwen3_probe_skel
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/probe_imp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_int8_tile
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_u8s8_projection
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mlp_u8
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/w4_u8_expand
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/w4_parallel_pipeline
    )
    build_idl(include/qwen3_probe.idl qwen3_probe_skel)
    target_compile_options(qwen3_probe_skel PRIVATE
        -Wall -Wextra -Werror -mhmx -mhvx -mhvx-length=128B
    )
    copy_binaries(qwen3_probe_skel)
endif()
