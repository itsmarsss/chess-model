cmake_minimum_required(VERSION 3.14)
project(chess_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Detect architecture for SIMD
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_definitions(-DUSE_NEON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mavx2 HAS_AVX2)
    if(HAS_AVX2)
        add_definitions(-DUSE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    else()
        add_definitions(-DUSE_SSE41)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
    endif()
endif()

add_executable(chess_engine
    src/main.cpp
    src/bitboard.cpp
    src/position.cpp
    src/movegen.cpp
    src/search.cpp
    src/eval.cpp
    src/tt.cpp
    src/movepick.cpp
    src/uci.cpp
)

target_include_directories(chess_engine PRIVATE src)
