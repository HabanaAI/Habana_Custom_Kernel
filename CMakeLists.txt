###################################################################################
# Copyright (c) 2021 Habana Labs.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# *   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# *   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################
cmake_minimum_required(VERSION 3.0.2)
project(tpc_kernels_db)

set(TPC_COMPILER_PATH "/usr/bin/tpc-clang" CACHE STRING "Path to TPC C/C++ compiler")
set(TPC_DISASSEMBLER_PATH "/usr/bin/tpc-llvm-objdump" CACHE STRING "Path to TPC disassembler")
set(TPCSIM_SHARED_LIBRARY "/usr/lib/habanatools/libtpcsim_shared.so" CACHE STRING "Path to TPC simulator shared object")
set(TPC_TEST_CORE_LIBRARY "/usr/lib/habanatools/libtpc_tests_core_ext.so" CACHE STRING "Path to TPC test core shared object")
set(TPC_ELF_READER_LIBRARY "/usr/lib/habanatools/libTpcElfReader.so" CACHE STRING "Path to Elf Reader shared object")
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "Path to TPC includes")
set(TPC_CURRENT_LIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/include" CACHE STRING "Path to current includes")
set(TPC_TESTS_CORE_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE STRING "Path to tests_core includes")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror -fno-strict-aliasing")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-ggdb -O2")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build type selected for Debug")
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Add external specs path.
#include_directories(${TPC_INCLUDE_DIR})
include_directories(${TPC_TESTS_CORE_INCLUDE_DIR} ${TPC_INCLUDE_DIR})


# Add TPC-C source files to be compiled by TPC Goya compiler and
# linked into tpc_kernels executable.
file(GLOB TPC_C_GOYA_SOURCES_TO_EMBED ./kernels/goya/*.c)

# Add TPC-C source files to be compiled by TPC Gaudi compiler and
# linked into tpc_kernels executable.
file(GLOB TPC_C_GAUDI_SOURCES_TO_EMBED ./kernels/gaudi/*.c)

# Add TPC assembly files to be compiled by TPC Goya assembler and
# linked into tpc_kernels executable.
file(GLOB TPC_ASSEMBLY_TO_EMBED  ./kernels/*.tpcasm)

# Find 'tpcsim' project.
string(CONCAT TPCSIM_BUILD_DIR "TPCSIM_" ${CMAKE_BUILD_TYPE} "_BUILD" )
string(TOUPPER ${TPCSIM_BUILD_DIR} TPCSIM_BUILD_DIR)

# Find Synapse builds
string(CONCAT SYNAPSE_PATH "SYNAPSE_" ${CMAKE_BUILD_TYPE} "_BUILD" )
string(TOUPPER ${SYNAPSE_PATH} SYNAPSE_PATH)


add_subdirectory(src)  # build production
add_subdirectory(tests) # unit test for kernels

