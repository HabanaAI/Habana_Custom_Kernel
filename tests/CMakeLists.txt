###################################################################################
# Copyright (c) 2018 Habana Labs.
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
include(ExternalProject)

#required for 'ExternalProject_Add' command
set(TEST_TARGET tpc_kernel_tests)

include_directories(../src/gaudi_src                          # include headers in src
                    ../src/gaudi2_src
                    ../src
                    ./gaudi_test
                    ./gaudi2_test
                    /usr/lib/habanatools/include
                    .
                    )

# Select files to compile
file(GLOB TESTS_FILES *.cpp gaudi_test/*.cpp gaudi2_test/*.cpp)

# Create test executable
add_executable(${TEST_TARGET} ${TESTS_FILES})

# link with test
target_link_libraries(
    ${TEST_TARGET}
    rt
    pthread
    ${TPC_TEST_CORE_LIBRARY} 
    ${TPCSIM_SHARED_LIBRARY}
    ${TPC_ELF_READER_LIBRARY}
    custom_tpc_perf_lib)
