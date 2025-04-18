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
set(TARGET custom_tpc_perf_lib)

# Decide which files to compile
file(GLOB SRC *.cpp gaudi_src/*.cpp gaudi2_src/*.cpp gaudi3_src/*.cpp)

# define .so object for compiling.
add_library(${TARGET} SHARED ${SRC})

include_directories(.
                    gaudi_src
                    gaudi2_src
                    gaudi3_src
                    )

# Generate a list of all TPC headers for compilation dependencies.
file(GLOB TPC_GAUDI_HEADERS ../kernels/gaudi/*.h)
file(GLOB TPC_GAUDI2_HEADERS ../kernels/gaudi2/*.h)
file(GLOB TPC_GAUDI3_HEADERS ../kernels/gaudi3/*.h)

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release ... FORCE )
ENDIF()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set (COND_EXEC "echo")
    set (OPT_FLAG "-O0")
    set (DEBUG_FLAG "-g")
    message("Debug mode is on")
else()
    set (OPT_FLAG "-O2")
    set (DEBUG_FLAG "")
    message("Debug mode is off")
endif()
separate_arguments(OPT_FLAG)

# This routine does the following:
# 1. Copies the TPC-C source file to 'bin' for reference.
# 2. Compiles sources using TPC-C compiler (two times - one is for testing of compiler regression).
# 3. Wraps compiler output in X86 elf.
# 4. Links the X86 elf into the executable
# 5. Disassemble the file for debug purposes
function(CompileTPCCWithClang TPC_C_SOURCES_TO_EMBED TPC_HEADERS deviceName)
    foreach(SOURCE_FILE  ${TPC_C_SOURCES_TO_EMBED})
        get_filename_component(SOURCE_NAME_NO_PATH_NO_SUFFIX ${SOURCE_FILE} NAME_WE)
        get_filename_component(SOURCE_FILE_NAME_NO_PATH ${SOURCE_FILE} NAME)
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND "${SOURCE_FILE_NAME_NO_PATH}" STREQUAL "filter_2d_i8_w33_s11.c")
            set (OPT_FLAG "-O1")
        elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set (OPT_FLAG "-O0")
        endif()
        add_custom_command(OUTPUT "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
            DEPENDS ${SOURCE_FILE} ${TPC_COMPILER_PATH} ${TPC_HEADERS}
                COMMAND cp ${SOURCE_FILE}  ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND ${TPC_COMPILER_PATH} -Wall -Werror -march=${deviceName} ${OPT_FLAG} ${DEBUG_FLAG} -I${TPC_INCLUDE_DIR}/ -I${TPC_CURRENT_LIST_DIR} ./${SOURCE_FILE_NAME_NO_PATH} -c
                COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
				COMMAND  ${TPC_DISASSEMBLER_PATH} --triple tpc -d -j .text --no-show-raw-insn --no-leading-addr --mcpu=${deviceName} ./${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o > ${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.tpcasm)
        add_library(${SOURCE_NAME_NO_PATH_NO_SUFFIX} STATIC "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o")
        set_target_properties( ${SOURCE_NAME_NO_PATH_NO_SUFFIX} PROPERTIES LINKER_LANGUAGE C  )
        target_link_libraries(${TARGET} ${SOURCE_NAME_NO_PATH_NO_SUFFIX})
    endforeach()
endfunction(CompileTPCCWithClang)

CompileTPCCWithClang("${TPC_C_GAUDI_SOURCES_TO_EMBED}" "${TPC_GAUDI_HEADERS}" gaudi)
CompileTPCCWithClang("${TPC_C_GAUDI2_SOURCES_TO_EMBED}" "${TPC_GAUDI2_HEADERS}" gaudi2)
CompileTPCCWithClang("${TPC_C_GAUDI3_SOURCES_TO_EMBED}" "${TPC_GAUDI3_HEADERS}" gaudi3)
