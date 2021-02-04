# Habana Custom Kernel
This repository provides the examples to write and build Habana custom kernels using the Habana Tools.

## Table Of Contents
* [TPC Kernels Overview](#tpc-kernels-overview)
* [Install Habana Tool](#install-habana-tool)
* [Template Examples](#template-examples)

## TPC Kernels Overview
The Tensor Processor Core™ (**TPC**) is a fully programmable VLIW4 processor designed to execute non-linear deep learning operators. It is embedded in Habana’s Gaudi deep learning accelerator. Habana’s Gaudi SoC contains numerous TPC cores all operating in parallel, with each core running a single thread. The TPC is designed with very long instruction word (VLIW) architecture. It has a wide single instruction multiple data (SIMD) vector unit that support 2048-bit SIMD operations with data types such as float, bfloat16, INT16, INT32 and INT8. In each cycle, the TPC’s ALU can execute up to 64 floats/INT32 ops, or 128 INT16 ops, or 256 INT8 ops.
TPC is designed for workloads that do not map to MME (Matrix Multiplication Engine). Those workloads or operators can be implemented using TPC kernels. 

## Install Habana Tool
To retrieve the package please login to your account at <https://ftp.habana.ai>. Details are provided via email. You can use any browser.
- Go into the tpc_tools/<version>/ folder and download habanatools_0.13_.noarch.rpm file for CentOs or habanatools_0.13_amd64.deb for Debian. 
- For Ubuntu
```  
  sudo dpkg -i ./habanatools_<version>_amd64.deb 
```
- Once installed the following files will be added to your machine 
  
  | Location | Purpose | |
  |--|--------------------|-----------------------------|
  |1 | /usr/bin/tpc-clang | TPC-C compiler and assember |
  |2 | /usr/bin/tpc-llvm-objdump | TPC dis-assembler|
  |3 | /usr/lib/haba natools/libtpcsim_shared.so | TPC simulator|
  |4 | /usr/li b/habanatools/include/TPC.h |Simulator headers |
  |5 | /usr/lib/habanat ools/include/gc_interface.h | Glue code interface header |
  |6 | /usr/lib/habanatoo ls/include/tpc-intrinsics.h | Available TPC-C intrinsics |
    
- Compiler usage example
The compiler supports a single translation unit, hence ‘-c’ argument should be defined.
```  
 /usr/bin/tpc-clang reduction.c -c -x c++ -o reduction.o
```  
The output of the compilation session will be an elf file named ‘reduction.o’ . To extract raw binary, from the elf, use the following command:
```  
 objcopy -O binary --only-section=.text reduction.o reduction.bin 
```  
Using CMAKE tool shown in the following template example
    
For details, please refer to <https://habana-labs-tpc-gaudi.readthedocs-hosted.com/en/latest/TPC_Tools_Installation/TPC_Tools_Installation_Guide.html>
## Template Examples
The template examples show users how to create and build the custom kernels, which can be used in Tensorflow (**TF**) custom ops later.
This template example has organized in the following way, which contains **TPC kennels**(kernels/), **Glue codes**(src/) and **Unit tests**(tests/).
* TPC kernel code is the ISA executed by the TPC processor. It contains the kernel implementation.
* Glue code is executed on the host machine serviced by the Habana DNN SoC, and it holds specifications regarding how the program input/outputs can be dynamically partitioned between the numerous TPC processors in the Habana device.
* Unit tests is to verify the kernel correctness using the buildin simulator provided in the Habana Tools.

### Build the template examples
Make sure your Habana tools are installed, check the /usr/bin/tpc-clang and Cmake are up-to-date version, you can download latest cmake via <https://cmake.org/download/>

Clone the repository
```  
 git clone https://github.com/habana-labs-demo/Habana_Custom_Kernel.git
``` 
In the terminal or linux command line, make sure you are in the project root directory, then create a directory called build
```  
mkdir build
cd build
```  
then run the following commands
```  
cmake ..
make
```  
Now in build/src directory you can see the libmy_tpc_perf_lib.so, which is your custom kernel library. 
