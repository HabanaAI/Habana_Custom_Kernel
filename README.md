# Habana Custom Kernel
This repository provides the examples to write and build Habana custom kernels using the HabanaTools.

## Table Of Contents
* [TPC Kernels Overview](#tpc-kernels-overview)
* [Install Habanatools For Ubuntu](#install-habanatools-for-ubuntu)
* [Template Examples](#template-examples)
* [Tensorflow Custom Ops](#tensorflow-custom-ops)

## TPC Kernels Overview
The Tensor Processor Core™ (**TPC**) is a fully programmable VLIW4 processor designed to execute non-linear deep learning operators. It is embedded in Habana’s Gaudi deep learning accelerator. Habana’s Gaudi SoC contains numerous TPC cores all operating in parallel, with each core running a single thread. The TPC is designed with very long instruction word (VLIW) architecture. It has a wide single instruction multiple data (SIMD) vector unit that support 2048-bit SIMD operations with data types such as float, bfloat16, INT16, INT32 and INT8. In each cycle, the TPC’s ALU (Arithmetic Logic Unit) can execute up to 64 floats/INT32 ops, or 128 INT16 ops, or 256 INT8 ops.
TPC is designed for workloads that do not map to MME (Matrix Multiplication Engine). Those workloads or operators can be implemented using TPC kernels. 

## Install Habanatools For Ubuntu
To retrieve the package please visit [Habana Vault](https://vault.habana.ai/ui/repos/tree/General/debian%2Fbionic%2Fpool%2Fmain%2Fh%2Fhabanatools) and download the latest release package for Ubuntu 18.04. You can find different packages for different OS you used. 
```  
  sudo dpkg -i ./habanatools_0.14.0-420_amd64.deb 
```
- Once installed the following files will be added to your machine 
  
  |  |Location | Purpose  |
  |--|--------------------|-----------------------------|
  |1 | /usr/bin/tpc-clang | TPC-C compiler and assembler |
  |2 | /usr/bin/tpc-llvm-objdump | TPC dis-assembler|
  |3 | /usr/lib/habanatools/libtpcsim_shared.so | TPC simulator|
  |4 | /usr/lib/habanatools/libtpc_tests_core.so | Test core library |  
  |5 | /usr/lib/habanatools/include/TPC.h |Simulator headers |
  |6 | /usr/lib/habanatools/include/gc_interface.h | Glue code interface header |
  |7 | /usr/lib/habanatools/include/tpc-intrinsics.h | Available TPC-C intrinsics |
  |8 | /usr/lib/habanatools/include/tpc_test_core_api.h | Test core API |
      
- Compiler usage example
The compiler supports a single translation unit, hence ‘-c’ argument should be defined.
```  
 /usr/bin/tpc-clang batch_norm_fwd_f32.c -c -x c++ -o batch_norm_fwd_f32.o
```  
The output of the compilation session will be an elf file named ‘batch_norm_fwd_f32.o’ . To extract raw binary, from the elf, use the following command:
```  
 objcopy -O binary --only-section=.text batch_norm_fwd_f32.o batch_norm_fwd_f32.bin 
```  
Using CMAKE tool shown in the following template examples.
    
For other OS, please refer to the [TPC Tools Installation Guide](https://docs.habana.ai/en/latest/TPC_Tools_Installation/TPC_Tools_Installation_Guide.html) for more details.

## Template Examples
The template examples show users how to create and build the custom kernels, which can be used in Tensorflow (**TF**) custom ops later.
This template example has organized in the following way, which contains **TPC kennels**(kernels/), **Glue codes**(src/) and **Unit tests**(tests/).
* TPC kernel codes are the ISA executed by the TPC processor. They contain the kernel implementation.
* Glue codes are executed on the host machine serviced by the Habana DNN SoC, and they hold specifications regarding how the program input/outputs can be dynamically partitioned between the numerous TPC processors in the Habana device.
* Unit tests are to verify the kernel's correctness using the build-in simulator provided in the HabanaTools, test core will be added later for ability to test on real device.

### Build the template examples
Make sure your Habana tools are installed, check the /usr/bin/tpc-clang and Cmake are up-to-date version, you can download latest cmake via <https://cmake.org/download/>

Clone the repository
```  
 git clone git@github.com:HabanaAI/Habana_Custom_Kernel.git
``` 
In the terminal, make sure you are in the project root directory, then create a directory called build
```  
mkdir build
cd build
```  
then run the following commands
```  
cmake ..
make
```  
After build, you can find libcustom_tpc_perf_lib.so in build/src directory, which is your custom kernel library, and tpc_kernel_tests in build/tests, which contains all the unit tests.
For more details about TPC kernel writing, please refer to the [TPC User Guide](https://docs.habana.ai/en/latest/TPC_User_Guide/TPC_User_Guide.html) for more information.

## Tensorflow Custom Ops
The user also can develop their own TF custom ops using their own TPC kernels. In this custom kernel project, we provide several custom kernel examples, such as custom_div (division), relu6_fwd (relu6 forward path) and relu6_bwd (relu6 backward path). Please visit [TensorFlow Custom OPs Examples](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/examples/custom_op) for more details and make sure add your custom kernel path to environment variable GC_KERNEL_PATH, like export GC_KERNEL_PATH=/path/to/your_so/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so.
