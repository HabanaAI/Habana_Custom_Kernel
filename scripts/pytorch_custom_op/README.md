# Create searchsorted custom op in PyTorch

This README provides an example of how to write custom PyTorch Ops using a TPC Kernel supported on an HPU device. For more details, refer to [PyTorch CustomOP API](https://docs.habana.ai/en/latest/PyTorch/PyTorch_CustomOp_API/page_index.html) documentation. 



## Table of Contents

* [Prerequisites](#Prerequisites) 
* [Content](#content)
* [Build and Run with Custom Kernels](#build-and-run-with-custom-kernels)
* [Important to Know](#important-to-know)
* [Applying CustomOps to a Real Training Model Example](#applying-customops-to-a-real-training-model-example)


## Prerequisites

- A TPC kernel on which the HpuKernel will run. To write a CustomOp, you must define the TPC kernel that HpuKernel will run on first. This document provides the required steps for using the custom TPC kernels `searchsorted_fwd_f32`, to implement CustomSearchsortedOp. For further information on how to write TPC kernels, refer to the [Habana Custom Kernel GitHub page](https://github.com/HabanaAI/Habana_Custom_Kernel).

- **habana-torch-plugin** Python package must be installed. Make sure to install by following the instructions detailed in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

## Content

- C++ file with **custom_op::custom_searchsorted**, definition and Kernel implementation on HPU:
    - `custom_searchsorted` performs searchsorted on sorted input.
- `setup.py` file for building the solution:
    - To compile to Op, run ```python setup.py build```.
- Python test to run and validate `CustomSearchSorted`:
    - ```python hpu_custom_op_searchsorted_test.py```

## Build and Run with Custom Kernels 

To build and run `custom_searchsorted`, run the following: 
```python setup.py build```

## Important to Know

In order to make the custom op work in the training process, usually we need to implement both forward and backward ops. But due to searchsorted op return an integer index, no backward op required at this time.

## Applying CustomSearchsorted to a Real Training Model Example

This section provides an example for applying CustomOps to a real training model NeuS. 
Follow the below steps:

1. Build the `custom_searchsorted` Op with the custom kernel `searchsorted_fwd_f32` as described above. 
2. If the build steps are successful, the run the unit test to make sure the custom_searchsorted op pass the test.
3. Make sure add the custom tc kernel to the GC_KERNEL_PATH, i.e., export `GC_KERNEL_PATH=/your/path/to/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH`.
4. Add the custom_searchsorted path to PYTHONPATH, i.e., `export PYTHONPATH = /your/path/to/pytorch_custom_op:$PYTHONPATH`.
4. Replace `inds = torch.searchsorted(cdf, u, right=True)` with the following to train the model.
    ```
    from custom_searchsorted import CustomSearchSorted

    cdf = cdf.to('hpu').detach()
    u = u.to('hpu').detach()
    cdf_h=cdf.transpose(0,1).unsqueeze(0)
    u_h=u.transpose(0,1).unsqueeze(0)

    sop_hpu = CustomSearchSorted()

    inds_h = sop_hpu(cdf_h, u_h, 1) # 1(right), 0(left)
    inds = inds_h.squeeze(0).transpose(0,1)
    ```



