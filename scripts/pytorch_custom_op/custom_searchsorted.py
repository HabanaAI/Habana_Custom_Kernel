###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_searchsorted_op_lib_path = "./build/lib.linux-x86_64-cpython-38/hpu_custom_searchsorted.cpython-38-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_searchsorted_op_lib_path))

class CustomSearchSortedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, value, side):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_searchsorted(sequence, value, side)
        ctx.tensor = tensor
        return tensor

class CustomSearchSorted(torch.nn.Module):
    def __init__(self):
        super(CustomSearchSorted, self).__init__()

    def forward(self, sequence, value, side):
        return CustomSearchSortedFunction.apply(sequence, value, side)

    def extra_repr(self):
        return 'CustomSearchSorted for float32 only'

