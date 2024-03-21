###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_ops_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_outer.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_ops_op_lib_path))

class CustomOuterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_outer(input1, input2)
        ctx.tensor = tensor
        return tensor


class CustomOuter(torch.nn.Module):
    def __init__(self):
        super(CustomOuter, self).__init__()

    def forward(self, input1, input2):
        return CustomOuterFunction.apply(input1, input2)

    def extra_repr(self):
        return 'CustomOuterFunction for float32 only'

