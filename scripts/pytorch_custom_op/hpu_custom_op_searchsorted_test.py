###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
from custom_searchsorted import CustomSearchSorted

def test_custom_searchsorted_op_function():
    print(torch.ops.custom_op.custom_searchsorted)
    input = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0], [2.0, 4.0, 6.0, 8.0, 10.0]], requires_grad=True)
    value = torch.tensor([[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]], requires_grad=True)
    
    output_cpu = torch.searchsorted(input, value, side='right')
    print(output_cpu)
    
    input_h=input.transpose(0,1).unsqueeze(0)
    value_h=value.transpose(0,1).unsqueeze(0)

    input_hpu = input_h.to('hpu').detach()
    value_hpu = value_h.to('hpu').detach()

    input_hpu.requires_grad = True
    sop_hpu = CustomSearchSorted()
    output = sop_hpu(input_hpu, value_hpu, 1)
    output_hpu = output.squeeze(0).transpose(0,1)
    print(output_hpu)
    assert(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))
    print("Searchsorted forward passed!!")

test_custom_searchsorted_op_function()

