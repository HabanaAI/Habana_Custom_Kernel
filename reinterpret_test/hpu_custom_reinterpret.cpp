#include "hpu_custom_op.h"

#include <torch/extension.h>

#include <synapse_common_types.hpp>


bool register_custom_reinterpret() {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc {
        habana::custom_op::input_type::TENSOR, 0
    };
    std::vector<habana::custom_op::InputDesc> inputs_desc { input_a_desc };

    auto output_size_lambda = [](const at::Stack& inputs) -> std::vector<int64_t> {
        return inputs[0].toTensor().sizes().vec(); // Output shape is same as input tensor shape
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Int, output_size_lambda}; // Output dtype will be set in execute function
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};
    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::reinterpret_float", //schema name
        "reinterpret_fwd_i32", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::reinterpret_float\n";
    return true;
}

at::Tensor custom_reinterpret_execute(torch::Tensor input_a) 
{
    // Registering the custom op, need to be called only once
    static bool registered = register_custom_reinterpret();
    TORCH_CHECK(registered, "custom_reinterpret kernel not registered" );
    std::vector<c10::IValue> inputs{input_a};
    
    // Get custom op descriptor from registry
    auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::reinterpret_float");


    // Actual call for op execution
    std::vector<at::Tensor> output = op_desc.execute(inputs);
    
    return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("reinterpret_float(Tensor self) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("reinterpret_float", custom_reinterpret_execute);
}


