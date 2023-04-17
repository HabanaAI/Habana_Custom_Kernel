/******************************************************************************
###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>
typedef struct sParam{
    int side;
}searchParam;

bool register_custom_searchsorted() {
    // Registering custom_op::custom_searchsorted
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::SCALAR, 0};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[1].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Int, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(searchParam);
      params->side = inputs[2].toInt(); // bottom
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_searchsorted", //schema name
        "searchsorted_fwd_f32", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_searchsorted\n";
    return true;
}

at::Tensor custom_searchsorted_execute(
    torch::Tensor sequence,
    torch::Tensor value,
    c10::Scalar side) {
  TORCH_CHECK(sequence.scalar_type() == c10::ScalarType::Float, "Input sequence expected to be Float tensor");
  TORCH_CHECK(value.scalar_type() == c10::ScalarType::Float, "Input value expected to be Float tensor");
  TORCH_CHECK(side.to<int>() == 0 || side.to<int>() == 1, "side values other than 0 or 1 are not supported")
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_searchsorted();
  TORCH_CHECK(registered, "custom_searchsorted kernel not registered" );
  std::vector<c10::IValue> inputs{sequence, value, side};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_searchsorted");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_searchsorted(Tensor self, Tensor value, Scalar side) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_searchsorted", custom_searchsorted_execute);
}

