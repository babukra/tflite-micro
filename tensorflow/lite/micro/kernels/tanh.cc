/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/tanh.h"

namespace tflite {
namespace {

void* TanhInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataTanh));
}

TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTanhInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTanhOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataTanh& data = *(static_cast<const OpDataTanh*>(node->user_data));

  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::Tanh(tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<float>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      TanhParams params;
      params.input_left_shift = data.input_left_shift;
      reference_ops::Tanh(params, tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<int16_t>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      TanhParams params;
      params.input_zero_point = data.input_zero_point;
      params.input_range_radius = data.input_range_radius;
      params.input_multiplier = data.input_multiplier;
      params.input_left_shift = data.input_left_shift;
      reference_ops::Tanh(params, tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<uint8_t>(input),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<uint8_t>(output));

      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      reference_integer_ops::Tanh(
          data.input_zero_point, data.input_range_radius, data.input_multiplier,
          data.input_left_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                         TfLiteTypeGetName(input->type),
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
}

}  // namespace

TfLiteRegistration Register_TANH() {
  return {/*init=*/TanhInit,
          /*free=*/nullptr,
          /*prepare=*/TanhPrepare,
          /*invoke=*/TanhEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}
}  // namespace tflite
