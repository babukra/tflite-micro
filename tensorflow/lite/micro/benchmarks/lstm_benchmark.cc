/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/benchmarks/micro_benchmark.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/svdf.h"
//#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/models/lstm_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"

/*
 * LSTM Benchmark for performance optimizations. The model used in
 * this benchmark only serves as a reference. The values assigned to the model
 dfdsf weights and parameters are not representative of the original model.
 */

namespace tflite {

using LSTMBenchmarkRunner = MicroBenchmarkRunner<int16_t>;
using LSTMOpResolver = MicroMutableOpResolver<8>;

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int kTensorArenaSize = 21 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

uint8_t benchmark_runner_buffer[sizeof(LSTMBenchmarkRunner)];
uint8_t op_resolver_buffer[sizeof(LSTMOpResolver)];

// Initialize benchmark runner instance explicitly to avoid global init order
// issues on Sparkfun. Use new since static variables within a method
// are automatically surrounded by locking, which breaks bluepill and stm32f4.
LSTMBenchmarkRunner* CreateBenchmarkRunner(MicroProfiler* profiler) {
  // We allocate the LSTMOpResolver from a global buffer because the object's
  // lifetime must exceed that of the LSTMBenchmarkRunner object.
  LSTMOpResolver* op_resolver = new (op_resolver_buffer) LSTMOpResolver();
  op_resolver->AddUnpack();
  op_resolver->AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
  op_resolver->AddSplit();
  op_resolver->AddLogistic();
  op_resolver->AddMul();
  op_resolver->AddTanh();
  op_resolver->AddAdd(tflite::Register_ADD());
  op_resolver->AddReshape();
  //op_resolver->AddQuantize();
  //op_resolver->AddSoftmax(tflite::Register_SOFTMAX_INT8_INT16());
  //op_resolver->AddSvdf(tflite::Register_SVDF_INT8());

  return new (benchmark_runner_buffer)
      LSTMBenchmarkRunner(g_lstm_model_data, op_resolver, tensor_arena, kTensorArenaSize, profiler);
}

void LSTMRunNIerations(int iterations, const char* tag,
                          LSTMBenchmarkRunner& benchmark_runner,
                          MicroProfiler& profiler) {
  int32_t ticks = 0;
  for (int i = 0; i < iterations; ++i) {
    benchmark_runner.SetRandomInput(i);
    profiler.ClearEvents();
    benchmark_runner.RunSingleIteration();
    ticks += profiler.GetTotalTicks();
  }
  MicroPrintf("%s took %d ticks or (%d ms)", tag, ticks, TicksToMs(ticks));
}

}  // namespace tflite

int main(int argc, char** argv) {
  tflite::InitializeTarget();
  tflite::MicroProfiler profiler;

  uint32_t event_handle = profiler.BeginEvent("InitializeLSTMRunner");
  tflite::LSTMBenchmarkRunner* benchmark_runner = CreateBenchmarkRunner(&profiler);
  profiler.EndEvent(event_handle);

  MicroPrintf("");  // null MicroPrintf serves as a newline.
  
  tflite::LSTMRunNIerations(1, "LSTMRunNIerations(1)", *benchmark_runner, profiler);
  MicroPrintf("");
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  //benchmark_runner->PrintAllocations();
}
