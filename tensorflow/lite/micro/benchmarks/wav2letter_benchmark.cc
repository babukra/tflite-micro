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
#include "tensorflow/lite/micro/models/tiny_wav2letter_int8_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"

/*
 * Wav2letter Spotting Benchmark for performance optimizations. The model used in
 * this benchmark only serves as a reference. The values assigned to the model
 * weights and parameters are not representative of the original model.
 */

namespace tflite {

using Wav2letterBenchmarkRunner = MicroBenchmarkRunner<int16_t>;
using Wav2letterOpResolver = MicroMutableOpResolver<6>;

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int kTensorArenaSize = 1024 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

uint8_t benchmark_runner_buffer[sizeof(Wav2letterBenchmarkRunner)];
uint8_t op_resolver_buffer[sizeof(Wav2letterOpResolver)];

// Initialize benchmark runner instance explicitly to avoid global init order
// issues on Sparkfun. Use new since static variables within a method
// are automatically surrounded by locking, which breaks bluepill and stm32f4.
Wav2letterBenchmarkRunner* CreateBenchmarkRunner(MicroProfiler* profiler) {
  // We allocate the Wav2letterOpResolver from a global buffer because the object's
  // lifetime must exceed that of the Wav2letterBenchmarkRunner object.
  Wav2letterOpResolver* op_resolver = new (op_resolver_buffer) Wav2letterOpResolver();
  op_resolver->AddReshape();
  op_resolver->AddConv2D();
  op_resolver->AddLeakyRelu();
  op_resolver->AddSoftmax();
  return new (benchmark_runner_buffer)
      Wav2letterBenchmarkRunner(g_tiny_wav2letter_int8_model_data, op_resolver,
                             tensor_arena, kTensorArenaSize, profiler);
}

void Wav2letterRunNIerations(int iterations, const char* tag,
                          Wav2letterBenchmarkRunner& benchmark_runner,
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

  uint32_t event_handle = profiler.BeginEvent("InitializeWav2letterRunner");
  tflite::Wav2letterBenchmarkRunner* benchmark_runner = CreateBenchmarkRunner(&profiler);
  profiler.EndEvent(event_handle);

  MicroPrintf("");  // null MicroPrintf serves as a newline.
  
  tflite::Wav2letterRunNIerations(1, "Wav2letterRunNIerations(1)", *benchmark_runner, profiler);
  MicroPrintf("");
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  //benchmark_runner->PrintAllocations();
}
