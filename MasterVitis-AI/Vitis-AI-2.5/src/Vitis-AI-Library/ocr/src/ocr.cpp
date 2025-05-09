/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "vitis/ai/ocr.hpp"

#include "./ocr_imp.hpp"

namespace vitis {
namespace ai {

OCR::OCR() {}
OCR::~OCR() {}

std::unique_ptr<OCR> OCR::create(
    const std::string &model_name, bool need_mean_scale_process) {
  return std::unique_ptr<OCR>(
      new OCRImp(model_name, need_mean_scale_process));
}

}  // namespace ai
}  // namespace vitis
