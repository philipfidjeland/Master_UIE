#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

vai_add_library(
  SRCS include/vitis/ai/ocr.hpp src/ocr.cpp
       src/ocr_imp.cpp src/ocr_imp.hpp
  PUBLIC_REQUIRE
    ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)

vai_add_test(test_ocr)
vai_add_test(test_ocr_accuracy)
vai_add_test(test_ocr_batch)

