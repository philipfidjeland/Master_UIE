#
# Copyright 2022 Xilinx Inc.
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
  SRCS
  include/vitis/ai/ofa_yolo.hpp
  src/ofa_yolo.cpp
  src/ofa_yolo_imp.cpp
  src/ofa_yolo_imp.hpp
  src/utils.hpp
  src/utils.cpp
  PUBLIC_REQUIRE
  ${PROJECT_NAME}::xnnpp
  ${PROJECT_NAME}::dpu_task)

vai_add_test(test_ofa_yolo)
vai_add_test(test_ofa_yolo_accuracy)
vai_add_test(test_ofa_yolo_batch)
