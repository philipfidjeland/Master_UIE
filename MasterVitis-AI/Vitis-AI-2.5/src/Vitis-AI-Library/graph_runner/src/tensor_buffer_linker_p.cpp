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
#include "./tensor_buffer_linker_p.hpp"

#include <glog/logging.h>

#include <map>
#include <memory>
#include <sstream>
#include <vart/batch_tensor_buffer_view.hpp>
#include <vitis/ai/profiling.hpp>

#include "./tensor_buffer_shared.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_GRAPH_RUNNER, "0");
DEF_ENV_PARAM(XLNX_DISABLE_LINKER_BUFFER, "0");
TensorBufferLinkerHostPhy::TensorBufferLinkerHostPhy(
    std::unique_ptr<vart::TensorBuffer>* master)
    : TensorBufferLinker{master} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "TensorBufferLinkerHostPhy create "
      << "@" << (void*)this;
}

TensorBufferLinkerHostPhy::~TensorBufferLinkerHostPhy() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "TensorBufferLinkerHostPhy destory "
      << "@" << (void*)this;
}

void TensorBufferLinkerHostPhy::finalize() {
  static constexpr int HOST_PHY = 0;
  static constexpr int HOST_VIRT = 1;
  //----------------------------
  auto kind = [](std::unique_ptr<vart::TensorBuffer>* s) {
    auto ret = HOST_VIRT;
    if ((*s)->get_location() == vart::TensorBuffer::location_t::HOST_PHY) {
      ret = HOST_PHY;
    } else if ((*s)->get_location() ==
               vart::TensorBuffer::location_t::HOST_VIRT) {
      ret = HOST_VIRT;
    } else {
      LOG(FATAL) << "TensorBufferLinkerHostPhy ： not support DEVICE";
    }
    return ret;
  };
  //--------------------------
  // build replacement_
  replacement_ = master_;
  for (auto s : slaves_) {
    if (kind(s.first) < kind(replacement_)) {
      replacement_ = s.first;
    }
  }
  auto real = std::shared_ptr<vart::TensorBuffer>(std::move(*replacement_));
  *replacement_ =
      std::make_unique<vart::TensorBufferShared>(real, real->get_tensor());

  LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
      << "replacement_ " << replacement_->get()->to_string();
  //---------------------------
  auto decide =
      [this, kind](
          std::pair<std::unique_ptr<vart::TensorBuffer>*, const xir::Subgraph*>
              s) {
        int ret = REPLACE;
        if (s.first == replacement_) {
          ret = THE_SELECTED;
        } else {
          switch (kind(s.first)) {
            case HOST_PHY:
              ret = KEEP;
              break;
            case HOST_VIRT:
              ret = REPLACE;
              break;
          }
        }
        if (ENV_PARAM(XLNX_DISABLE_LINKER_BUFFER)) {
          ret = KEEP;
        }
        // ret = KEEP;
        // LOG(INFO) << "===========ALL KEEP===========";
        return ret;
      };
  // ---------------------------
  // -- replace others
  auto replace = [real](std::unique_ptr<vart::TensorBuffer>* x) {
    // be careful about the ownership of the tensor
    auto tensor = xir::Tensor::clone((*x)->get_tensor());
    *x = std::make_unique<vart::TensorBufferShared>(real, tensor.get());
  };
  if (master_ != replacement_) {
    replace(master_);
  }
  int index = 0;
  linker_decisions_ = vitis::ai::vec_map(slaves_, decide);
  for (auto s : slaves_) {
    switch (linker_decisions_[index]) {
      case REPLACE:
        replace(s.first);
        break;
      case THE_SELECTED:
      case KEEP:
      default:
        // do nothing
        break;
    }
    index++;
  }
}

void TensorBufferLinkerHostPhy::after_invoke_runner(
    const xir::Subgraph* subgraph) {
  int index = 0;
  for (auto s : slaves_) {
    if (linker_decisions_[index] == KEEP) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_GRAPH_RUNNER))
          << " copy tensor buffer \n\tfrom" << replacement_->get()->to_string()
          << " \n\tto " << s.first->get()->to_string();
      LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))
          << " ZERO_COPY = 0  tensor name : "
          << s.first->get()->get_tensor()->get_name()  //
          << " From : " << subgraph->get_attr<std::string>("device") << " "
          << subgraph->get_name()  //
          << " To : " << s.second->get_attr<std::string>("device") << " "
          << s.second->get_name();

      __TIC__(COPY_TENSOR_BUFFER)
      vart::TensorBuffer::copy_tensor_buffer(replacement_->get(),
                                             s.first->get());
      __TOC__(COPY_TENSOR_BUFFER)
    } else {
      LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))
          << " ZERO_COPY = 1  tensor name : "
          << s.first->get()->get_tensor()->get_name()  //
          << " From : " << subgraph->get_attr<std::string>("device") << " "
          << subgraph->get_name()  //
          << " To : " << s.second->get_attr<std::string>("device") << " "
          << s.second->get_name();
    }
    index++;
  }
  return;
}
