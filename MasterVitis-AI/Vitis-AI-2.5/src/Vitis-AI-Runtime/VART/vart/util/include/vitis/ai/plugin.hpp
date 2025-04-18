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
#pragma once
#include <string>
#if _WIN32
#include <windows.h>
#include <libloaderapi.h>
#else
#include <dlfcn.h>
#endif

namespace vitis {
namespace ai {
#if _WIN32
using plugin_t = HMODULE;
#else
using plugin_t = void *;
#endif 
enum class scope_t { PUBLIC, PRIVATE };
plugin_t open_plugin(const std::string& name, scope_t scope);
void* plugin_sym(plugin_t plugin, const std::string& name);
std::string plugin_error(plugin_t plugin);
void close_plugin(plugin_t plugin);
} // namespace ai
}  // namespace vitis
