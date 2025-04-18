#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_C_STANDARD 99)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Werror -ggdb -O0 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-inline")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
  set(CMAKE_EXE "${CMAKE_C_FLAGS} -Wall -Werror")
  set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

  set(CMAKE_MACOSX_RPATH 1)
else(NOT MSVC)
	string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(NOT MSVC)

if(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_DEFINITIONS(-DUNI_LOG_NDEBUG)
endif()

include(CMakePackageConfigHelpers)

if(NOT IS_EDGE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH True)
endif()

##
if(CMAKE_CROSSCOMPILING)
  set(_IS_EDGE_DEFULAT_VALUE ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(_IS_EDGE_DEFULAT_VALUE ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(_IS_EDGE_DEFULAT_VALUE ON)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(_IS_EDGE_DEFULAT_VALUE OFF)
else()
  set(_IS_EDGE_DEFULAT_VALUE OFF)
endif(CMAKE_CROSSCOMPILING)
option(IS_EDGE "ENABLE building for edge platform" ${_IS_EDGE_DEFULAT_VALUE})
message(STATUS "building for edge ${IS_EDGE}")

add_library(gcc_atomic INTERFACE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(microblazeel)")
  # for a unknown reason, gcc on microblaze requires libatomic
  target_link_libraries(gcc_atomic INTERFACE -latomic)
endif()
