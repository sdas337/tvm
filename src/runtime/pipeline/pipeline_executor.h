/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief pipeline executor
 * \file pipeline_executor.h
 */
#ifndef TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
#define TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
#include <tvm/runtime/registry.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
using namespace std;
namespace tvm {
namespace runtime {

/*!
 * \brief pipeline runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class TVM_DLL SubGraphRuntime : public ModuleNode {
 public:
  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "SubGraphRuntime"; }
  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   * \param lookup_linked_param_func If given, a PackedFunc invoked to lookup linked parameters
   *  by storage_id. If not given, linked parameters are looked-up using an internal implementation,
   *  which is not compatible with RPCModules.
   */
  void Init(const Array<tvm::runtime::Module>& modules, const std::string& pipeline_json);
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PIPELINE_PIPELINE_EXECUTOR_H_
