# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Minimum subgraph executor that executes subgraph containing TVM PackedFunc."""
import tvm._ffi
from tvm.contrib import graph_executor


def create(sub_mods):
    """Create a subgraph runtime executor.

    Parameters
    ----------
    sub_mods :
        {"lib": <module>,
         "dev": <device>}

    Returns
    -------
    submodule : SubGraphModule
        Runtime subgraph module.
    """
    mods = []
    for sub_mod in sub_mods:
        mod = graph_executor.GraphModule(sub_mod["default"](sub_mods[sub_mod]["dev"]))
        mods.append(mod)

    submodule = SubGraphModule(mods)
    return submodule


class SubGraphModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

    """

    def __init__(self, graph_modules):
        mods = []
        for module in graph_modules:
            mods.append(module.module)

        subgraphcreate = tvm._ffi.get_global_func("tvm.subgraph_executor.create")
        module = subgraphcreate(mods)

        self.graph_modules_ = graph_modules

        self._set_input = module["set_input"]
        self._run = module["run"]
        self._stop = module["stop"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_num_inputs = module["get_num_inputs"]

    def set_input(self, key=None, value=None, params=None):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if key is not None:
            self.graph_modules_[0].set_input(key, value)

        if params:
            indx = 0
            for param in params:
                self.graph_modules_[indx].set_input(**param)
                indx = indx + 1

    def run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def stop(self):
        """Stop subgraph run"""
        self._stop()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the graph

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, input_indx, runtime_index=0, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(input_indx, runtime_index).copyto(out)
            return out

        return self._get_input(input_indx, runtime_index)

    def get_output(self):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index
        """
        return self._get_output()

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]
