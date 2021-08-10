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
"""Pipeline executor that executes pipeline containing TVM PackedFunc."""
import json
import tvm._ffi
from tvm import relay
from tvm.contrib import graph_executor


def pipeline_executor_enabled():
    """check if pipeline executor enabled.
    Return
    ------
    enable: bool
        return pipeline executor get enabled or not
    """
    pipeline_enabled = False
    try:
        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        pipeline_enabled = True
    except ValueError:
        print("pipeline executor not enabled!")

    return pipeline_enabled


def build_pipeline(mod_n_configs):
    """build module list that can use for pipeline execution.

    Parameters
    ----------
    mod_n_configs: Dict[IRModule, Dict[str, Any]]
        build configuration informaton, structure like following.
        {IRModule: {"target":target,
                    "target_host":target_host,
                    "params":params,
                    "mod_name"mod_name,
                    "build":build}}

    Returns
    -------
    ret: List[IRModule]
        list of IRModule
    string_config: Dict[int, Dict[str, any]]
        pipeline configuration
    """
    mods = {}
    config_len = len(mod_n_configs)
    string_config = [{} for _ in range(config_len)]
    for _, (ir_mod, mod_config) in enumerate(mod_n_configs.items()):
        # init lib_name and json_name params with empty
        lib_name = ""
        json_name = ""
        params_name = ""
        # Get module configuration
        assert "pipeline" in mod_config and "mod_indx" in mod_config["pipeline"]
        # Get module index in pipeline configuration
        mconf = mod_config["pipeline"].copy()
        # Get mod device config
        dev = mod_config["dev"]
        mod_indx = mconf["mod_indx"] - 1
        target = mod_config["target"]
        assert mod_indx < config_len
        build_func = relay.build
        # if there is a self defined build function then use it.
        if "build" in mod_config and mod_config["build"]:
            build_func = mod_config["build"]

        # build IRModule
        mod = build_func(
            ir_mod,
            target,
            params=mod_config["params"],
            target_host=mod_config["target_host"],
            mod_name=mod_config["mod_name"],
        )

        mconf["lib_name"] = lib_name
        mconf["json_name"] = json_name
        mconf["params_name"] = params_name
        mconf["dev"] = "{},{}".format(dev.device_type, dev.device_id)
        # Create pipeline configuration
        string_config[mod_indx] = mconf
        # associate mod with device
        mods[mod] = {"dev": dev}

    # return IRModule list and pipeline configuration
    return mods, string_config


def create(pipeline_mods, mod_config):
    """Create a pipeline runtime executor.

    Parameters
    ----------
    pipeline_mods : List[IRModule]
        list of IRModule

    mod_config : Dict[int, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    Returns
    -------
    submodule : PipelineModule
        Runtime pipeline module.
    """

    submodule = PipelineModule(pipeline_mods, mod_config)
    return submodule


class PipelineModule(object):
    """Wrapper runtime module. This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output of underlying module functions.

    Parameters
    ----------
    graph_module : List[GraphModule]
        The internal tvm module that holds the actual graph functions.

    pipeline_config : Dict[IRModule, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    """

    def graph_executor_create(self, pipeline_mods, mod_config):
        """Create a pipeline runtime executor.

        Parameters
        ----------
        pipeline_mods : List[IRModule]
          list of IRModule

        mod_config : Dict[int, Dict[str, Any]]
            modules and modules dependency configuration informaiton.

        Returns
        -------
        mods : GreaphModule
            Runtime graph module.
        """

        mods = []
        for pipeline_mod in pipeline_mods:
            mod = graph_executor.GraphModule(
                pipeline_mod["default"](pipeline_mods[pipeline_mod]["dev"])
            )
            mods.append(mod.module)

        return mods, json.dumps(mod_config)

    def __init__(self, pipeline_mods, mod_config):
        self.pipeline_mods = pipeline_mods
        self.mod_config = mod_config
        mods, config = self.graph_executor_create(pipeline_mods, mod_config)

        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        module = pipelinecreate(mods, config)

        self.module_ = module
