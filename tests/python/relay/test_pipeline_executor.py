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

import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor


def run_modules(mod_configs, dev, target, dname, data):
    mod_input = {}
    final_output = {}
    indx = 1
    for mod in mod_configs:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)

        m = graph_executor.GraphModule(lib["default"](dev))
        # Get input information
        mod_key = "mod_{}".format(indx)
        if mod_input.has_key(mod_key):
            for input in mod_input[mod_key]:
                m.set_input("data_{}".format(input["index"]), input["data"])
        else:
            m.set_input(dname, data)

        m.run()
        n = m.get_num_outputs()
        # parse mod_config and set current output as next mod input data
        mconfig = mod_configs[mod]
        for output in mconfig["output"]:
            output_data = m.get_output(output["output_indx"]).asnumpy()
            for dep in output["dependent"]:
                for mod in dep:
                    if mod == "final":
                        final_output[mod_input[mod]["index"]] = output_data
                    else:
                        mod_input[mod]["index"] = dep[mod]["input_indx"]
                        mod_input[mod]["data"] = output_data
        #output = m.get_output(0).asnumpy()
        #data = output

        indx = indx + 1

    return output


def get_mannual_mod():
    mods = []
    dshape = (3, 3)
    data = relay.var("data", relay.TensorType(dshape, "float32"))
    data_net1_output_1 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    data_net1_output_2 = relay.var("data_2", relay.TensorType(dshape, "float32"))
    data_net2_output_1 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    mvalue1 = np.full((1), 5).astype("float32")
    mvalue2 = np.full((1), 2).astype("float32")
    mvalue3 = np.full((1), 3).astype("float32")
    mv1 = relay.Constant(tvm.nd.array(mvalue1))
    mv2 = relay.Constant(tvm.nd.array(mvalue2))
    mv3 = relay.Constant(tvm.nd.array(mvalue3))

    # net1 have three output, output3 is final output
    net_output1 = relay.add(data, mv1)
    net_output2 = relay.subtract(data, mv2)
    net_output3 = relay.multiply(data, mv3)

    # net2 use net1 output1 as input
    net2 = relay.add(data_net1_output_1, mv2)
    net2 = relay.add(net2, mv3)

    # net3 use net2 output1 and net1 outpu2 as input
    net3 = relay.multiply(data_net2_output_1, mv4)
    net3 = relay.add(net3, data_net1_output_2)

    mods.append(tvm.IRModule.from_expr(relay.Function([data],
                                       relay.Tuple([net_output1,
                                                    net_output2,
                                                    net_output3])
        )))
    mods.append(tvm.IRModule.from_expr(relay.Function([data_net1_output_1, data_net1_output_2],
                                                       net2)))
    mods.append(tvm.IRModule.from_expr(relay.Function([data_net2_output_1],
                                                       net3)))

    return mods, dshape


def run_pipeline(target):
    """
    #Get 4 pipeline module.
    """
    mods, dshape = get_mannual_mod()
    """
    #Prepare batch data for pipeline feeding
    """
    datas = []
    for i in range(len(mods) + 1):
        datas.append(np.full(dshape, 3 + i).astype("float32"))


    indx = 0
    if 0:
        mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
        mconfig1 = mconfig.copy()
        mconfig1["target"] = target[0]
        mconfig1["dev"] = target[1]
        # third output is final output, second output for mod2, third for  mod3
        # input
        mconfig1["output"] = {
                "output_1":{"output_indx":1,
                            "dependent":{"mod_2":{"mod_indx":2, "input_indx":1}}}, 
                "output_2":{"output_indx":2,
                            "dependent":{"mod_3":{"mod_indx":3, "input_indx":2}}},
                "output_3":{"output_indx":3, 
                            "dependent":{"final":{"mod_indx":0, "input_indx":1}}}
                             }
        mod_config[mods[0]] = mconfig1 

        mconfig2 = mconfig.copy()
        mconfig2["target"] = "llvm"
        mconfig2["dev"] = tvm.cpu(0)
        mconfig2["output"] = {
                "output_1":{"output_indx":1,
                            "dependent":{"mod_3":{"mod_indx":3, "input_indx":1}},
                             }
        mod_config[mods[1]] = mconfig2

        mconfig3 = mconfig.copy()
        mconfig3["target"] = "llvm"
        mconfig3["dev"] = tvm.cpu(0)
        mconfig3["output"] = {"output_indx":1,
                              "dpendent":{"final":{"mod_indx":0, "input_indx":2}}
                             }
        mod_config[mods[2]] = mconfig3


    """
    for mod in mods:
        mconfig = {"target_host": None, "mod_name": "default", "build": None, "params": None}
        # first two module use target that could be "cuda", "nvptx" etc.
        if indx < 2:
            mconfig["target"] = target[0]
            mconfig["dev"] = target[1]
        else:
            mconfig["target"] = "llvm"
            mconfig["dev"] = tvm.cpu()

        mod_config[mod] = mconfig
        indx = indx + 1
    #"""

    """
    #Run with graph executor for verification purpose
    """
    outs = [run_modules(mod_config, tvm.cpu(), "llvm", "data", data) for data in datas]
    """


    #build and create pipeline module
    """
    with relay.build_config(opt_level=3):
        pipeline_module = pipeline_executor.create(mods, mod_config)

    """
    #Use pipeline executor to pipeline the said pipeline which use different backend
    """
    for data in datas:
        pipeline_module.set_input("data", data)
        pipeline_module.run()

    """
    Get result
    """
    pipeline_outputs = []
    for i in range(len(datas)):
        pipeline_outputs.append(pipeline_module.get_output()[0].asnumpy())

    """
    #Stop pipeline execution.
    """
    pipeline_module.stop()
    """

    #Verify result
    """
    for ref_out, out in zip(outs, pipeline_outputs):
        tvm.testing.assert_allclose(ref_out, out)


def test_pipeline():
    target_list = tvm.testing.enabled_targets()
    for target in target_list:
        run_pipeline(target)


if __name__ == "__main__":
    test_pipeline()
