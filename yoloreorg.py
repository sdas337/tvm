import tvm
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor

transformFuseTest = True
if transformFuseTest:
    shape = (1, 1, 2, 2, 4)
    var = relay.var("data", shape=shape, dtype="float32")
    net = relay.layout_transform(var, "NCHW4c", "NCHW")
else:
    shape = (1, 4, 2, 2)
    var = relay.var("data", shape=shape, dtype="float32")
    net = var

net = relay.vision.yolo_reorg(net, stride=2)
#net = relay.reshape(net, (1, 256, 19, 19))
func = relay.Function([var], net)

mod = tvm.IRModule()
mod["main"] = func
target = tvm.target.Target("llvm", host="llvm")
tvm.transform.PrintIR()(mod)
with tvm.transform.PassContext(opt_level=1):
    print(relay.build(mod, target="c").module.imported_modules[0].get_source())
    lib = relay.build(mod, target="llvm")
#'''
m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

#data = np.fromfile("/scratch/hj/json0", np.float32).reshape(shape)
data = np.full(shape, 1, np.float32)

for c in range(shape[1]):
    for h in range(shape[2]):
        for w in range(shape[3]):
            for c4 in range(shape[4]):
                data[0][c][h][w][c4] = c * 16 + h * shape[3] * shape[4] + w * shape[4] + c4

m.set_input("data", data)
m.run()
out = m.get_output(0).numpy()
out.tofile("/scratch/hj/dtest")
print(data.reshape((data.shape[1]*data.shape[2]*data.shape[3]*data.shape[4], 1)))
print(out)
#'''
