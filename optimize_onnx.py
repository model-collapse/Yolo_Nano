import onnx
from onnx import optimizer

# Preprocessing: load the model to be optimized.
model_path = './yolo-nano_person.sim.onnx'
original_model = onnx.load(model_path)

# A full list of supported optimization passes can be found using get_available_passes()
all_passes = optimizer.get_available_passes()
print("Available optimization passes:")
for p in all_passes:
    print(p)
print()

# Pick one pass as example
passes = [
            'eliminate_deadend'
          #'fuse_bn_into_conv'
          ]

# Apply the optimization on the original model
optimized_model = optimizer.optimize(original_model, passes, fix_point=False)
onnx.save(optimized_model, "./yolo-nano_person.optm.onnx")
