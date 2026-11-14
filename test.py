import torch
import os

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

print("hehe",torch_lib)