import os
import sys
import glob
import numpy as np

ENV = sys.prefix

# =========================================================
# ADD NVIDIA DLL FOLDERS
# =========================================================
nvidia_root = ENV + r"\Lib\site-packages\nvidia"

dll_dirs = []

for d in glob.glob(nvidia_root + r"\*\bin"):
 dll_dirs.append(d)

dll_dirs.append(ENV + r"\Library\bin")

for d in dll_dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        print("Added:", d)

# =========================================================
# ONNX TEST
# =========================================================
import onnxruntime as ort

print("\n===================================")
print("ONNX")
print("===================================")

print("Python:", sys.executable)
print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
print("ONNX device:", ort.get_device())

session = ort.InferenceSession(
 r"E:\Desktop\ccnb\best_unet_resnet101_norm_1024.ONNX",
 providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Active providers:", session.get_providers())

if session.get_providers()[0] != "CUDAExecutionProvider":
 raise RuntimeError("ONNX is NOT using CUDA.")

# =========================================================
# DUMMY INFERENCE
# =========================================================
x = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

input_name = session.get_inputs()[0].name

y = session.run(None, {input_name: x})

print("Output shape:", y[0].shape)

print("\nSUCCESS")