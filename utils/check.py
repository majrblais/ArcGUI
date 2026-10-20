#!/usr/bin/env python
# ============================================================
# ArcGUI ONNX model check
#
# Launch this script, select an ONNX model, and it will:
#   1) detect SegFormer vs U-Net ResNet101 from the filename,
#   2) select CUDA/CPU the same way as ArcGUI,
#   3) create an ONNX Runtime session,
#   4) validate the expected 1024x1024 RGB model contract,
#   5) run a real dummy inference.
#
# Supported naming families:
#   - filename contains "segformer"
#   - filename contains "unet" AND "resnet101" or "r101"
#
# Execution target:
#   - filename contains "cpu"             -> CPU only
#   - filename contains "cuda" or "fp16" -> CUDA required
#   - otherwise                            -> CUDA preferred, CPU fallback
# ============================================================

from pathlib import Path
import glob
import os
import sys
import time

import numpy as np

MODEL_INPUT_SIZE = 1024


# ------------------------------------------------------------
# FILE PICKER
# ------------------------------------------------------------

def choose_model_file():
    """Open a file picker; fall back to a typed path if Tk is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        selected = filedialog.askopenfilename(
            title="Select ArcGUI ONNX model",
            filetypes=[
                ("ONNX models", "*.onnx"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if not selected:
            raise SystemExit("No model was selected.")

        return Path(selected).expanduser().resolve()

    except SystemExit:
        raise
    except Exception as exc:
        print(f"File picker unavailable ({exc}).")
        selected = input("Enter the full path to the ONNX model: ").strip().strip('"')
        if not selected:
            raise SystemExit("No model was selected.")
        return Path(selected).expanduser().resolve()


# ------------------------------------------------------------
# MODEL DETECTION
# ------------------------------------------------------------

def detect_model_profile(model_path):
    path = Path(model_path)
    stem = path.stem.lower()

    if path.suffix.lower() != ".onnx":
        raise ValueError("The selected file must have the .onnx extension.")

    if "segformer" in stem:
        architecture = "SegFormer MiT-B5"
        family = "segformer"
    elif "unet" in stem and ("resnet101" in stem or "r101" in stem):
        architecture = "U-Net ResNet101"
        family = "unet_resnet101"
    else:
        raise ValueError(
            "Unsupported ONNX filename.\n\n"
            "Accepted families are:\n"
            "  - SegFormer: filename contains 'segformer'\n"
            "  - U-Net ResNet101: filename contains 'unet' and 'resnet101'/'r101'\n\n"
            f"Selected: {path.name}"
        )

    if "cpu" in stem:
        target = "cpu"
    elif "cuda" in stem or "fp16" in stem:
        target = "cuda"
    else:
        target = "auto"

    return {
        "family": family,
        "architecture": architecture,
        "target": target,
    }


# ------------------------------------------------------------
# SELECT MODEL BEFORE IMPORTING ONNX RUNTIME
# ------------------------------------------------------------

MODEL_PATH = choose_model_file()

if not MODEL_PATH.exists() or not MODEL_PATH.is_file():
    raise FileNotFoundError(f"ONNX model not found:\n{MODEL_PATH}")

if MODEL_PATH.stat().st_size <= 0:
    raise RuntimeError(f"The selected ONNX file is empty:\n{MODEL_PATH}")

PROFILE = detect_model_profile(MODEL_PATH)


# ------------------------------------------------------------
# WINDOWS / CONDA / NVIDIA DLL DISCOVERY
# ------------------------------------------------------------

def add_dll_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return

    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(directory))
        except Exception:
            pass

    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


ENV = Path(sys.prefix)
add_dll_directory(ENV / "Library" / "bin")

# Discover NVIDIA DLLs for CUDA/auto models.
if PROFILE["target"] in ("cuda", "auto"):
    nvidia_root = ENV / "Lib" / "site-packages" / "nvidia"
    for directory in glob.glob(str(nvidia_root / "*" / "bin")):
        add_dll_directory(directory)


# ------------------------------------------------------------
# IMPORT ONNX RUNTIME
# ------------------------------------------------------------

try:
    import onnxruntime as ort
except Exception as exc:
    raise RuntimeError(
        "Could not import onnxruntime.\n"
        "For CUDA models, install/use onnxruntime-gpu in this environment.\n"
        f"Original error: {exc}"
    ) from exc


# Best-effort explicit CUDA DLL preload.
if PROFILE["target"] in ("cuda", "auto"):
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            print("CUDA/cuDNN/MSVC DLL preload: OK")
    except Exception as exc:
        print("CUDA DLL preload warning:", exc)


# ------------------------------------------------------------
# ENVIRONMENT REPORT
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("ARCGUI ONNX MODEL CHECK")
print("=" * 76)
print("Model             :", MODEL_PATH)
print("Architecture      :", PROFILE["architecture"])
print("Filename target   :", PROFILE["target"].upper())
print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("ONNX Runtime      :", ort.__version__)
print("ORT device        :", ort.get_device())
print("Available EPs     :", ort.get_available_providers())


# ------------------------------------------------------------
# PROVIDER SELECTION
# ------------------------------------------------------------

available = ort.get_available_providers()

target = PROFILE["target"]

if target == "cuda":
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "This model is marked as CUDA/FP16 by its filename, but "
            "CUDAExecutionProvider is not available.\n\n"
            "Use an environment with onnxruntime-gpu and compatible CUDA/cuDNN."
        )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    effective_target = "cuda"

elif target == "cpu":
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("CPUExecutionProvider is not available.")
    providers = ["CPUExecutionProvider"]
    effective_target = "cpu"

else:
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        effective_target = "cuda"
    elif "CPUExecutionProvider" in available:
        providers = ["CPUExecutionProvider"]
        effective_target = "cpu"
    else:
        raise RuntimeError("Neither CUDAExecutionProvider nor CPUExecutionProvider is available.")


# ------------------------------------------------------------
# CREATE SESSION
# ------------------------------------------------------------

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

start_load = time.perf_counter()

try:
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=providers,
    )
except Exception as exc:
    raise RuntimeError(
        f"ONNX Runtime could not load the selected model using {providers}.\n"
        f"Model: {MODEL_PATH}\n"
        f"Original error: {exc}"
    ) from exc

load_seconds = time.perf_counter() - start_load
active = session.get_providers()

print("Requested EPs     :", providers)
print("Active EPs        :", active)
print("Effective target  :", effective_target.upper())
print("Session load time : {:.2f} s".format(load_seconds))

if effective_target == "cuda" and "CUDAExecutionProvider" not in active:
    raise RuntimeError(
        "The model session was created, but CUDAExecutionProvider is not active. "
        "Stopping instead of silently accepting CPU-only execution."
    )

if effective_target == "cpu" and (not active or active[0] != "CPUExecutionProvider"):
    raise RuntimeError("The model did not activate CPUExecutionProvider as requested.")


# ------------------------------------------------------------
# MODEL CONTRACT CHECK
# ------------------------------------------------------------

inputs = session.get_inputs()
outputs = session.get_outputs()

if len(inputs) != 1:
    raise RuntimeError(f"Expected exactly one ONNX input, found {len(inputs)}.")

if len(outputs) != 1:
    raise RuntimeError(f"Expected exactly one ONNX output, found {len(outputs)}.")

input_meta = inputs[0]
output_meta = outputs[0]

print("\nInput name        :", input_meta.name)
print("Input type        :", input_meta.type)
print("Input shape       :", input_meta.shape)
print("Output name       :", output_meta.name)
print("Output type       :", output_meta.type)
print("Output shape      :", output_meta.shape)

if input_meta.type != "tensor(float)":
    raise RuntimeError(f"Expected FP32 input tensor, got {input_meta.type}.")

shape = input_meta.shape

if len(shape) != 4:
    raise RuntimeError(f"Expected NCHW input [B,3,H,W], got {shape}.")

if isinstance(shape[0], int) and shape[0] not in (1,):
    raise RuntimeError(
        f"This simple checker runs batch=1, but the model requires static batch={shape[0]}."
    )

if isinstance(shape[1], int) and shape[1] != 3:
    raise RuntimeError(f"Expected 3 RGB channels, got input shape {shape}.")

if isinstance(shape[2], int) and shape[2] != MODEL_INPUT_SIZE:
    raise RuntimeError(f"Expected height {MODEL_INPUT_SIZE}, got {shape[2]}.")

if isinstance(shape[3], int) and shape[3] != MODEL_INPUT_SIZE:
    raise RuntimeError(f"Expected width {MODEL_INPUT_SIZE}, got {shape[3]}.")

if "float" not in output_meta.type:
    raise RuntimeError(f"Expected floating-point model output, got {output_meta.type}.")


# ------------------------------------------------------------
# REAL DUMMY INFERENCE
# ------------------------------------------------------------

# Normalized-looking input; exact values are irrelevant for compatibility.
rng = np.random.default_rng(12345)
x = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
).astype(np.float32)

try:
    _ = session.run([output_meta.name], {input_meta.name: x})[0]
except Exception as exc:
    raise RuntimeError(
        "The model loaded, but dummy inference failed.\n"
        f"Original error: {exc}"
    ) from exc

times = []
y = None

for _ in range(3):
    start = time.perf_counter()
    y = session.run([output_meta.name], {input_meta.name: x})[0]
    times.append(time.perf_counter() - start)

y = np.asarray(y)

if not np.all(np.isfinite(y)):
    raise RuntimeError("Dummy inference output contains NaN or infinite values.")

print("\nDummy output shape :", y.shape)
print("Dummy output dtype :", y.dtype)
print("Output min/max     :", float(y.min()), float(y.max()))
print("Average inference  : {:.2f} ms".format(float(np.mean(times)) * 1000.0))

print("\n" + "=" * 76)
if effective_target == "cuda":
    print("SUCCESS - MODEL LOADED AND INFERENCE RAN WITH CUDA")
else:
    print("SUCCESS - MODEL LOADED AND INFERENCE RAN WITH CPU")
print("=" * 76)
