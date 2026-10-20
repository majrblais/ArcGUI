#!/usr/bin/env python
# ============================================================
# BARE-BONES ARCGUI ONNX FOLDER INFERENCE EXAMPLE
#
# Launch the script and select:
#   1) an ONNX model
#   2) a folder containing images
#
# The script creates one binary road mask per image.
#
# Supported model filename families:
#   - SegFormer: filename contains "segformer"
#   - U-Net ResNet101: filename contains "unet" and "resnet101"/"r101"
#
# Execution target:
#   - contains "cpu"             -> CPU only
#   - contains "cuda" or "fp16" -> CUDA required
#   - otherwise                  -> CUDA preferred, CPU fallback
#
# Expected model preprocessing:
#   RGB -> 1024x1024 -> ImageNet normalization -> NCHW float32
#
# Expected output:
#   one-channel logits -> sigmoid -> threshold -> binary mask
# ============================================================

from pathlib import Path
import glob
import os
import sys

import numpy as np
from PIL import Image

MODEL_INPUT_SIZE = 1024
THRESHOLD = 0.50

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


# ------------------------------------------------------------
# PICKERS
# ------------------------------------------------------------

def choose_model_and_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        model = filedialog.askopenfilename(
            title="Select ArcGUI ONNX model",
            filetypes=[
                ("ONNX models", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
        if not model:
            root.destroy()
            raise SystemExit("No model was selected.")

        image_folder = filedialog.askdirectory(
            title="Select folder containing input images"
        )
        root.destroy()

        if not image_folder:
            raise SystemExit("No image folder was selected.")

        return Path(model).expanduser().resolve(), Path(image_folder).expanduser().resolve()

    except SystemExit:
        raise
    except Exception as exc:
        print(f"File picker unavailable ({exc}).")
        model = input("Full ONNX model path: ").strip().strip('"')
        folder = input("Input image folder: ").strip().strip('"')
        if not model or not folder:
            raise SystemExit("Model and image folder are required.")
        return Path(model).expanduser().resolve(), Path(folder).expanduser().resolve()


# ------------------------------------------------------------
# MODEL PROFILE
# ------------------------------------------------------------

def detect_model_profile(model_path):
    path = Path(model_path)
    stem = path.stem.lower()

    if path.suffix.lower() != ".onnx":
        raise ValueError("The selected model must be an .onnx file.")

    if "segformer" in stem:
        architecture = "SegFormer MiT-B5"
        family = "segformer"
    elif "unet" in stem and ("resnet101" in stem or "r101" in stem):
        architecture = "U-Net ResNet101"
        family = "unet_resnet101"
    else:
        raise ValueError(
            "Unsupported model filename.\n"
            "Use a SegFormer filename containing 'segformer', or a U-Net "
            "ResNet101 filename containing 'unet' and 'resnet101'/'r101'."
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


MODEL_PATH, IMAGE_FOLDER = choose_model_and_folder()

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not IMAGE_FOLDER.is_dir():
    raise NotADirectoryError(f"Input image folder not found:\n{IMAGE_FOLDER}")

PROFILE = detect_model_profile(MODEL_PATH)


# ------------------------------------------------------------
# DLL DISCOVERY BEFORE ONNX RUNTIME IMPORT
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

if PROFILE["target"] in ("cuda", "auto"):
    nvidia_root = ENV / "Lib" / "site-packages" / "nvidia"
    for directory in glob.glob(str(nvidia_root / "*" / "bin")):
        add_dll_directory(directory)


try:
    import onnxruntime as ort
except Exception as exc:
    raise RuntimeError(
        "Could not import onnxruntime. For CUDA models, use onnxruntime-gpu.\n"
        f"Original error: {exc}"
    ) from exc


if PROFILE["target"] in ("cuda", "auto"):
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    except Exception:
        pass


# ------------------------------------------------------------
# PROVIDERS + SESSION
# ------------------------------------------------------------

available = ort.get_available_providers()

if PROFILE["target"] == "cuda":
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "The selected model requires CUDA, but CUDAExecutionProvider is not available."
        )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    effective_target = "cuda"

elif PROFILE["target"] == "cpu":
    providers = ["CPUExecutionProvider"]
    effective_target = "cpu"

else:
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        effective_target = "cuda"
    else:
        providers = ["CPUExecutionProvider"]
        effective_target = "cpu"

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=providers,
)

if effective_target == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
    raise RuntimeError("CUDA was requested but the ONNX session did not activate CUDAExecutionProvider.")

inputs = session.get_inputs()
outputs = session.get_outputs()

if len(inputs) != 1 or len(outputs) != 1:
    raise RuntimeError(
        f"Expected one model input and one output, found {len(inputs)} input(s) and {len(outputs)} output(s)."
    )

input_meta = inputs[0]
output_meta = outputs[0]

if input_meta.type != "tensor(float)":
    raise RuntimeError(f"Expected FP32 model input, got {input_meta.type}.")

shape = input_meta.shape
if len(shape) != 4:
    raise RuntimeError(f"Expected NCHW model input, got {shape}.")
if isinstance(shape[0], int) and shape[0] != 1:
    raise RuntimeError(f"This simple example uses batch=1, but the model requires batch={shape[0]}.")
if isinstance(shape[1], int) and shape[1] != 3:
    raise RuntimeError(f"Expected 3 RGB channels, got {shape}.")
if isinstance(shape[2], int) and shape[2] != MODEL_INPUT_SIZE:
    raise RuntimeError(f"Expected input height 1024, got {shape[2]}.")
if isinstance(shape[3], int) and shape[3] != MODEL_INPUT_SIZE:
    raise RuntimeError(f"Expected input width 1024, got {shape[3]}.")


# ------------------------------------------------------------
# PREPROCESS / OUTPUT
# ------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def preprocess(image):
    image = image.convert("RGB")
    resized = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = array.transpose(2, 0, 1)
    return array[None, ...].astype(np.float32)


def output_to_probability(raw_output):
    pred = np.asarray(raw_output)

    if pred.ndim == 4:
        if pred.shape[1] == 1:
            pred = pred[0, 0]
        elif pred.shape[-1] == 1:
            pred = pred[0, :, :, 0]
        else:
            raise RuntimeError(f"Unexpected multi-channel output shape: {pred.shape}")
    elif pred.ndim == 3:
        if pred.shape[0] != 1:
            raise RuntimeError(f"Unexpected output shape: {pred.shape}")
        pred = pred[0]
    elif pred.ndim == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected output shape: {pred.shape}")

    pred = np.asarray(pred, dtype=np.float32)

    if not np.all(np.isfinite(pred)):
        raise RuntimeError("Model output contains NaN or infinite values.")

    pred = np.clip(pred, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-pred))


# ------------------------------------------------------------
# INPUT / OUTPUT FOLDERS
# ------------------------------------------------------------

images = sorted(
    p for p in IMAGE_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
)

if not images:
    raise RuntimeError(
        f"No supported images were found in:\n{IMAGE_FOLDER}\n\n"
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )

OUTPUT_FOLDER = IMAGE_FOLDER / f"masks_{MODEL_PATH.stem}"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 76)
print("ARCGUI SIMPLE FOLDER INFERENCE")
print("=" * 76)
print("Model          :", MODEL_PATH.name)
print("Architecture   :", PROFILE["architecture"])
print("Execution      :", effective_target.upper())
print("Input folder   :", IMAGE_FOLDER)
print("Output folder  :", OUTPUT_FOLDER)
print("Images         :", len(images))
print("Threshold      :", THRESHOLD)
print("=" * 76)


# ------------------------------------------------------------
# RUN ONE IMAGE AT A TIME
# ------------------------------------------------------------

for index, image_path in enumerate(images, start=1):
    try:
        with Image.open(image_path) as image:
            original_size = image.size  # (width, height)
            input_tensor = preprocess(image)

        raw = session.run(
            [output_meta.name],
            {input_meta.name: input_tensor},
        )[0]

        probability = output_to_probability(raw)
        mask = (probability > THRESHOLD).astype(np.uint8) * 255

        mask_image = Image.fromarray(mask, mode="L")
        if mask_image.size != original_size:
            mask_image = mask_image.resize(original_size, NEAREST)

        output_path = OUTPUT_FOLDER / f"{image_path.stem}_mask.png"
        mask_image.save(output_path)

        print(f"[{index}/{len(images)}] {image_path.name} -> {output_path.name}")

    except Exception as exc:
        print(f"[{index}/{len(images)}] FAILED: {image_path.name} -> {exc}")

print("\nDone.")
print("Masks saved to:")
print(OUTPUT_FOLDER)
