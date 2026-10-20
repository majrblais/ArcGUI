# ArcGUI Utilities — User Guide

This folder contains two small standalone utilities:

```text
utils/
├── README.md
├── check.py
└── python_inference_script.py
```

They are intended to make model/environment testing simple before using the full ArcGIS toolbox.

---

# 1. `check.py` — verify a model and execution environment

Use `check.py` first on a new computer or Python environment.

## Launch

From the repository root:

```bash
python utils/check.py
```

or from inside the `utils` folder:

```bash
python check.py
```

A file-selection window opens.

Select the ONNX model to test.

Recommended CUDA choices:

```text
segformer_cuda_fp16.onnx
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

## What it checks

The script checks that:

- the selected file exists and is not empty;
- the filename identifies a supported SegFormer or U-Net ResNet101 model family;
- ONNX Runtime imports correctly;
- the requested CPU/CUDA provider is available;
- the ONNX session loads;
- the input/output model structure is compatible;
- the model accepts RGB `1024 × 1024` float32 input;
- dummy inference actually runs;
- the output does not contain NaN or infinite values.

For a CUDA model, success should end with:

```text
SUCCESS - MODEL LOADED AND INFERENCE RAN WITH CUDA
```

For a CPU model:

```text
SUCCESS - MODEL LOADED AND INFERENCE RAN WITH CPU
```

If a CUDA model cannot activate `CUDAExecutionProvider`, fix the GPU environment before relying on ArcGUI.

---

# 2. `python_inference_script.py` — simple folder inference

This is a deliberately simple example that applies the selected ONNX model to a folder of ordinary images.

It does **not** use ArcGIS, georeferencing, large-area export, overlap mosaicking, or GIS clipping.

Use it to quickly verify that a model produces masks on normal image files.

## Launch

From the repository root:

```bash
python utils/python_inference_script.py
```

or from inside `utils`:

```bash
python python_inference_script.py
```

Two dialogs appear:

1. select the ONNX model;
2. select the folder containing input images.

## Supported image types

```text
.png
.jpg
.jpeg
.tif
.tiff
.bmp
```

## What the script does

For every image:

```text
image
→ RGB
→ resize to 1024 × 1024
→ ImageNet normalization
→ ONNX inference
→ sigmoid
→ threshold 0.50
→ binary road mask
```

The mask is resized back to the original image dimensions.

## Output folder

If the input folder is:

```text
images/
```

and the selected model is:

```text
segformer_cuda_fp16.onnx
```

then the script creates:

```text
images/
├── image_01.tif
├── image_02.tif
└── masks_segformer_cuda_fp16/
    ├── image_01_mask.png
    └── image_02_mask.png
```

Mask coding:

```text
0   = background
255 = road
```

## Important limitation

This simple utility is for demonstration/testing only.

It should not be used as a substitute for the main ArcGUI toolbox when you need:

- georeferenced outputs;
- ArcGIS World Imagery export;
- exact polygon processing;
- overlapping 1024-window reconstruction;
- reusable GIS export caching;
- final GIS-ready raster outputs.

---

# 3. Supported model naming

Both utilities recognize:

```text
filename contains "segformer"
    -> SegFormer MiT-B5

filename contains "unet" and "resnet101" or "r101"
    -> U-Net ResNet101
```

Execution target:

```text
contains "cpu"
    -> CPU only

contains "cuda" or "fp16"
    -> CUDA required

otherwise
    -> CUDA preferred, CPU fallback
```

Keep the distributed model filenames when possible.

---

# 4. Recommended order of use

On a new computer:

```text
1. configure the Python environment
2. run check.py
3. optionally run python_inference_script.py on a few sample images
4. open ArcGIS Pro
5. use the main toolbox
```
