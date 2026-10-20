# ArcGUI `toolbox_files/`

This folder contains the ArcGIS Pro implementation of **ArcGUI Road Segmentation**.

```text
toolbox_files/
├── README.md
├── roadsegmentation_tool.pyt
└── roadsegmentation_python.py
```

Both Python files must remain together.

---

## `roadsegmentation_tool.pyt`

This is the ArcGIS Pro Python toolbox definition.

It is responsible for:

- displaying **ArcGUI Road Segmentation** in ArcGIS Pro;
- defining the GUI parameters;
- validating user selections before execution;
- showing informative warnings and errors;
- loading the backend from the same folder;
- launching production or debug processing.

The current GUI exposes:

```text
Extent Layer(s)
Input Raster Layer Name
Output Folder
Road Segmentation ONNX Model
Tile Overlap (%)
Optimized Large-Area Export + Python Cropping
Prediction Threshold
ONNX Batch Size
Debug Footprint Preview Only
Reuse Matching Exported Imagery
```

The current deployment uses the **exact selected polygon**. There is no user-configurable extent buffer.

---

## `roadsegmentation_python.py`

This is the processing backend.

It contains the main logic for:

- polygon and CRS validation;
- imagery-layer validation;
- direct 1024-tile planning;
- optimized large-area block planning;
- skipping areas outside the requested polygon;
- ArcGIS imagery export;
- reusable export-cache creation and validation;
- ONNX Runtime provider selection;
- SegFormer / U-Net model validation;
- RGB + ImageNet preprocessing;
- inference;
- probability averaging;
- thresholding;
- georeferencing;
- clipping;
- final raster output;
- runtime safeguards and error handling.

The `.pyt` intentionally loads the backend located beside it so ArcGIS Pro does not accidentally reuse another copy of `roadsegmentation_python.py` from a previous deployment.

---

# Supported Model Families

Primary recommended models:

```text
segformer_cuda_fp16.onnx
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

Additional distributed variants may include:

```text
segformer_cpu_fp32_optimized.onnx
segformer_last_cuda_fp16.onnx
last_unet_resnet101_norm_1024_cuda_fp16.onnx
old_milestone_unet_resnet101_norm_1024.onnx
```

Model recognition is family based:

```text
filename contains "segformer"
    -> SegFormer MiT-B5

filename contains "unet" and "resnet101" or "r101"
    -> U-Net ResNet101
```

Provider behavior:

```text
filename contains "cpu"
    -> CPU only

filename contains "cuda" or "fp16"
    -> CUDA required

otherwise
    -> CUDA preferred, CPU fallback
```

ArcGUI also validates the real ONNX input/output contract after the file is loaded.

---

# Fixed Deployment Contract

```text
Input:
    N x 3 x 1024 x 1024
    RGB
    float32

Normalization:
    ImageNet mean/std

Output:
    single-channel road logits

Probability:
    sigmoid(logits)

Final threshold:
    default 0.50

Export resolution:
    2 m/pixel
```

The active ArcGIS map must use a **projected coordinate system in metres**.

---

# Processing Modes

## Direct 1024-Tile Export

When optimized mode is unchecked, ArcGUI plans 1024 × 1024 windows directly.

Completely outside windows are rejected before export.

Any tile intersecting the target polygon is retained so road context is not lost along the boundary.

At 50% overlap, the stride is 512 pixels.

## Optimized Large-Area Export

Recommended for large regions.

ArcGUI:

1. plans balanced non-overlapping larger export blocks;
2. rejects large blocks that do not intersect the target;
3. exports the retained imagery with ArcGIS;
4. creates the 1024 × 1024 overlapping model windows locally;
5. skips model windows outside the requested polygon;
6. runs ONNX inference;
7. averages overlapping model probabilities.

---

# Reusable Imagery Cache

The reusable cache prevents re-downloading the same imagery for every model comparison.

A matching cache depends on export-relevant settings such as:

- selected polygon geometry;
- imagery source;
- coordinate system;
- overlap;
- optimized/direct mode;
- fixed deployment settings.

These can change without forcing a new imagery export:

```text
ONNX model
Prediction threshold
ONNX batch size
```

Therefore imagery exported for SegFormer can be reused for the U-Net model when the spatial/export configuration is unchanged.

ArcGUI validates cached TIFFs before reuse and rejects incomplete, modified, incompatible, or incorrectly georeferenced caches.

---

# Debug Footprint Preview

Enable:

```text
Debug Footprint Preview Only
```

to inspect the spatial processing plan without exporting imagery or running ONNX inference.

---

# Important Validation / Failure Behavior

The hardened backend checks common deployment failures before they can silently create misleading output, including:

- missing/duplicate/non-polygon extents;
- unknown extent CRS;
- invalid geometry;
- missing active map;
- missing or duplicated imagery-layer name;
- incorrect map units;
- invalid overlap/threshold/batch size;
- missing or empty ONNX model;
- unsupported model family;
- unavailable CUDA provider;
- incompatible ONNX I/O;
- NaN/Inf model output;
- low disk space;
- failed or black imagery export;
- invalid raster georeferencing;
- incomplete cache;
- zero valid inference coverage.

Review the ArcGIS geoprocessing messages if a job fails.

---

# Installation

In ArcGIS Pro:

```text
Catalog
→ Toolboxes
→ Add Toolbox
```

Choose:

```text
roadsegmentation_tool.pyt
```

Do not move `roadsegmentation_python.py` away from the `.pyt`.

---

# Recommended Production Settings

SegFormer:

```text
Model: segformer_cuda_fp16.onnx
Tile Overlap: 50%
Optimized Large-Area Export: Enabled
Prediction Threshold: 0.50
ONNX Batch Size: 2
Reuse Matching Exported Imagery: Enabled
```

Faster U-Net alternative:

```text
Model: best_unet_resnet101_norm_1024_cuda_fp16.onnx
ONNX Batch Size: 4
```

Reduce the batch size if GPU memory is limited.

---

# Model Weights

The model weights are not stored in this repository.

Contact **Laurie Vidal** or **Dodick Gasser**.

The weights are located in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal under the **`final`** folder.

---

# Full Guide

For the complete illustrated deployment guide, use:

```text
../reports/technical_guide.pdf
```
