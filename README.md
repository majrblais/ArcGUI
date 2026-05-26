# 🛣️ ArcGIS Road Segmentation Toolkit

Deep learning–based road extraction workflow for ArcGIS Pro using ONNX Runtime and U-Net segmentation.

---

# 📌 Overview

This repository provides a practical deployment pipeline for large-scale road segmentation directly inside **ArcGIS Pro** using a trained deep learning model exported to **ONNX** format.

The system was designed primarily for:

- Forest road extraction
- Rural road monitoring
- GIS-assisted mapping workflows
- Large-scale segmentation inference
- Multi-region processing inside ArcGIS Pro

The implementation includes:

- ✅ ArcGIS Pro GUI toolbox
- ✅ ONNX GPU inference pipeline
- ✅ Automatic tile export and mosaicking
- ✅ Multi-extent support
- ✅ Standalone Python inference script
- ✅ CUDA / ONNX Runtime acceleration
- ✅ Edge artifact reduction
- ✅ Automatic georeferenced output generation

---

# 🧠 Model Information

The final deployed model is based on:

| Component | Value |
|---|---|
| Architecture | U-Net |
| Encoder | ResNet101 |
| Framework | PyTorch |
| Deployment Format | ONNX |
| Input Size | 1024 × 1024 |
| Input Type | RGB imagery |
| Cell Size | 2 map units |
| Output | Binary road mask |

The model was trained using aerial and satellite RGB imagery and optimized for road extraction tasks.

---

# 📂 Repository Structure

```text
.
├── RoadSegmentationTool_FIXED_PARAMS_CUDA_CLEAN.pyt
├── road_tool_runner_FIXED_PARAMS_CUDA_CLEAN.py
├── best_unet_resnet101_norm_1024.onnx
├── standalone_inference.py
├── check.py
├── README.md
```

---

# ⚙️ Requirements

## Software

- ArcGIS Pro
- Python / Conda
- NVIDIA GPU (recommended)
- CUDA-compatible drivers

Although CPU inference is supported, GPU acceleration is highly recommended for practical deployment speed.

---

# 🖥️ Environment Setup

## 1. Clone ArcGIS Environment

Inside **ArcGIS Pro**:

```text
Package Manager → Clone Environment
```

Recommended environment name:

```bash
torch_roads
```

---

## 2. Activate Environment

Open the **Anaconda Prompt**:

```bash
conda activate torch_roads
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision torchaudio segmentation-models-pytorch \
albumentations opencv-python pillow tqdm onnxruntime-gpu \
nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

---

# ✅ Verify GPU Acceleration

Run:

```bash
python check.py
```

Expected output:

```text
CUDAExecutionProvider active
```

If ONNX Runtime falls back to CPU execution:

- verify CUDA installation
- verify NVIDIA drivers
- verify DLL paths
- reinstall CUDA-related dependencies

---

# 🧩 ArcGIS Toolbox Installation

Inside ArcGIS Pro:

```text
Catalog → Add Toolbox
```

Select:

```text
RoadSegmentationTool_FIXED_PARAMS_CUDA_CLEAN.pyt
```

⚠️ The `.pyt` and `.py` files must remain in the same folder.

---

# 🚀 GUI Workflow

The toolbox automatically:

1. Exports imagery tiles
2. Applies preprocessing
3. Performs ONNX inference
4. Reconstructs predictions
5. Generates georeferenced rasters
6. Creates final mosaics
7. Clips outputs to the original extent

---

# 🗺️ Main GUI Parameters

## Extent Layer

Defines the region(s) processed by the model.

Supports:

- single polygon
- multiple polygons
- multiple cells / regions

Each extent is processed independently.

---

## Input Raster Layer

Usually:

```text
World Imagery
```

Wayback imagery can also be used for temporal analysis.

---

## Output Folder

Stores:

- exported tiles
- temporary rasters
- mosaics
- final predictions

Final clean outputs are automatically copied into:

```text
final_predictions/
```

---

## Model File

Select the provided ONNX model:

```text
best_unet_resnet101_norm_1024.onnx
```

---

## Threshold

Controls binary conversion.

Typical values:

| Threshold | Effect |
|---|---|
| 0.4 | Higher recall |
| 0.5 | Balanced |
| 0.6 | Cleaner predictions |

Default:

```text
0.5
```

---

## Batch Size

Controls the number of tiles processed simultaneously.

Recommended:

| GPU Memory | Suggested Batch Size |
|---|---|
| Low VRAM | 1–2 |
| Moderate VRAM | 4 |
| High VRAM | 8–32 |

---

## Stride

Controls overlap between exported tiles.

| Configuration | Effect |
|---|---|
| No overlap | Faster |
| Overlap | Better edge consistency |

---

## Edge Handling

The implementation supports buffered extents to reduce edge artifacts.

Recommended:

```text
Buffer = 100 pixels
```

---

# 🔒 Fixed Parameters

These parameters are intentionally fixed to match training conditions:

| Parameter | Value |
|---|---|
| Tile Size | 1024 × 1024 |
| Cell Size | 2 |
| Model Input Size | 1024 × 1024 |

Changing these values may produce invalid predictions.

---

# 📤 Output Structure

For each processed extent:

```text
extent_name/
├── exported_tiles/
├── predictions/
├── mosaics/
└── extent_name_roads_prediction_clean.tif
```

All final predictions are additionally copied into:

```text
final_predictions/
```

---

# 🧪 Standalone Inference Script

A lightweight standalone ONNX inference script is also included.

Features:

- folder-based prediction
- PNG/JPG/TIF support
- automatic normalization
- automatic resizing
- GPU inference
- binary mask generation

Example:

```bash
python standalone_inference.py
```

---

# 📈 Inference Pipeline

The standalone pipeline performs:

```text
Image Loading
    ↓
Resize to 1024×1024
    ↓
ImageNet Normalization
    ↓
ONNX Inference
    ↓
Sigmoid Activation
    ↓
Thresholding
    ↓
Binary Mask Generation
    ↓
Resize to Original Size
    ↓
Save Prediction
```

---

# 🧰 Troubleshooting

## ONNX Runtime Falling Back to CPU

Possible causes:

- incompatible CUDA version
- missing DLLs
- incorrect environment activation
- incompatible ONNX Runtime version

Verify:

```bash
python check.py
```

---

## Error 126

Typical error:

```text
LoadLibrary failed with error 126
```

Usually related to:

- missing CUDA DLLs
- cuDNN mismatch
- incorrect GPU drivers

Reinstall CUDA dependencies and verify environment paths.

---

## CUDA Out of Memory

Reduce:

```text
Batch Size
```

Recommended low-memory values:

```text
1 or 2
```

---

## ArcGIS Export Failures

Possible causes:

- invalid extents
- unavailable imagery
- internet connectivity
- insufficient disk space

---

## Poor Prediction Quality

Verify that imagery:

- is RGB
- is similar to training imagery
- has approximately 2 m resolution
- contains visible-spectrum information

Prediction quality may also be affected by annotation inconsistencies.

---

# ⚡ Performance Notes

The largest deployment bottleneck is generally:

```text
Imagery export from ArcGIS
```

ONNX GPU inference itself is typically very fast.

Performance can be improved using:

- faster GPUs
- SSD storage
- larger batch sizes
- buffered extraction

---

# 📚 Technical Guide

A complete technical deployment guide is included in the uploaded document:

fileciteturn0file0L1-L13

The guide contains:

- environment setup
- ArcGIS integration
- ONNX deployment
- GUI usage
- troubleshooting
- deployment recommendations

