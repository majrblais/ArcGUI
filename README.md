# 🛣️ ArcGIS Road Segmentation Toolkit

Deep learning–based road extraction workflow for ArcGIS Pro using a U-Net ResNet101 model exported to ONNX.

---

# Overview

This repository contains the deployment package developed for automated road extraction within ArcGIS Pro. The project includes:

- ArcGIS Pro GUI toolbox
- ONNX road segmentation model
- Standalone Python inference script
- CUDA / ONNX Runtime GPU acceleration
- Multi-extent processing support
- Technical deployment guide
- Technical report with methodology and results

The system was developed primarily for forestry, agricultural and rural road extraction using high-resolution RGB imagery.

---

# Repository Structure

```text
ArcGUI/
│
├── README.md
├── requirements.txt
├── check.py
├── python_inference_script.py
│
├── reports/
│   ├── technical_guide.pdf
│   └── technical_report.pdf
│
└── toolbox_files/
    ├── roadsegmentation_tool.pyt
    └── roadsegmentation_python.py
```

## ONNX Model

The trained ONNX model is distributed separately because of GitHub file size limitations.

Download the model from the provided OneDrive link and place it in a convenient location on your computer.

Example:

```text
E:\Models\best_unet_resnet101_norm_1024.onnx
```

The toolbox and standalone script both allow the model path to be selected directly.

---

# Model Information

| Component | Value |
|------------|------------|
| Architecture | U-Net |
| Encoder | ResNet101 |
| Framework | PyTorch |
| Deployment | ONNX Runtime |
| Input Size | 1024 × 1024 |
| Input Type | RGB |
| Output | Binary Road Mask |
| Cell Size | 2 m/pixel |

---

# Documentation

The repository includes two documents inside the `reports` folder.

## Technical Guide

```text
reports/technical_guide.pdf
```

Contains:

- Environment setup
- ArcGIS installation instructions
- Toolbox deployment
- GUI usage
- Standalone inference script
- Troubleshooting

## Technical Report

```text
reports/technical_report.pdf
```

Contains:

- Dataset description
- Model training procedure
- Performance evaluation
- Threshold analysis
- Deployment benchmarks
- Explainability results
- Conclusions and future work

---

# Environment Setup

## 1. Clone ArcGIS Environment

Inside ArcGIS Pro:

```text
Package Manager → Clone Environment
```

Recommended name:

```text
torch_roads
```

---

## 2. Activate Environment

Open Anaconda Prompt:

```bash
conda activate torch_roads
```

---

## 3. Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

---

# Verify CUDA / ONNX Runtime

Run:

```bash
python check.py
```

Expected result:

```text
CUDAExecutionProvider
```

If CUDA is active, ONNX Runtime is correctly using GPU acceleration.

---

# ArcGIS Toolbox Installation

The ArcGIS toolbox files are located inside:

```text
toolbox_files/
```

Both files must remain together in the same folder:

```text
toolbox_files/
├── roadsegmentation_tool.pyt
└── roadsegmentation_python.py
```

Inside ArcGIS Pro:

```text
Catalog → Add Toolbox
```

Select:

```text
roadsegmentation_tool.pyt
```

---

# GUI Workflow

The ArcGIS implementation automatically performs:

1. Export Training Data for Deep Learning
2. Image preprocessing
3. ONNX inference
4. Georeferenced raster creation
5. Raster mosaicking
6. Extent clipping
7. Output generation

The implementation supports:

- Single extent processing
- Multiple extent processing
- Buffered extent extraction
- CUDA acceleration
- Automatic mosaicking

---

# Main GUI Parameters

## Extent Layer

Polygon layer defining the processing area.

Supports:

- Single polygon
- Multiple polygons
- Multiple cells or management regions

## Input Raster

Typically:

```text
World Imagery
```

Wayback imagery may also be used.

## Output Folder

Stores:

- Exported tiles
- Temporary rasters
- Intermediate mosaics
- Final predictions

A shared folder named:

```text
final_predictions
```

is automatically created.

## Model File

Path to:

```text
best_unet_resnet101_norm_1024.onnx
```

## Threshold

Recommended:

```text
0.5
```

Typical range:

```text
0.4 - 0.6
```

## ONNX Batch Size

Typical values:

| Hardware | Batch Size |
|-----------|-----------|
| CPU | 1 |
| Small GPU | 2 |
| RTX 3060 | 4–8 |
| Larger GPUs | 8–32 |

## Stride

Controls overlap between tiles.

Smaller stride:

- More overlap
- Better edge consistency
- Longer processing time

Larger stride:

- Faster processing
- Less overlap

## Edge Handling

Buffered extraction can be enabled to reduce border artifacts.

Recommended:

```text
100 pixels
```

---

# Fixed Parameters

The following parameters are intentionally fixed to match the trained model:

| Parameter | Value |
|------------|------------|
| Tile Size | 1024 × 1024 |
| Cell Size | 2 |
| Model Input Size | 1024 × 1024 |

---

# Standalone Inference Script

A lightweight ONNX inference implementation is provided:

```text
python_inference_script.py
```

This version:

- Does not require ArcGIS
- Supports folder-based prediction
- Uses ONNX Runtime directly
- Saves predicted masks automatically

Suitable for:

- Testing
- Batch inference
- Integration into other workflows

---

# Typical Output Structure

```text
output_folder/
│
├── extent_1/
├── extent_2/
├── extent_3/
│
└── final_predictions/
    ├── cell_001_roads_prediction_clean.tif
    ├── cell_002_roads_prediction_clean.tif
    └── cell_003_roads_prediction_clean.tif
```

---

# Troubleshooting

## CUDA Not Detected

Verify:

```bash
python check.py
```

Check:

- NVIDIA drivers
- CUDA installation
- ONNX Runtime installation
- Active Conda environment

## Error 126

Usually caused by:

- Missing CUDA DLLs
- Missing cuDNN libraries
- Version incompatibilities

## Out of Memory

Reduce:

```text
ONNX Batch Size
```

Recommended values:

```text
1 or 2
```

## ArcGIS Export Issues

Possible causes:

- Invalid extent geometry
- Internet connectivity issues
- World Imagery availability
- Insufficient disk space

---

# Performance Notes

According to the deployment benchmarks included in the technical report:

- Approximately 196 km² can be processed in under one minute on an RTX 3060 system.
- ONNX inference itself is only a small portion of the total runtime.
- The largest bottleneck is generally imagery extraction and GIS processing.

Performance can be improved using:

- Faster GPUs
- SSD storage
- Larger batch sizes
- Local imagery instead of remote services

---

# Citation

If this repository is used in a project or publication, please also reference the accompanying technical report and technical guide located in the `reports` folder.
