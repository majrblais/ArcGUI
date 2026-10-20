# 🛣️ ArcGUI Road Segmentation

ArcGUI is an ArcGIS Pro deployment package for automated forestry, agricultural, rural, and dirt-road extraction from high-resolution RGB imagery.

The repository contains the ArcGIS Pro toolbox, ONNX deployment/testing utilities, environment requirements, and project documentation.

> **Start here for a quick deployment.**  
> For the complete workflow, screenshots, troubleshooting, and implementation details, use the technical deployment guide in [`reports/`](reports/).

---

## Repository

GitHub: https://github.com/majrblais/ArcGUI

Current repository layout:

```text
ArcGUI/
│
├── README.md
├── MODEL_WEIGHTS.md
├── requirements.txt
├── check.py
├── python_inference_script.py
│
├── reports/
│   ├── README.md
│   ├── technical_guide.pdf
│   └── technical_report.pdf
│
└── toolbox_files/
    ├── README.md
    ├── roadsegmentation_tool.pyt
    └── roadsegmentation_python.py
```

The ONNX model weights are distributed separately and are **not stored in the GitHub repository**.

---

# Recommended Models

ArcGUI recognizes the current SegFormer MiT-B5 and U-Net ResNet101 ONNX variants.

For normal deployment, the two recommended CUDA models are:

| Recommended model | Use |
|---|---|
| `segformer_cuda_fp16.onnx` | **Primary final model.** Best overall segmentation quality and consistency. |
| `best_unet_resnet101_norm_1024_cuda_fp16.onnx` | **Faster CUDA alternative.** Similar overall road-mapping results with lower inference cost. |

The project evaluation found that the successfully trained models generally produced comparable road-segmentation results. SegFormer was retained as the primary final model because of its small but consistent performance advantage, while U-Net remains useful when faster inference is preferred.

Additional models may also be provided for compatibility, comparison, reproducibility, or CPU-only use:

```text
segformer_cpu_fp32_optimized.onnx
segformer_last_cuda_fp16.onnx
last_unet_resnet101_norm_1024_cuda_fp16.onnx
old_milestone_unet_resnet101_norm_1024.onnx
```

These additional files are mainly courtesy/alternative deployment options rather than the recommended default models.

## Obtaining the model weights

To obtain the model weights, contact:

- **Laurie Vidal**
- **Dodick Gasser**

The project model files are stored in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal, under the **`final`** folder.

The models can be stored anywhere on the deployment computer because ArcGUI lets the user browse directly to the `.onnx` file.

---

# What Each File Does

## `toolbox_files/roadsegmentation_tool.pyt`

This is the ArcGIS Pro Python toolbox interface.

It defines the **ArcGUI Road Segmentation** geoprocessing tool, its parameters, validation messages, warnings, and user-facing behavior.

Keep this file beside `roadsegmentation_python.py`.

More information: [`toolbox_files/README.md`](toolbox_files/README.md)

## `toolbox_files/roadsegmentation_python.py`

This is the main processing backend.

It handles:

- extent validation;
- imagery export;
- direct and optimized tiling;
- reusable imagery caching;
- ONNX Runtime inference;
- overlap averaging;
- thresholding;
- georeferencing;
- final clipping;
- output creation;
- runtime validation and failure handling.

The `.pyt` toolbox explicitly loads the backend located beside it, so the two files should remain together.

## `check.py`

Small model/environment verification utility.

Run:

```bash
python check.py
```

A file-selection window opens. Select one of the ONNX model files and the script checks whether the model can be loaded and executed with the expected ONNX Runtime provider.

For CUDA models, the check confirms that `CUDAExecutionProvider` is actually active and performs a real dummy inference.

This is the recommended first test after setting up a new computer.

## `python_inference_script.py`

Simple standalone inference example.

Run:

```bash
python python_inference_script.py
```

The script asks the user to:

1. select an ONNX model;
2. select a folder of input images.

It then generates binary road masks from the images.

This script is intentionally simple and is useful for confirming that the model works outside the full ArcGIS workflow.

It is **not a replacement for ArcGUI** when georeferenced ArcGIS processing, large-area export, overlap reconstruction, or GIS outputs are required.

## `requirements.txt`

Python dependencies used by the deployment environment.

## `reports/`

Contains the complete documentation.

See [`reports/README.md`](reports/README.md).

---

# Environment Setup

## 1. Install ArcGIS Pro

The ArcGIS toolbox requires ArcGIS Pro.

Production imagery export also requires access to the ArcGIS **Image Analyst** functionality used by the workflow.

## 2. Clone the ArcGIS Pro Python environment

In ArcGIS Pro:

```text
Project / Settings
→ Package Manager
→ Clone Environment
```

A practical environment name is:

```text
torch_roads
```

Do not modify the default ArcGIS Pro Python environment directly.

## 3. Activate the cloned environment

From an Anaconda Prompt or ArcGIS Pro Python Command Prompt:

```bash
conda activate torch_roads
```

## 4. Install the project dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

For GPU deployment, ONNX Runtime must expose:

```text
CUDAExecutionProvider
```

For CPU-only deployment, the environment must at minimum expose:

```text
CPUExecutionProvider
```

CUDA deployments additionally require a compatible NVIDIA driver / CUDA / cuDNN / ONNX Runtime configuration.

## 5. Verify the selected model

Run:

```bash
python check.py
```

Select the model file when prompted.

For the recommended GPU deployment, test either:

```text
segformer_cuda_fp16.onnx
```

or:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

A CUDA model should only be considered ready when the script confirms that the model loaded and inference ran using `CUDAExecutionProvider`.

---

# Install ArcGUI in ArcGIS Pro

The two main toolbox files must remain together:

```text
toolbox_files/
├── roadsegmentation_tool.pyt
└── roadsegmentation_python.py
```

In ArcGIS Pro:

```text
Catalog
→ Toolboxes
→ Add Toolbox
```

Select:

```text
toolbox_files/roadsegmentation_tool.pyt
```

The tool appears as:

```text
ArcGUI Road Segmentation
```

---

# Before Running the Tool

Prepare the ArcGIS Pro project as follows:

1. Add the imagery layer to the active map.
2. Normally name the imagery layer `World Imagery`, or enter its exact layer name in the GUI.
3. Add the polygon extent layer(s) to process.
4. Make sure the extent layer has a **defined coordinate system**.
5. Use a **projected map coordinate system in metres**.
6. Select an output folder with sufficient free disk space.
7. Select the desired ONNX model.

The current deployment assumes:

| Parameter | Value |
|---|---:|
| Model input | `1024 × 1024` pixels |
| Input channels | RGB |
| Export resolution | `2 m/pixel` |
| Normalization | ImageNet |
| Default prediction threshold | `0.50` |
| Default overlap | `50%` |

The current version uses the **exact selected polygon** as the target extent. There is no user-configurable extent buffer.

---

# ArcGUI Parameters

## Extent Layer(s)

One or more polygon layers defining the processing area.

Selections and definition queries are respected.

If multiple polygons are selected inside one layer, they are treated as one processing job for that layer entry.

## Input Raster Layer Name

Name of the imagery layer in the active map.

Typical value:

```text
World Imagery
```

The name must match one unique usable imagery/raster layer in the active map.

## Output Folder

Root folder where ArcGUI stores reusable imagery, model-specific intermediate results, and final predictions.

A short local path with sufficient free disk space is recommended.

## Road Segmentation ONNX Model

Browse to one of the supported ONNX files.

Recommended:

```text
segformer_cuda_fp16.onnx
```

or:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

ArcGUI recognizes the model family from the filename:

```text
contains "segformer"
    -> SegFormer MiT-B5

contains "unet" and "resnet101" or "r101"
    -> U-Net ResNet101
```

Execution mode is also inferred from the filename:

```text
contains "cpu"
    -> CPU only

contains "cuda" or "fp16"
    -> CUDA required

otherwise
    -> CUDA preferred, CPU fallback
```

The ONNX model itself is still validated after loading, so a filename alone is not sufficient to bypass model compatibility checks.

## Tile Overlap (%)

Controls overlap between neighbouring 1024 × 1024 model windows.

Default:

```text
50%
```

At 50% overlap, the model-window stride is 512 pixels.

Higher overlap can improve edge consistency but increases processing time.

## Optimized Large-Area Export + Python Cropping

**Recommended for large areas.**

When enabled:

1. ArcGIS exports larger balanced imagery blocks.
2. Blocks completely outside the selected polygon are skipped.
3. Python generates the overlapping 1024 × 1024 model windows from the retained imagery.
4. Only model windows intersecting the target are inferred.

When disabled, ArcGIS exports the retained 1024 × 1024 model tiles directly.

## Prediction Threshold

Default:

```text
0.50
```

The model first produces probabilities. The threshold is applied after overlapping predictions have been averaged.

## ONNX Batch Size

Number of 1024 × 1024 windows processed in one ONNX Runtime call.

Recommended starting values:

| Model | Starting batch |
|---|---:|
| SegFormer CUDA | 2 |
| U-Net CUDA | 4 |
| SegFormer CPU | 1 |
| U-Net CPU / automatic fallback | 1 |

If GPU memory errors occur, reduce the batch size.

## Debug Footprint Preview Only

Runs only the spatial planner.

It does **not**:

- download imagery;
- load the ONNX model;
- run inference.

Use it to inspect which large blocks and/or 1024 × 1024 windows ArcGUI intends to process.

## Reuse Matching Exported Imagery

Enabled by default.

This is especially useful when comparing multiple models on the same region.

ArcGUI can reuse existing imagery when the export configuration matches, including the same:

- extent geometry;
- imagery source;
- coordinate system;
- overlap;
- direct/optimized export mode;
- fixed deployment settings.

Changing the following does **not** require imagery to be exported again:

- ONNX model;
- prediction threshold;
- ONNX batch size.

For example, the same exported imagery can be used first with SegFormer and then with U-Net.

If the existing cache is incomplete, corrupt, or does not match the current configuration, ArcGUI rejects it and creates a new export instead of silently using incompatible data.

---

# Recommended Workflow

For a normal GPU workstation:

1. Clone and configure the ArcGIS Python environment.
2. Obtain the ONNX weights from the project OneDrive.
3. Run `python check.py`.
4. Select `segformer_cuda_fp16.onnx`.
5. Confirm CUDA inference succeeds.
6. Add `roadsegmentation_tool.pyt` to ArcGIS Pro.
7. Add the imagery and polygon extent layer to the active map.
8. Open **ArcGUI Road Segmentation**.
9. Leave **Tile Overlap** at `50%` unless another value is required.
10. Keep **Optimized Large-Area Export + Python Cropping** enabled for large regions.
11. Keep **Reuse Matching Exported Imagery** enabled.
12. Start with threshold `0.50`.
13. Start with batch size `2` for SegFormer CUDA or `4` for U-Net CUDA.
14. Optionally run **Debug Footprint Preview Only** before a large job.
15. Run the tool and review the final raster in ArcGIS Pro.

For a faster GPU model comparison, repeat the same cached region using:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

If the spatial/export configuration is unchanged, ArcGUI can reuse the existing imagery instead of downloading it again.

---

# Typical Output Organization

The exact names include the selected model and a configuration/cache identifier, but the general structure is:

```text
output_folder/
│
├── Extent_Name/
│   │
│   ├── export_cache/
│   │   └── <matching export configuration>/
│   │       ├── export_config.txt
│   │       ├── export_config.json
│   │       └── imagery/
│   │
│   └── model_outputs/
│       ├── segformer_best_cuda/
│       │   └── <configuration>/
│       │       └── run/
│       │
│       └── unet_best_cuda/
│           └── <configuration>/
│               └── run/
│
└── final_predictions/
    ├── segformer_best_cuda/
    └── unet_best_cuda/
```

This separation allows several models to use the same imagery cache without mixing their outputs.

---

# Quick Troubleshooting

## CUDA model will not load

Run:

```bash
python check.py
```

Confirm that ONNX Runtime reports:

```text
CUDAExecutionProvider
```

If it does not, check:

- NVIDIA driver;
- `onnxruntime-gpu`;
- CUDA/cuDNN compatibility;
- the active cloned environment;
- DLL visibility.

## GPU out of memory

Reduce **ONNX Batch Size**.

Try:

```text
1
```

## CPU inference is very slow

This is expected, particularly for SegFormer MiT-B5.

Use a CUDA model when an NVIDIA GPU is available.

## Model filename is rejected

Use the distributed filenames without arbitrary renaming.

Recommended examples:

```text
segformer_cuda_fp16.onnx
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

## Input imagery cannot be found

Make sure:

- the imagery is in the active map;
- the GUI name matches the layer name exactly;
- only one layer uses that exact name.

## Extent CRS error

Define the correct CRS on the polygon before running.

Do not leave the source extent as `Unknown`.

The active map should use a projected CRS measured in metres.

## Export is slow

Remote imagery export can represent a significant part of total runtime.

For repeated testing:

- leave **Reuse Matching Exported Imagery** enabled;
- keep the same extent/export configuration;
- rerun different models on the existing cache.

## A cached export is rejected

ArcGUI intentionally validates cached imagery before reuse.

If the geometry, export mode, overlap, imagery source, georeferencing, or cache files no longer match, a new export is safer than reusing incompatible imagery.

---

# Documentation

The README is intended as a practical quick-start reference.

For complete instructions, screenshots, detailed processing behavior, and troubleshooting, use:

```text
reports/technical_guide.pdf
```

For the project methodology, model training, validation results, performance comparisons, conclusions, and future work, use:

```text
reports/technical_report.pdf
```

See [`reports/README.md`](reports/README.md) for a summary.

---

# Model Access / Project Contact

For access to the project ONNX weights, contact **Laurie Vidal** or **Dodick Gasser**.

The files are stored in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal under the **`final`** folder.

---

# Citation

If ArcGUI is used in a project, report, or publication, reference the accompanying technical report and deployment guide in the `reports/` folder, together with any required imagery/data-source attribution.
