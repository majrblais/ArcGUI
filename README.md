# 🛣️ ArcGUI Road Segmentation

ArcGUI is an ArcGIS Pro workflow for extracting forestry, rural, agricultural, and dirt-road networks from high-resolution RGB imagery with ONNX deep-learning models.

This README is the **starting point** for using the repository. It explains the overall workflow, the required environment, the recommended models, and which tool to open for each task.

For step-by-step use of a specific tool, open the README inside that tool's folder.

Repository: https://github.com/majrblais/ArcGUI

---

## Repository layout

```text
ArcGUI/
│
├── README.md
├── MODEL_WEIGHTS.md
├── requirements.txt
│
├── toolbox_files/
│   ├── README.md
│   ├── roadsegmentation_tool.pyt
│   └── roadsegmentation_python.py
│
├── slider_toolbox/
│   ├── README.md
│   └── RoadProbabilityThreshold.pyt
│
├── utils/
│   ├── README.md
│   ├── check.py
│   └── python_inference_script.py
│
└── reports/
    ├── README.md
    ├── technical_guide.pdf
    └── technical_report.pdf
```

Folder guides:

- Main ArcGIS tool: [`toolbox_files/README.md`](toolbox_files/README.md)
- Threshold / slider-style tool: [`slider_toolbox/README.md`](slider_toolbox/README.md)
- Model-check and simple inference utilities: [`utils/README.md`](utils/README.md)
- Full project documents: [`reports/README.md`](reports/README.md)
- Model access: [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md)

---

# 1. What the workflow does

The normal ArcGUI workflow is:

```text
ArcGIS polygon extent
        ↓
RGB imagery / World Imagery
        ↓
ArcGIS imagery export
        ↓
1024 × 1024 model windows
        ↓
ONNX road-segmentation model
        ↓
road probabilities
        ↓
overlap averaging
        ↓
threshold
        ↓
georeferenced road raster
```

For large regions, ArcGUI can export larger imagery blocks first and create the 1024 × 1024 model windows locally. This is the recommended mode for large areas because it reduces repeated imagery-service requests.

ArcGUI can also reuse previously exported imagery when the spatial/export configuration matches. This is useful when comparing SegFormer and U-Net on the same region.

---

# 2. Recommended models

The two recommended CUDA models are:

```text
segformer_cuda_fp16.onnx
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

Use **SegFormer MiT-B5** as the primary model when the strongest overall segmentation quality is preferred.

Use **U-Net ResNet101** when faster inference is more important. In the project evaluation, the successfully trained models generally produced similar road-mapping results, so the choice between the final SegFormer and best U-Net is mainly a quality-versus-speed deployment decision.

Additional CPU, last-checkpoint, and old-milestone models may also be provided for compatibility, comparison, or reproducibility.

See [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md) for model access information.

---

# 3. Environment setup

## 3.1 Install ArcGIS Pro

The main toolbox is designed to run inside ArcGIS Pro.

The production imagery-export workflow requires the ArcGIS **Image Analyst** capability used by `Export Training Data for Deep Learning`.

## 3.2 Clone the ArcGIS Pro Python environment

Do not install the project packages directly into the default ArcGIS Pro environment.

In ArcGIS Pro:

```text
Project / Settings
→ Package Manager
→ Clone Environment
```

A practical cloned environment name is:

```text
torch_roads
```

Activate it from an ArcGIS Pro / Conda prompt:

```bash
conda activate torch_roads
```

## 3.3 Install the Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU deployment, ONNX Runtime must expose:

```text
CUDAExecutionProvider
```

For CPU-only deployment, ONNX Runtime must expose:

```text
CPUExecutionProvider
```

## 3.4 Check the model before using ArcGUI

Run:

```bash
python utils/check.py
```

Select the ONNX model when prompted.

For a CUDA model, the check should confirm that the model actually loads and performs inference using `CUDAExecutionProvider`.

Detailed instructions: [`utils/README.md`](utils/README.md)

---

# 4. ArcGIS project setup

Before running the main toolbox:

1. Open an ArcGIS Pro map.
2. Add the imagery layer to the active map.
3. Add the polygon extent layer to process.
4. Make sure the extent layer has a **defined coordinate system**.
5. Use a **projected map coordinate system measured in metres**.
6. Set the ArcGIS raster-analysis cell size to **2**.
7. Choose an output folder with sufficient free disk space.
8. Select the ONNX model.

## Cell size = 2

The deployment is designed around imagery at approximately **2 metres per pixel**.

In the ArcGIS Geoprocessing pane:

```text
Environments
→ Raster Analysis
→ Cell Size
→ 2
```

The active map should therefore use a projected coordinate system whose units are metres.

The main backend also uses the fixed 2 m/pixel deployment setting internally, but keeping the ArcGIS environment at `2` makes related raster processing consistent and makes the project setup explicit for users.

---

# 5. Which tool should I use?

## Main road-segmentation toolbox

Folder:

https://github.com/majrblais/ArcGUI/tree/main/toolbox_files

Use this for the actual georeferenced ArcGIS road-extraction workflow.

Main file:

```text
roadsegmentation_tool.pyt
```

It handles imagery export, model inference, overlap, caching, clipping, and final GIS raster creation.

Guide: [`toolbox_files/README.md`](toolbox_files/README.md)

## Probability-threshold toolbox

Folder:

https://github.com/majrblais/ArcGUI/tree/main/slider_toolbox

Use this after ArcGUI has produced a probability raster and you want to test a different threshold without rerunning the ONNX model.

Main file:

```text
RoadProbabilityThreshold.pyt
```

It contains:

```text
1 - Load / Refresh Threshold Preview
2 - Generate Binary Road Mask
```

Guide: [`slider_toolbox/README.md`](slider_toolbox/README.md)

## Utilities

Folder:

https://github.com/majrblais/ArcGUI/tree/main/utils

Contains:

```text
check.py
python_inference_script.py
```

Use `check.py` to verify a model/environment.

Use `python_inference_script.py` as a simple folder-of-images → masks example.

Guide: [`utils/README.md`](utils/README.md)

## Reports

Folder:

https://github.com/majrblais/ArcGUI/tree/main/reports

The README files are intended as practical how-to documentation. The PDFs contain the complete technical material.

Guide: [`reports/README.md`](reports/README.md)

---

# 6. Quick-start ArcGUI workflow

For a normal NVIDIA GPU workstation:

1. Clone the ArcGIS Python environment.
2. Install `requirements.txt`.
3. Obtain the ONNX weights.
4. Run `python utils/check.py` and confirm the selected CUDA model works.
5. Open ArcGIS Pro and activate the cloned Python environment.
6. Set the active map to a projected CRS in metres.
7. Set **Environments → Raster Analysis → Cell Size = 2**.
8. Add the imagery layer, normally `World Imagery`.
9. Add the polygon extent layer.
10. Add `toolbox_files/roadsegmentation_tool.pyt` to ArcGIS Pro.
11. Open **ArcGUI Road Segmentation**.
12. Use `segformer_cuda_fp16.onnx` or `best_unet_resnet101_norm_1024_cuda_fp16.onnx`.
13. Start with `50%` overlap.
14. Keep **Optimized Large-Area Export + Python Cropping** enabled for large regions.
15. Start with threshold `0.50`.
16. Keep **Reuse Matching Exported Imagery** enabled.
17. Run the tool.
18. If you want to experiment with the final threshold, use the `slider_toolbox` on the probability raster.

---

# 7. Main output idea

The output structure separates shared imagery exports from model-specific results.

A typical organization is:

```text
output_folder/
│
├── Extent_Name/
│   ├── export_cache/
│   │   └── <matching export configuration>/
│   │       ├── export_config.txt
│   │       ├── export_config.json
│   │       └── imagery/
│   │
│   └── model_outputs/
│       ├── segformer_best_cuda/
│       └── unet_best_cuda/
│
└── final_predictions/
```

This lets different models reuse the same matching imagery cache instead of downloading the same region again.

---

# 8. Important deployment rules

- Model input size is fixed at `1024 × 1024` pixels.
- Input imagery is RGB.
- The deployment resolution is `2 m/pixel`.
- The active map should use a projected CRS in metres.
- The extent must be polygon geometry with a defined CRS.
- The current ArcGUI uses the **exact selected polygon**; there is no user-configurable extent buffer.
- Default overlap is `50%`.
- Default threshold is `0.50`.
- Keep the distributed ONNX filenames unless there is a specific reason to rename them.
- CUDA models require `CUDAExecutionProvider`.
- The main `.pyt` and `roadsegmentation_python.py` must remain together.

---

# 9. Model weights

The ONNX weights are not stored in the public GitHub repository.

Contact **Laurie Vidal** or **Dodick Gasser**.

The project weights are stored in the **CCNB Innov – Chemin forestier** OneDrive shared by Laurie Vidal, under the **`final`** folder.

See [`MODEL_WEIGHTS.md`](MODEL_WEIGHTS.md).

---

# 10. Need more detail?

Use the folder README that corresponds to the task you are doing:

- [`toolbox_files/README.md`](toolbox_files/README.md) — main ArcGIS tool
- [`slider_toolbox/README.md`](slider_toolbox/README.md) — threshold preview and binary-mask creation
- [`utils/README.md`](utils/README.md) — model check and simple image-folder inference
- [`reports/README.md`](reports/README.md) — full PDF documentation

The GitHub README set is intended to contain everything needed to **install, launch, and operate** the project tools. The PDFs are for the deeper technical explanation, model-development details, and project results.
