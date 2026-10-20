# ArcGUI Road Segmentation — Main Toolbox Guide

This folder contains the main ArcGIS Pro road-segmentation tool.

Repository folder:

https://github.com/majrblais/ArcGUI/tree/main/toolbox_files

```text
toolbox_files/
├── README.md
├── roadsegmentation_tool.pyt
└── roadsegmentation_python.py
```

Keep the `.pyt` and backend `.py` file together.

---

# 1. What this toolbox does

**ArcGUI Road Segmentation** takes a polygon processing area, obtains RGB imagery, runs the selected ONNX road-segmentation model, and creates georeferenced road outputs.

The tool can work in two ways:

- **Direct mode:** ArcGIS exports the retained 1024 × 1024 model tiles directly.
- **Optimized mode:** ArcGIS exports larger imagery blocks and Python creates the 1024 × 1024 model windows locally.

Optimized mode is recommended for large areas.

The current version also supports reusable imagery caches so the same region does not need to be downloaded again when only the model, threshold, or batch size changes.

---

# 2. Before opening the toolbox

## ArcGIS Pro

Open an ArcGIS Pro project with an active map.

## Coordinate system

The active map must use a **projected coordinate system measured in metres**.

Do not leave the input extent with an Unknown/undefined CRS.

Generally, when using World Imagery, the default CRS is ok and nothing has to be modified.

## Cell size

The deployment is designed for:

```text
2 metres / pixel
```

In the ArcGIS Geoprocessing pane, open:

```text
Environments
→ Raster Analysis
→ Cell Size
```

Enter:

```text
2
```

This keeps ArcGIS raster processing aligned with the model deployment resolution.

## Imagery

Add the RGB imagery layer to the active map.

Typical layer name:

```text
World Imagery
```

If another imagery layer is used, enter its exact ArcGIS layer name in the GUI.

ArcGUI expects that name to identify one unique usable imagery/raster layer.

## Extent

Add the polygon layer that defines the processing region.
It can be a layer such as the CSRNO limit, or a drawn polygon.

---

# 3. Install the toolbox

In ArcGIS Pro:

```text
Catalog
→ Toolboxes
→ Add Toolbox
```

Select:

```text
roadsegmentation_tool.pyt
```

The tool appears as:

```text
ArcGUI Road Segmentation
```

Do not separate `roadsegmentation_tool.pyt` from `roadsegmentation_python.py`.

---

# 4. Recommended model choices

Primary model:

```text
segformer_cuda_fp16.onnx
```

Faster CUDA alternative:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

Start with the primary SegFormer model when segmentation quality is the priority.

Use the best U-Net model when faster inference is preferred.

Other CPU / last / old models are mainly for compatibility, comparison, or reproducibility.

Before first use on a new computer, run:

```bash
python ../utils/check.py
```

and verify the selected model.

---

# 5. GUI settings

## Extent Layer(s)

Select one or more polygon layer entries.

Requirements:

- polygon geometry;
- defined CRS;
- valid geometry;
- no duplicate extent entry in the same job.

If a layer has selected polygons, the selection is used.

If one layer contains several selected polygons, those polygons are processed together for that layer entry.

## Input Raster Layer Name

Name of the imagery layer in the active map.

Default:

```text
World Imagery
```

The name must match exactly one usable imagery/raster layer.

If two layers have the same name, ArcGUI stops instead of guessing.

## Output Folder

Choose the root folder where ArcGUI will create:

- reusable imagery cache;
- model-specific processing folders;
- probability outputs;
- final predictions.

Use a short local path when possible.

Make sure the drive has sufficient free disk space.

## Road Segmentation ONNX Model

Browse to the ONNX model.

Recommended:

```text
segformer_cuda_fp16.onnx
```

or:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

ArcGUI uses the filename to identify the model family and intended execution target, then validates the actual ONNX graph before inference.

Basic filename behavior:

```text
contains "segformer"
    -> SegFormer MiT-B5

contains "unet" + "resnet101" or "r101"
    -> U-Net ResNet101

contains "cpu"
    -> CPU only

contains "cuda" or "fp16"
    -> CUDA required

otherwise
    -> CUDA preferred, CPU fallback
```

## Tile Overlap (%)

Default:

```text
50
```

The model window is always 1024 × 1024 pixels.

Examples:

| Overlap | Approx. stride |
|---:|---:|
| 0% | 1024 px |
| 25% | 768 px |
| 50% | 512 px |
| 75% | 256 px |

More overlap means more model evaluations and longer runtime.

Use `50%` as the normal starting value.

## Optimized Large-Area Export + Python Cropping

Recommended for large areas.

### Enabled

ArcGUI exports larger balanced non-overlapping imagery blocks first.

Then Python creates the overlapping 1024 × 1024 model windows from those exported TIFFs.

Large blocks and model windows that do not intersect the target polygon are skipped.

This mode usually makes large-area processing much more efficient.

### Disabled

ArcGIS directly exports the retained 1024 × 1024 model tiles.

This can be useful for smaller jobs or comparison/debugging.

## Prediction Threshold

Default:

```text
0.50
```

Valid range:

```text
0.0 to 1.0
```

ArcGUI first reconstructs the probability surface and then applies the threshold.

This means a saved probability output can later be re-thresholded with the `slider_toolbox` without rerunning the model.

## ONNX Batch Size

Number of model windows sent to ONNX Runtime in one call.

Recommended starting values:

| Model | Starting batch |
|---|---:|
| SegFormer CUDA | 2 |
| U-Net CUDA | 4 |
| CPU deployment | 1 |

If GPU memory errors occur, reduce the batch size to `1`.

Do not change the 1024 × 1024 model input size as a memory workaround.

## Debug Footprint Preview Only

Enable this before a large job if you want to inspect the spatial processing plan.

Debug mode does **not**:

- export imagery;
- load the ONNX model;
- run inference.

It creates map footprints showing which blocks/windows will be retained or skipped.

## Reuse Matching Exported Imagery

Recommended:

```text
Enabled
```

If ArcGUI finds a complete matching cache, it can skip the World Imagery export and run the selected model directly on the existing TIFFs.

A matching cache depends on the export-relevant setup, including:

- polygon geometry;
- imagery source;
- map/export CRS;
- overlap;
- optimized/direct mode;
- fixed deployment settings.

The following may change without forcing a new imagery export:

```text
ONNX model
Prediction threshold
ONNX batch size
```

This is especially useful when comparing SegFormer and U-Net on the same extent.

If a cache is missing files, changed, incomplete, incorrectly georeferenced, or otherwise incompatible, ArcGUI rejects it and creates a fresh export.

---

# 6. Recommended normal run

For a typical large-area CUDA run:

```text
Extent Layer(s):                  your polygon layer
Input Raster Layer Name:          World Imagery
Output Folder:                    your output folder
Road Segmentation ONNX Model:     segformer_cuda_fp16.onnx
Tile Overlap (%):                 50
Optimized Large-Area Export:      checked
Prediction Threshold:             0.50
ONNX Batch Size:                  2
Debug Footprint Preview Only:     unchecked
Reuse Matching Exported Imagery:  checked
```

For the best U-Net CUDA model, a batch size of `4` is a reasonable starting point if GPU memory allows it.

---

# 7. Optional ArcGIS processing extent control

ArcGIS Pro also has an **Environments → Processing Extent** section where you can draw or enter a rectangular processing extent.

This is useful when preparing or inspecting a rectangular area of interest.

However, the **Extent Layer(s)** polygon selected in ArcGUI remains the authoritative target geometry for the ArcGUI workflow.

If a rectangle drawn in ArcGIS is intended to be the true road-extraction area, prepare/use that rectangle as the polygon extent supplied to ArcGUI.

---

# 8. What happens when you click Run

At a high level ArcGUI:

1. validates the GUI settings;
2. validates the polygon and coordinate systems;
3. resolves the imagery layer;
4. checks for a reusable matching imagery cache;
5. plans direct tiles or optimized large blocks;
6. exports imagery only where needed;
7. validates the selected ONNX model and execution provider;
8. runs inference on the 1024 × 1024 windows;
9. averages overlapping probabilities;
10. clips the result to the requested polygon;
11. applies the selected threshold;
12. writes model-specific outputs and final predictions.

---

# 9. Typical output structure

ArcGUI organizes the outputs by:

1. processing extent;
2. reusable imagery/export configuration;
3. model used for inference.

This makes it possible to process the same area with several models without overwriting previous results and, when the configuration is unchanged, without downloading the imagery again.

For example, assume the ArcGIS polygon extent layer is named:

```text
TestExtent
```

and the selected model is:

```text
segformer_cuda_fp16.onnx
```

with:

```text
Tile Overlap = 50%
Optimized Large-Area Export = enabled
```

ArcGUI will create a structure similar to:

```text
output_folder/
│
├── TestExtent/
│   │
│   ├── export_cache/
│   │   └── optimized_ov50_a1b2c3d4e5f6/
│   │       │
│   │       ├── export_config.txt
│   │       ├── export_config.json
│   │       │
│   │       └── imagery/
│   │           └── ...
│   │
│   └── model_outputs/
│       └── segformer_best_cuda/
│           └── optimized_ov50_a1b2c3d4e5f6/
│               └── run/
│                   │
│                   ├── run_status.txt
│                   ├── model_run_info.txt
│                   ├── ...
│                   │
│                   └── final/
│                       ├── roads_probability_mosaic.tif
│                       ├── roads_probability_clipped_to_extent.tif
│                       ├── roads_prediction_binary.tif
│                       └── TestExtent_segformer_best_cuda_roads_prediction_clean.tif
│
└── final_predictions/
    └── segformer_best_cuda/
        └── optimized_ov50_a1b2c3d4e5f6/
            └── TestExtent_segformer_best_cuda_roads_prediction_clean.tif
```

> `a1b2c3d4e5f6` is only an example configuration/cache identifier. ArcGUI generates the real identifier automatically from the processing configuration.

## Which files are important?

### Final road prediction

The main final product for normal GIS use is:

```text
TestExtent_segformer_best_cuda_roads_prediction_clean.tif
```

A convenient copy is stored under:

```text
output_folder/
└── final_predictions/
    └── segformer_best_cuda/
        └── <configuration>/
            └── TestExtent_segformer_best_cuda_roads_prediction_clean.tif
```

This is the final thresholded road raster generated using the threshold selected in the ArcGUI interface.

For example, if:

```text
Prediction Threshold = 0.50
```

then the final road raster is the road/no-road result produced at that threshold.

This is normally the file to add to a GIS project, use for mapping, or provide as the final road-extraction result.

### Probability raster to keep for future threshold testing

The most useful file to keep for later threshold adjustment is:

```text
roads_probability_clipped_to_extent.tif
```

It is located inside the model run folder:

```text
TestExtent/
└── model_outputs/
    └── segformer_best_cuda/
        └── <configuration>/
            └── run/
                └── final/
                    └── roads_probability_clipped_to_extent.tif
```

This raster contains the continuous road-probability values produced by the model and clipped to the selected `TestExtent` polygon.

Its values are approximately:

```text
0.0  -> low road probability
...
1.0  -> high road probability
```

This is the recommended input for the **Road Probability Threshold** toolbox.

Keep this file if you may want to try a different threshold later.

For example, you can test values such as:

```text
0.30
0.40
0.50
0.60
0.70
```

without:

- downloading imagery again;
- rerunning SegFormer or U-Net;
- repeating ONNX inference.

Only the conversion from probability to a visible/binary road mask changes.

### `roads_probability_mosaic.tif`

ArcGUI also creates:

```text
roads_probability_mosaic.tif
```

This is the reconstructed continuous probability surface created from the overlapping model predictions before the final polygon clip.

It is an important intermediate product, but for normal post-processing and threshold experimentation, use:

```text
roads_probability_clipped_to_extent.tif
```

because that file has already been clipped to the requested polygon.

### `roads_prediction_binary.tif`

ArcGUI also creates:

```text
roads_prediction_binary.tif
```

This is the direct thresholded raster created during finalization using the threshold selected in the ArcGUI run.

It is mainly an intermediate product used to produce the cleaned final road raster.

For normal use, prefer:

```text
TestExtent_segformer_best_cuda_roads_prediction_clean.tif
```

### `run_status.txt` and `model_run_info.txt`

Each model run also records small text files such as:

```text
run_status.txt
model_run_info.txt
```

These are useful when reviewing or troubleshooting a previous run.

They record basic run information such as the selected model, execution target, threshold, batch size, and associated export cache.

## Running another model on the same extent

Suppose `TestExtent` was first processed using:

```text
segformer_cuda_fp16.onnx
```

and you later run exactly the same extent/export configuration using:

```text
best_unet_resnet101_norm_1024_cuda_fp16.onnx
```

ArcGUI can reuse the same imagery cache:

```text
TestExtent/
├── export_cache/
│   └── optimized_ov50_a1b2c3d4e5f6/
│
└── model_outputs/
    ├── segformer_best_cuda/
    │   └── optimized_ov50_a1b2c3d4e5f6/
    │       └── run/
    │
    └── unet_best_cuda/
        └── optimized_ov50_a1b2c3d4e5f6/
            └── run/
```

The imagery under `export_cache` is shared, while each model receives its own model-output folder.

The final prediction folder will then contain separate results:

```text
final_predictions/
├── segformer_best_cuda/
│   └── optimized_ov50_a1b2c3d4e5f6/
│       └── TestExtent_segformer_best_cuda_roads_prediction_clean.tif
│
└── unet_best_cuda/
    └── optimized_ov50_a1b2c3d4e5f6/
        └── TestExtent_unet_best_cuda_roads_prediction_clean.tif
```

This makes it easy to compare SegFormer and U-Net on exactly the same exported imagery.

## Repeated runs

If the same model/configuration is run again, ArcGUI does not overwrite an existing model run.

A new run folder can be created automatically, for example:

```text
run/
run_1/
run_2/
```

This prevents previous outputs from being accidentally deleted.

## Quick reference

```text
FINAL ROAD MAP TO USE:
TestExtent_segformer_best_cuda_roads_prediction_clean.tif

KEEP FOR FUTURE THRESHOLD CHANGES:
roads_probability_clipped_to_extent.tif
```

---

# 10. After inference: try another threshold

The threshold selected in ArcGUI does **not** permanently lock the model prediction to that value.

ArcGUI keeps the continuous probability raster so different thresholds can be tested later without rerunning the neural network.

For the `TestExtent` example, use:

```text
TestExtent/
└── model_outputs/
    └── segformer_best_cuda/
        └── <configuration>/
            └── run/
                └── final/
                    └── roads_probability_clipped_to_extent.tif
```

as the input to:

```text
../slider_toolbox/RoadProbabilityThreshold.pyt
```

See the complete threshold-tool guide:

[`../slider_toolbox/README.md`](../slider_toolbox/README.md)

The recommended workflow is:

```text
ArcGUI inference
        ↓
roads_probability_clipped_to_extent.tif
        ↓
Load / Refresh Threshold Preview
        ↓
try different threshold values
        ↓
choose the preferred threshold
        ↓
Generate Binary Road Mask
        ↓
final custom-threshold road raster
```

For example, if the original ArcGUI run used:

```text
Threshold = 0.50
```

but visual inspection shows that some narrow roads are being missed, you can preview:

```text
Threshold = 0.40
```

using the probability raster.

If `0.40` gives the preferred result, use **Generate Binary Road Mask** to create a permanent road raster at that threshold.

No imagery export or ONNX inference needs to be repeated.

---

# 11. Common problems

## CUDA model selected but CUDA is unavailable

Run:

```bash
python ../utils/check.py
```

If `CUDAExecutionProvider` is not available, fix the GPU ONNX Runtime / CUDA environment or use an appropriate CPU model.

## Extent has Unknown CRS

Define the correct source CRS before running ArcGUI.

## Active map uses degrees or feet

Switch to a projected CRS measured in metres.

## Imagery layer not found

Check the exact layer name in the active map.

## Duplicate imagery-layer names

Rename the intended imagery layer so its name is unique.

## GPU out of memory

Reduce ONNX Batch Size.

## Large job is exporting imagery again

Check whether:

- the polygon changed;
- overlap changed;
- direct/optimized mode changed;
- imagery source changed;
- cache files were moved/modified;
- the current cache schema is different.

---

# 12. Full technical guide

This README is intentionally a how-to guide.

For the deeper technical details, figures, validation behavior, model experiments, and results, use the PDFs in:

```text
../reports/
```
