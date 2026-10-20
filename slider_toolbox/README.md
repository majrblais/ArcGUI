# 🎚️ Road Probability Threshold Toolbox

This folder contains the optional ArcGIS Pro toolbox used to explore and apply probability thresholds to ArcGUI road-probability rasters.

GitHub folder:

https://github.com/majrblais/ArcGUI/tree/main/slider_toolbox

```text
slider_toolbox/
├── README.md
└── RoadProbabilityThreshold.pyt
```

The toolbox is independent from the main ArcGUI inference toolbox.

It does **not** load an ONNX model and does **not** rerun road segmentation.

Instead, it works with an existing single-band road-probability raster produced by ArcGUI or another compatible workflow.

---

# Why Use This Toolbox?

ArcGUI normally applies a default prediction threshold of `0.50` after reconstructing the probability surface.

However, the best visual threshold can vary slightly depending on imagery, road appearance, and the desired balance between missed roads and false positives.

This toolbox makes it easy to:

- preview different thresholds quickly;
- hide low-probability pixels without creating a new raster every time;
- choose a road display color and opacity;
- create a permanent binary 0/1 road raster once a useful threshold has been selected.

---

# Installation

In ArcGIS Pro:

```text
Catalog
→ Toolboxes
→ Add Toolbox
```

Select:

```text
slider_toolbox/RoadProbabilityThreshold.pyt
```

The toolbox appears as:

```text
Road Probability Threshold
```

It contains two tools:

```text
1 - Load / Refresh Threshold Preview
2 - Generate Binary Road Mask
```

---

# Tool 1 — Load / Refresh Threshold Preview

This is the recommended tool for trying different threshold values.

## What it does

The tool adds or reuses a map layer pointing to the **original probability raster** and changes only its ArcGIS symbology.

The display rule is:

```text
probability <= threshold
    -> transparent

probability > threshold
    -> visible road color
```

No new threshold raster is created.

Because only the layer symbology is updated, changing the threshold is fast.

## Parameters

### Road Probability Raster

Select the single-band probability raster to preview.

This is normally an ArcGUI probability output containing values approximately between `0` and `1`.

### Probability Threshold

Valid range:

```text
0.0 - 1.0
```

Recommended starting value:

```text
0.50
```

The comparison is strict:

```text
probability > threshold
```

For example, with a threshold of `0.50`, a pixel with probability `0.50` is hidden and a pixel with probability `0.51` is displayed.

### Road Preview Color

Available presets include:

```text
Red
Orange
Yellow
Green
Cyan
Blue
Magenta
White
Black
Custom RGB
```

### Custom RGB

When **Custom RGB** is selected, enter a value such as:

```text
255,0,0
```

Each value must be an integer from `0` to `255`.

### Visible Road Opacity (%)

Controls the opacity of the visible road pixels.

Valid range:

```text
0 - 100
```

### Turn Off Original Probability Layer

When enabled, another map layer pointing to the same probability raster is turned off so that only the threshold preview remains visible.

## How to refresh the preview

A standard ArcGIS Pro `.pyt` Python toolbox does not provide a continuously updating drag slider.

To test another threshold:

1. change **Probability Threshold**;
2. click **Run** again.

The source probability raster is not rewritten.

---

# Tool 2 — Generate Binary Road Mask

Use this tool once a useful threshold has been selected.

## What it creates

The output is a permanent single-band binary road raster:

```text
0 = background
1 = road
```

The rule is:

```text
probability > threshold  -> 1
probability <= threshold -> 0
```

Source NoData is preserved as NoData.

## Parameters

### Road Probability Raster

The input single-band road-probability raster.

### Probability Threshold

The final selected threshold, normally first tested with Tool 1.

### Output Binary Road Raster

Destination for the permanent binary output.

When the input is a TIFF, ArcGIS attempts to suggest a filename similar to:

```text
<probability_name>_binary_threshold.tif
```

### Road Display Color

Controls the map color applied to road value `1` after the output is added to the map.

Background value `0` is displayed transparently.

### Custom RGB

Optional custom road color in `R,G,B` form.

Example:

```text
0,255,0
```

### Road Display Opacity (%)

Controls road-layer opacity in the ArcGIS map.

### Add Binary Raster to Current Map

When enabled, the new binary raster is automatically added to the active map and styled so:

```text
0 -> transparent background
1 -> selected road color
```

---

# Recommended Workflow

A typical ArcGUI workflow is:

```text
ArcGUI Road Segmentation
        ↓
continuous road-probability raster
        ↓
Tool 1: Load / Refresh Threshold Preview
        ↓
try 0.40, 0.45, 0.50, 0.55, ...
        ↓
visually select the preferred threshold
        ↓
Tool 2: Generate Binary Road Mask
        ↓
permanent 0/1 raster
```

For most deployments, begin with:

```text
Threshold = 0.50
```

Then adjust only if the visual road network is too permissive or too restrictive.

A lower threshold generally increases road detections but can also increase false positives.

A higher threshold generally removes weaker detections but can also create missing road segments.

---

# Probability Raster Requirements

The toolbox expects a **single-band raster**.

The intended input contains road probabilities approximately in the range:

```text
0.0 - 1.0
```

The toolbox does not rescale the data.

If the raster statistics extend outside `[0,1]`, the preview tool displays a warning because the input may not represent probabilities.

---

# Important Notes

## This is not a true live slider

The name `slider_toolbox` refers to threshold exploration, but a standard ArcGIS Python toolbox cannot update continuously while the user drags a GUI slider.

The practical workflow is:

```text
change threshold
→ click Run
→ inspect preview
→ repeat
```

A true continuously updating slider would require an ArcGIS Pro SDK add-in rather than a normal `.pyt` toolbox.

## Tool 1 does not duplicate the raster

The preview is symbology only.

This is intentional so users can test many thresholds quickly without generating many large TIFF files.

## Tool 2 is the permanent output step

Only use Tool 2 when a threshold has been selected and a real binary raster is required for export, GIS processing, or sharing.

---

# Troubleshooting

## Probability raster cannot be resolved

Make sure the raster:

- exists on disk or is loaded in the active map;
- is readable by ArcGIS Pro;
- is a raster dataset rather than an unrelated layer type.

## The raster has several bands

The toolbox requires a single-band probability raster.

Use the actual road-probability output rather than the original RGB imagery.

## Custom RGB gives an error

Use exactly three comma-separated integers:

```text
255,0,0
```

Each value must be from `0` to `255`.

## Threshold preview does not update automatically while typing or dragging

This is expected.

Change the threshold and click **Run** again.

## Binary raster is created but symbology is not perfect

The raster itself can still be valid.

Automatic map styling is a convenience step and ArcGIS symbology behavior can vary slightly between versions.

The important output values remain:

```text
0 = background
1 = road
```

---

# Relationship to the Main ArcGUI Toolbox

The main toolbox is located in:

```text
toolbox_files/
```

and performs imagery export and model inference.

The threshold toolbox is located in:

```text
slider_toolbox/
```

and operates **after inference** on the probability raster.

The two toolboxes can be installed independently, but they are designed to work together.

For the full ArcGUI deployment instructions, see:

```text
../reports/technical_guide.pdf
```
