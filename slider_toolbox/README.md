# Road Probability Threshold Toolbox — User Guide

This folder contains the threshold-adjustment toolbox used after ArcGUI inference.

Repository folder:

https://github.com/majrblais/ArcGUI/tree/main/slider_toolbox

```text
slider_toolbox/
├── README.md
└── RoadProbabilityThreshold.pyt
```

The toolbox is useful when ArcGUI has already created a road-probability raster and you want to test a different threshold **without rerunning imagery export or ONNX inference**.

---

# 1. Install the toolbox

In ArcGIS Pro:

```text
Catalog
→ Toolboxes
→ Add Toolbox
```

Select:

```text
RoadProbabilityThreshold.pyt
```

The toolbox contains two tools:

```text
1 - Load / Refresh Threshold Preview
2 - Generate Binary Road Mask
```

---

# 2. Input raster

Use a **single-band road probability raster** produced by the ArcGUI workflow.

The expected probability range is approximately:

```text
0.0 to 1.0
```

The tool does not rescale the raster.

A value closer to `1` indicates stronger road probability.

---

# 3. Tool 1 — Load / Refresh Threshold Preview

Use this tool to visually test a threshold quickly.

It does **not** create a new threshold raster.

It only changes the map symbology of a layer pointing to the original probability raster.

## Parameters

### Road Probability Raster

Select the probability raster to inspect.

### Probability Threshold

Default:

```text
0.50
```

The preview rule is:

```text
probability <= threshold  -> hidden
probability > threshold   -> visible road
```

Lower threshold:

- detects more possible road pixels;
- can increase false positives.

Higher threshold:

- keeps only stronger predictions;
- can remove weak/narrow road segments.

### Road Preview Color

Choose one of:

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

### Custom RGB (R,G,B)

Only enabled when **Custom RGB** is selected.

Example:

```text
255,0,0
```

Each value must be from `0` to `255`.

### Visible Road Opacity (%)

Controls the opacity of visible road pixels.

Default:

```text
100
```

### Turn Off Original Probability Layer

Recommended:

```text
True
```

This hides another map layer pointing to the same source raster so only the threshold preview is visible.

## Run the preview

Click **Run**.

The toolbox creates or refreshes a map layer named:

```text
Road Probability Threshold Preview
```

To test another threshold:

1. change the threshold value;
2. click **Run** again.

### Important: this is not a live dragged slider

A standard ArcGIS Python toolbox does not provide a continuously updating drag-slider control.

The workflow is:

```text
change threshold
→ Run
→ inspect map
→ change threshold
→ Run again
```

This is still fast because the probability raster is not recalculated.

---

# 4. Tool 2 — Generate Binary Road Mask

Use this tool after choosing the threshold you want to keep.

It creates a permanent binary raster.

The output rule is:

```text
probability > threshold  -> 1 = road
probability <= threshold -> 0 = background
NoData in source         -> NoData in output
```

## Parameters

### Road Probability Raster

Select the probability raster.

### Probability Threshold

Use the threshold selected during preview.

Example:

```text
0.50
```

### Output Binary Road Raster

Choose where to save the permanent result.

When possible, the toolbox suggests a filename next to the source raster similar to:

```text
<source>_binary_threshold.tif
```

### Road Display Color

Choose the map color for road value `1`.

### Custom RGB (R,G,B)

Used only when **Custom RGB** is selected.

### Road Display Opacity (%)

Controls the map display opacity.

### Add Binary Raster to Current Map

Default:

```text
True
```

When enabled, the output is added to the active map automatically.

The display is styled so:

```text
0 = transparent background
1 = visible road color
```

---

# 5. Recommended threshold workflow

A simple workflow is:

1. run ArcGUI with threshold `0.50`;
2. locate the saved probability raster;
3. open **1 - Load / Refresh Threshold Preview**;
4. test values such as `0.35`, `0.40`, `0.45`, `0.50`, `0.55`, or `0.60`;
5. compare road continuity and false positives in the map;
6. choose the threshold that best matches the intended mapping task;
7. run **2 - Generate Binary Road Mask** with that threshold.

You do **not** need to redownload imagery or rerun the ONNX model simply to change the threshold.

---

# 6. What the colors mean

The color and opacity settings affect **display only**.

They do not change the probability values or the binary road classification.

For the permanent binary raster:

```text
0 = background
1 = road
```

---

# 7. Troubleshooting

## Preview does not appear

Make sure:

- an ArcGIS Pro map is open;
- the raster can be resolved from the map or disk;
- the raster is single-band;
- the raster has valid values/statistics.

## The preview looks almost completely filled

The threshold may be too low, or the source raster may not be a 0–1 probability raster.

## Almost no roads appear

The threshold may be too high.

## Custom RGB gives an error

Use exactly three integers between 0 and 255:

```text
R,G,B
```

Example:

```text
0,255,255
```

## I want a true live drag slider

A standard `.pyt` toolbox cannot provide a continuously updating ArcGIS slider.

That would require a separate ArcGIS Pro SDK add-in.

---

# 8. Relationship to the main ArcGUI toolbox

The main toolbox performs the expensive work:

```text
imagery export
→ ONNX inference
→ probability reconstruction
```

This threshold toolbox works on the saved probability result:

```text
probability raster
→ preview threshold
or
→ permanent binary mask
```

Use the main toolbox first, then this toolbox for threshold refinement.
