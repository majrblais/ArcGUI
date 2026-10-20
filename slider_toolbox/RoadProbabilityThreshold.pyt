# -*- coding: utf-8 -*-
"""
RoadProbabilityThreshold.pyt

Secondary ArcGIS Pro Python toolbox for:
1) fast probability-threshold preview using raster symbology only
2) permanent binary road-mask generation

Preview behavior:
- Does NOT rewrite the probability raster.
- Adds/reuses a map layer pointing to the original probability TIFF.
- Symbolizes values <= threshold as fully transparent.
- Symbolizes values > threshold in the selected road color.
- Re-running the tool with a different threshold is therefore fast.

A Python toolbox does not provide a true continuously-dragged slider control.
The Threshold parameter is constrained to [0, 1]; change it and click Run to
refresh the preview. A true drag-and-update slider would require an ArcGIS Pro
SDK add-in rather than a standard .pyt geoprocessing toolbox.
"""

import arcpy
import os
from pathlib import Path


TOOLBOX_NAME = "Road Probability Threshold"
PREVIEW_LAYER_NAME = "Road Probability Threshold Preview"


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

COLOR_TABLE = {
    "Red": (255, 0, 0),
    "Orange": (255, 165, 0),
    "Yellow": (255, 255, 0),
    "Green": (0, 255, 0),
    "Cyan": (0, 255, 255),
    "Blue": (0, 102, 255),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
}


def _active_map():
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    m = aprx.activeMap
    if m is None:
        raise RuntimeError(
            "No active ArcGIS Pro map is available. Open a map and try again."
        )
    return aprx, m


def _resolve_raster_path(value):
    """
    Resolve either a map raster layer or a raster dataset path to a catalog path.
    """
    if value is None:
        raise ValueError("No probability raster was supplied.")

    text = str(value)

    # Direct path first.
    if arcpy.Exists(text):
        try:
            desc = arcpy.Describe(text)
            catalog = getattr(desc, "catalogPath", None)
            if catalog:
                return str(catalog)
        except Exception:
            return text

    # If the parameter came from a map layer, Describe often resolves it.
    try:
        desc = arcpy.Describe(value)
        catalog = getattr(desc, "catalogPath", None)
        if catalog:
            return str(catalog)
    except Exception:
        pass

    # Search active map by layer name.
    try:
        _, m = _active_map()
        for lyr in m.listLayers():
            if lyr.name == text:
                try:
                    return str(lyr.dataSource)
                except Exception:
                    try:
                        return str(arcpy.Describe(lyr).catalogPath)
                    except Exception:
                        pass
    except Exception:
        pass

    raise RuntimeError(
        f"Could not resolve the probability raster source from: {text}"
    )


def _validate_probability_raster(path):
    r = arcpy.Raster(path)

    if int(r.bandCount) != 1:
        raise RuntimeError(
            f"Probability raster must be single-band; found {r.bandCount} bands."
        )

    # These are advisory only. We do not rescale the probability raster.
    try:
        mn = float(arcpy.management.GetRasterProperties(path, "MINIMUM").getOutput(0))
        mx = float(arcpy.management.GetRasterProperties(path, "MAXIMUM").getOutput(0))
    except Exception:
        mn, mx = None, None

    return r, mn, mx


def _parse_rgb(color_name, custom_rgb):
    if color_name != "Custom RGB":
        return COLOR_TABLE[color_name]

    if custom_rgb is None or not str(custom_rgb).strip():
        raise ValueError(
            "Custom RGB is selected but no RGB value was supplied. "
            "Use a value such as 255,0,0."
        )

    parts = [p.strip() for p in str(custom_rgb).split(",")]
    if len(parts) != 3:
        raise ValueError(
            "Custom RGB must contain exactly three comma-separated values, "
            "for example: 255,0,0"
        )

    try:
        rgb = tuple(int(p) for p in parts)
    except Exception:
        raise ValueError("Custom RGB values must be integers from 0 to 255.")

    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("Custom RGB values must be between 0 and 255.")

    return rgb


def _find_preview_layer(m):
    matches = [
        lyr for lyr in m.listLayers()
        if lyr.name == PREVIEW_LAYER_NAME
    ]
    return matches[0] if matches else None


def _remove_all_preview_layers(m):
    for lyr in list(m.listLayers()):
        if lyr.name == PREVIEW_LAYER_NAME:
            try:
                m.removeLayer(lyr)
            except Exception:
                pass


def _ensure_preview_layer(m, raster_path):
    """
    Reuse the existing preview layer only if it points to the same probability
    raster. Otherwise remove it and add a fresh copy of the input raster.
    """
    existing = _find_preview_layer(m)

    if existing is not None:
        try:
            existing_source = str(existing.dataSource)
        except Exception:
            existing_source = ""

        try:
            same = (
                os.path.normcase(os.path.abspath(existing_source))
                ==
                os.path.normcase(os.path.abspath(raster_path))
            )
        except Exception:
            same = existing_source == raster_path

        if same:
            return existing

        try:
            m.removeLayer(existing)
        except Exception:
            pass

    layer = m.addDataFromPath(raster_path)
    layer.name = PREVIEW_LAYER_NAME
    return layer


def _style_probability_threshold(layer, threshold, rgb, opacity_percent):
    """
    Use a 2-class RasterClassifyColorizer:
        class 1: <= threshold -> fully transparent
        class 2: > threshold  -> selected color

    This only changes layer symbology. It does not create a new raster.
    """
    threshold = float(threshold)
    opacity_percent = max(0.0, min(100.0, float(opacity_percent)))
    alpha = int(round(opacity_percent))

    sym = layer.symbology

    if not hasattr(sym, "colorizer"):
        raise RuntimeError(
            "The preview layer does not expose raster colorizer symbology."
        )

    if sym.colorizer.type != "RasterClassifyColorizer":
        sym.updateColorizer("RasterClassifyColorizer")

    c = sym.colorizer
    c.classificationField = "Value"
    c.breakCount = 2

    try:
        c.classificationMethod = "ManualInterval"
    except Exception:
        pass

    breaks = c.classBreaks
    if len(breaks) < 2:
        raise RuntimeError(
            "ArcGIS could not create the two raster classes needed for the preview."
        )

    # <= threshold is invisible.
    breaks[0].upperBound = threshold
    breaks[0].label = f"<= {threshold:.3f} (hidden)"
    breaks[0].description = "Probability at or below threshold"
    breaks[0].color = {"RGB": [0, 0, 0, 0]}

    # > threshold is visible. Probability maps are expected to be in [0, 1].
    # Set a final upper bound of 1.0, or slightly higher if ArcGIS statistics
    # contain numerical overshoot.
    try:
        mx = float(
            arcpy.management.GetRasterProperties(
                layer.dataSource, "MAXIMUM"
            ).getOutput(0)
        )
    except Exception:
        mx = 1.0

    final_max = max(1.0, mx)
    breaks[1].upperBound = final_max
    breaks[1].label = f"> {threshold:.3f} (road)"
    breaks[1].description = "Probability above threshold"
    breaks[1].color = {"RGB": [rgb[0], rgb[1], rgb[2], alpha]}

    try:
        c.noDataColor = {"RGB": [0, 0, 0, 0]}
    except Exception:
        pass

    layer.symbology = sym


def _style_binary_layer(layer, rgb, opacity_percent):
    """
    Style a permanent 0/1 raster so background 0 is transparent and road 1
    uses the selected color.
    """
    opacity_percent = max(0.0, min(100.0, float(opacity_percent)))
    alpha = int(round(opacity_percent))

    sym = layer.symbology
    if not hasattr(sym, "colorizer"):
        return

    try:
        if sym.colorizer.type != "RasterUniqueValueColorizer":
            sym.updateColorizer("RasterUniqueValueColorizer")

        c = sym.colorizer
        c.field = "Value"

        # Apply once so ArcGIS has generated items.
        layer.symbology = sym
        sym = layer.symbology
        c = sym.colorizer

        for grp in c.groups:
            for itm in grp.items:
                vals = []
                try:
                    vals = itm.values
                except Exception:
                    vals = []

                flat = []
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        flat.extend(v)
                    else:
                        flat.append(v)

                numeric = None
                for v in flat:
                    try:
                        numeric = float(v)
                        break
                    except Exception:
                        pass

                if numeric is None:
                    try:
                        numeric = float(itm.label)
                    except Exception:
                        continue

                if abs(numeric - 0.0) < 1e-9:
                    itm.label = "Background"
                    itm.color = {"RGB": [0, 0, 0, 0]}
                elif abs(numeric - 1.0) < 1e-9:
                    itm.label = "Road"
                    itm.color = {"RGB": [rgb[0], rgb[1], rgb[2], alpha]}

        try:
            c.useDefaultSymbol = False
        except Exception:
            pass

        layer.symbology = sym

    except Exception as exc:
        arcpy.AddWarning(
            "Binary raster was created successfully, but automatic red/transparent "
            f"symbology could not be fully applied: {exc}"
        )


# ----------------------------------------------------------------------
# Toolbox
# ----------------------------------------------------------------------

class Toolbox(object):
    def __init__(self):
        self.label = TOOLBOX_NAME
        self.alias = "road_probability_threshold"
        self.tools = [
            LoadThresholdPreview,
            GenerateBinaryRoadMask,
        ]


# ----------------------------------------------------------------------
# Tool 1
# ----------------------------------------------------------------------

class LoadThresholdPreview(object):

    def __init__(self):
        self.label = "1 - Load / Refresh Threshold Preview"
        self.description = (
            "Load a road probability raster into the active map and display only "
            "pixels above a chosen probability threshold. The source raster is not "
            "modified and no threshold raster is written to disk."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):

        in_raster = arcpy.Parameter(
            displayName="Road Probability Raster",
            name="probability_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        threshold = arcpy.Parameter(
            displayName="Probability Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        threshold.value = 0.50
        threshold.filter.type = "Range"
        threshold.filter.list = [0.0, 1.0]

        color = arcpy.Parameter(
            displayName="Road Preview Color",
            name="preview_color",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        color.filter.type = "ValueList"
        color.filter.list = [
            "Red",
            "Orange",
            "Yellow",
            "Green",
            "Cyan",
            "Blue",
            "Magenta",
            "White",
            "Black",
            "Custom RGB",
        ]
        color.value = "Red"

        custom_rgb = arcpy.Parameter(
            displayName="Custom RGB (R,G,B)",
            name="custom_rgb",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        custom_rgb.value = "255,0,0"
        custom_rgb.enabled = False

        opacity = arcpy.Parameter(
            displayName="Visible Road Opacity (%)",
            name="opacity_percent",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        opacity.value = 100.0
        opacity.filter.type = "Range"
        opacity.filter.list = [0.0, 100.0]

        remove_original = arcpy.Parameter(
            displayName="Turn Off Original Probability Layer",
            name="turn_off_original",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        remove_original.value = True

        return [
            in_raster,
            threshold,
            color,
            custom_rgb,
            opacity,
            remove_original,
        ]

    def updateParameters(self, parameters):
        parameters[3].enabled = (
            parameters[2].valueAsText == "Custom RGB"
        )
        return

    def updateMessages(self, parameters):
        if parameters[1].value is not None:
            t = float(parameters[1].value)
            if t < 0.0 or t > 1.0:
                parameters[1].setErrorMessage(
                    "Probability threshold must be between 0 and 1."
                )

        # Python toolboxes have no native drag-slider control.
        if parameters[1].altered:
            parameters[1].setWarningMessage(
                "Change the threshold and click Run to refresh the map preview. "
                "Standard Python toolboxes do not provide a true continuously "
                "dragged live slider."
            )
        return

    def execute(self, parameters, messages):

        source = _resolve_raster_path(parameters[0].value)
        threshold = float(parameters[1].value)
        color_name = parameters[2].valueAsText
        custom_rgb = parameters[3].valueAsText
        opacity = float(parameters[4].value)
        turn_off_original = bool(parameters[5].value)

        rgb = _parse_rgb(color_name, custom_rgb)
        _, mn, mx = _validate_probability_raster(source)

        aprx, m = _active_map()

        layer = _ensure_preview_layer(m, source)

        _style_probability_threshold(
            layer=layer,
            threshold=threshold,
            rgb=rgb,
            opacity_percent=opacity,
        )

        # Optionally hide another layer pointing to the same source so that
        # only the red threshold preview is visible.
        if turn_off_original:
            for lyr in m.listLayers():
                if lyr is layer:
                    continue
                try:
                    ds = str(lyr.dataSource)
                except Exception:
                    continue
                try:
                    same = (
                        os.path.normcase(os.path.abspath(ds))
                        ==
                        os.path.normcase(os.path.abspath(source))
                    )
                except Exception:
                    same = ds == source
                if same:
                    try:
                        lyr.visible = False
                    except Exception:
                        pass

        try:
            layer.visible = True
        except Exception:
            pass

        arcpy.AddMessage("=" * 70)
        arcpy.AddMessage("ROAD PROBABILITY THRESHOLD PREVIEW")
        arcpy.AddMessage("=" * 70)
        arcpy.AddMessage(f"Probability raster: {source}")
        arcpy.AddMessage(f"Threshold: > {threshold:.3f}")
        arcpy.AddMessage(
            f"Visible color: {color_name} RGB{rgb}, opacity {opacity:.0f}%"
        )
        if mn is not None and mx is not None:
            arcpy.AddMessage(
                f"Raster statistics: minimum={mn:.6f}, maximum={mx:.6f}"
            )
            if mn < -1e-6 or mx > 1.000001:
                arcpy.AddWarning(
                    "The raster statistics extend outside [0, 1]. "
                    "This tool assumes the raster stores probabilities."
                )
        arcpy.AddMessage(
            "No new threshold raster was written. The preview is symbology only."
        )
        arcpy.AddMessage(
            "To test another threshold, change the value and run this tool again."
        )
        arcpy.AddMessage("=" * 70)


# ----------------------------------------------------------------------
# Tool 2
# ----------------------------------------------------------------------

class GenerateBinaryRoadMask(object):

    def __init__(self):
        self.label = "2 - Generate Binary Road Mask"
        self.description = (
            "Create a permanent 0/1 binary road raster from a probability map. "
            "Pixels greater than the threshold become 1; all other valid pixels "
            "become 0."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):

        in_raster = arcpy.Parameter(
            displayName="Road Probability Raster",
            name="probability_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        threshold = arcpy.Parameter(
            displayName="Probability Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        threshold.value = 0.50
        threshold.filter.type = "Range"
        threshold.filter.list = [0.0, 1.0]

        out_raster = arcpy.Parameter(
            displayName="Output Binary Road Raster",
            name="output_binary_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Output",
        )

        color = arcpy.Parameter(
            displayName="Road Display Color",
            name="road_color",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        color.filter.type = "ValueList"
        color.filter.list = [
            "Red",
            "Orange",
            "Yellow",
            "Green",
            "Cyan",
            "Blue",
            "Magenta",
            "White",
            "Black",
            "Custom RGB",
        ]
        color.value = "Red"

        custom_rgb = arcpy.Parameter(
            displayName="Custom RGB (R,G,B)",
            name="custom_rgb",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        custom_rgb.value = "255,0,0"
        custom_rgb.enabled = False

        opacity = arcpy.Parameter(
            displayName="Road Display Opacity (%)",
            name="opacity_percent",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input",
        )
        opacity.value = 100.0
        opacity.filter.type = "Range"
        opacity.filter.list = [0.0, 100.0]

        add_to_map = arcpy.Parameter(
            displayName="Add Binary Raster to Current Map",
            name="add_to_map",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        add_to_map.value = True

        return [
            in_raster,
            threshold,
            out_raster,
            color,
            custom_rgb,
            opacity,
            add_to_map,
        ]

    def updateParameters(self, parameters):
        parameters[4].enabled = (
            parameters[3].valueAsText == "Custom RGB"
        )

        # Suggest an output filename next to the source when a source path is
        # already available and the user has not chosen an output.
        if (
            parameters[0].valueAsText
            and
            not parameters[2].altered
        ):
            try:
                source = _resolve_raster_path(parameters[0].value)
                src = Path(source)
                if src.suffix.lower() in (".tif", ".tiff"):
                    parameters[2].value = str(
                        src.with_name(
                            src.stem + "_binary_threshold.tif"
                        )
                    )
            except Exception:
                pass

        return

    def updateMessages(self, parameters):
        if parameters[1].value is not None:
            t = float(parameters[1].value)
            if t < 0.0 or t > 1.0:
                parameters[1].setErrorMessage(
                    "Probability threshold must be between 0 and 1."
                )
        return

    def execute(self, parameters, messages):

        source = _resolve_raster_path(parameters[0].value)
        threshold = float(parameters[1].value)
        output = parameters[2].valueAsText
        color_name = parameters[3].valueAsText
        custom_rgb = parameters[4].valueAsText
        opacity = float(parameters[5].value)
        add_to_map = bool(parameters[6].value)

        rgb = _parse_rgb(color_name, custom_rgb)
        source_raster, mn, mx = _validate_probability_raster(source)

        arcpy.env.overwriteOutput = True

        if arcpy.Exists(output):
            arcpy.management.Delete(output)

        arcpy.AddMessage("=" * 70)
        arcpy.AddMessage("GENERATING BINARY ROAD MASK")
        arcpy.AddMessage("=" * 70)
        arcpy.AddMessage(f"Probability raster: {source}")
        arcpy.AddMessage(f"Threshold rule: probability > {threshold:.3f}")
        arcpy.AddMessage("Output coding: 0 = background, 1 = road")
        arcpy.AddMessage(f"Output: {output}")

        # Convert valid probability pixels to strict 0/1.
        # IMPORTANT: do not use None as a Con() branch. ArcPy Con expects a
        # raster/layer or numeric constant there and raises
        # "TypeError: expected a raster or layer name" when given None.
        #
        # NoData is already preserved automatically: where source_raster is
        # NoData, the comparison (source_raster > threshold) is NoData, and
        # Con propagates that NoData to the output.
        binary = arcpy.sa.Con(
            source_raster > threshold,
            1,
            0,
        )

        binary.save(output)

        # Ensure a compact unsigned-byte output for TIFFs where possible.
        # CopyRaster is only used if the saved output is not already U8.
        try:
            out_desc = arcpy.Describe(output)
            # Calculate statistics and attribute table for clean ArcGIS behavior.
            arcpy.management.CalculateStatistics(
                output,
                x_skip_factor=1,
                y_skip_factor=1,
                skip_existing="OVERWRITE",
            )
        except Exception as exc:
            arcpy.AddWarning(f"CalculateStatistics warning: {exc}")

        try:
            arcpy.management.BuildRasterAttributeTable(
                output,
                overwrite="Overwrite",
            )
        except Exception as exc:
            arcpy.AddWarning(f"Raster attribute table warning: {exc}")

        if add_to_map:
            _, m = _active_map()

            # Remove an already-added copy of the same output if present.
            for lyr in list(m.listLayers()):
                try:
                    ds = str(lyr.dataSource)
                except Exception:
                    continue

                try:
                    same = (
                        os.path.normcase(os.path.abspath(ds))
                        ==
                        os.path.normcase(os.path.abspath(output))
                    )
                except Exception:
                    same = ds == output

                if same:
                    try:
                        m.removeLayer(lyr)
                    except Exception:
                        pass

            out_layer = m.addDataFromPath(output)
            out_layer.name = (
                f"Road Mask > {threshold:.3f}"
            )

            _style_binary_layer(
                out_layer,
                rgb=rgb,
                opacity_percent=opacity,
            )

        arcpy.AddMessage("Binary road mask created successfully.")
        arcpy.AddMessage("=" * 70)
