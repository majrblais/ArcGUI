# -*- coding: utf-8 -*-

import arcpy
import os
import sys


class Toolbox(object):
    def __init__(self):
        self.label = "Road Segmentation Toolbox (ONNX)"
        self.alias = "roadseg_onnx"
        self.tools = [RunRoadSegmentationONNX]


class RunRoadSegmentationONNX(object):
    def __init__(self):
        self.label = "Run Road Segmentation (ONNX)"
        self.description = (
            "Export fixed 1024 x 1024 tiles at 2.0 m cell size, run ONNX road "
            "segmentation, and mosaic the results. Stride remains configurable."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = []

        p0 = arcpy.Parameter(
            displayName="Extent Layer(s)",
            name="extent_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        p0.multiValue = True
        p0.description = (
            "One or more polygon layers defining the areas to process. "
            "If multiple extent layers are selected, they are processed one by one."
        )
        p0.category = "Input Data"
        params.append(p0)

        p1 = arcpy.Parameter(
            displayName="Input Raster Layer Name",
            name="input_raster_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        p1.value = "World Imagery"
        p1.description = "Name of the raster layer in the current ArcGIS Pro map."
        p1.category = "Input Data"
        params.append(p1)

        p2 = arcpy.Parameter(
            displayName="Output Folder",
            name="output_root",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )
        p2.description = "Folder where exported tiles, predictions and final outputs are saved."
        p2.category = "Input Data"
        params.append(p2)

        p3 = arcpy.Parameter(
            displayName="Model File (.onnx)",
            name="model_path",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        p3.filter.list = ["onnx"]
        p3.description = "Path to the ONNX road segmentation model."
        p3.category = "Model Settings"
        params.append(p3)

        p4 = arcpy.Parameter(
            displayName="Stride X",
            name="stride_x",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p4.value = 1024
        p4.description = (
            "Horizontal step between exported tiles, in pixels. "
            "Use 1024 for no overlap, or a smaller value for overlap."
        )
        p4.category = "Tiling Settings"
        params.append(p4)

        p5 = arcpy.Parameter(
            displayName="Stride Y",
            name="stride_y",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        p5.value = 1024
        p5.description = (
            "Vertical step between exported tiles, in pixels. "
            "Use 1024 for no overlap, or a smaller value for overlap."
        )
        p5.category = "Tiling Settings"
        params.append(p5)

        p6 = arcpy.Parameter(
            displayName="Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        p6.value = 0.5
        p6.description = "Binary threshold applied after sigmoid on the ONNX output logits."
        p6.category = "Model Settings"
        params.append(p6)

        p7 = arcpy.Parameter(
            displayName="Use Buffered Export Extent",
            name="use_extent_buffer",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        p7.value = True
        p7.description = (
            "If checked, the export extent is expanded by Buffer Pixels * fixed cell size 2.0, "
            "then clipped back to the original extent."
        )
        p7.category = "Edge Handling"
        params.append(p7)

        p8 = arcpy.Parameter(
            displayName="Buffer Pixels",
            name="buffer_pixels",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p8.value = 100
        p8.description = "Number of pixels added around the extent when buffered export is enabled."
        p8.category = "Edge Handling"
        params.append(p8)

        p9 = arcpy.Parameter(
            displayName="ONNX Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )
        p9.value = 4
        p9.description = "Number of tiles predicted in one ONNX inference call."
        p9.category = "Model Settings"
        params.append(p9)

        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        try:
            use_extent_buffer = parameters[7].value
            if use_extent_buffer is not None:
                parameters[8].enabled = bool(use_extent_buffer)
        except Exception:
            pass
        return

    def updateMessages(self, parameters):
        try:
            stride_x = parameters[4].value
            stride_y = parameters[5].value
            threshold = parameters[6].value
            buffer_pixels = parameters[8].value
            batch_size = parameters[9].value

            if stride_x is not None and int(stride_x) <= 0:
                parameters[4].setErrorMessage("Stride X must be > 0.")

            if stride_y is not None and int(stride_y) <= 0:
                parameters[5].setErrorMessage("Stride Y must be > 0.")

            if threshold is not None and (float(threshold) < 0 or float(threshold) > 1):
                parameters[6].setErrorMessage("Threshold must be between 0 and 1.")

            if buffer_pixels is not None and int(buffer_pixels) < 0:
                parameters[8].setErrorMessage("Buffer Pixels must be >= 0.")

            if batch_size is not None and int(batch_size) <= 0:
                parameters[9].setErrorMessage("ONNX Batch Size must be > 0.")

        except Exception:
            pass
        return

    def execute(self, parameters, messages):
        tool_dir = os.path.dirname(__file__)
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)

        from roadsegmentation_python import run_road_tool_multiple

        extent_layer_text = parameters[0].valueAsText
        extent_layers = []
        for item in extent_layer_text.split(";"):
            item = item.strip().strip("'\"")
            if item:
                extent_layers.append(item)

        if not extent_layers:
            raise RuntimeError("No extent layer was provided.")

        input_raster_name = parameters[1].valueAsText
        output_root = parameters[2].valueAsText
        model_path = parameters[3].valueAsText
        stride_x = int(parameters[4].value)
        stride_y = int(parameters[5].value)
        threshold = float(parameters[6].value)
        use_extent_buffer = bool(parameters[7].value)
        buffer_pixels = int(parameters[8].value) if parameters[8].value is not None else 100
        batch_size = int(parameters[9].value) if parameters[9].value is not None else 4

        arcpy.AddMessage("Starting road segmentation tool (ONNX)...")
        arcpy.AddMessage("Fixed tile size: 1024 x 1024")
        arcpy.AddMessage("Fixed cell size: 2.0")
        arcpy.AddMessage("Fixed model input size: 1024")
        arcpy.AddMessage(f"Extent layers: {len(extent_layers)}")
        for lyr in extent_layers:
            arcpy.AddMessage(f" - {lyr}")
        arcpy.AddMessage(f"Input raster layer name: {input_raster_name}")
        arcpy.AddMessage(f"Output folder: {output_root}")
        arcpy.AddMessage(f"ONNX model path: {model_path}")
        arcpy.AddMessage(f"Stride X: {stride_x}")
        arcpy.AddMessage(f"Stride Y: {stride_y}")
        arcpy.AddMessage(f"Threshold: {threshold}")
        arcpy.AddMessage(f"Use buffered export extent: {use_extent_buffer}")
        arcpy.AddMessage(f"Buffer pixels: {buffer_pixels}")
        arcpy.AddMessage(f"ONNX batch size: {batch_size}")

        outputs, failures = run_road_tool_multiple(
            extent_layers=extent_layers,
            input_raster_name=input_raster_name,
            output_root=output_root,
            model_path=model_path,
            stride_x=stride_x,
            stride_y=stride_y,
            threshold=threshold,
            use_extent_buffer=use_extent_buffer,
            buffer_pixels=buffer_pixels,
            batch_size=batch_size,
        )

        arcpy.AddMessage(f"Successful extent layers: {len(outputs)}")
        arcpy.AddMessage(f"Failed extent layers: {len(failures)}")

        if failures:
            for layer_name, error in failures:
                arcpy.AddWarning(f"Failed {layer_name}: {error}")

        arcpy.AddMessage("Tool completed.")
