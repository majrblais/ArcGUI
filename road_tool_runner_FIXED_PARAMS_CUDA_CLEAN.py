# =========================================================
# ARC GIS NOTEBOOK / SCRIPT - FULL SINGLE BLOCK PIPELINE
# EXPORT TRAINING DATA DIRECTLY ON EXTENT -> RUN ONNX -> MOSAIC
# Optional buffered export extent -> clip final prediction back to original extent
# FUNCTION VERSION FOR TOOLBOX USE
# =========================================================

from pathlib import Path
import os
import sys
import glob
import re

# =========================================================
# CUDA / ONNX DLL FIX
# This must run before importing onnxruntime.
# It allows ArcGIS/Conda environments to find CUDA/cuDNN DLLs
# installed through pip packages.
# =========================================================
_ENV = sys.prefix
_NVIDIA_ROOT = Path(_ENV) / "Lib" / "site-packages" / "nvidia"

for _dll_dir in glob.glob(str(_NVIDIA_ROOT / "*" / "bin")):
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)
        os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

_LIBRARY_BIN = Path(_ENV) / "Library" / "bin"
if _LIBRARY_BIN.exists():
    os.add_dll_directory(str(_LIBRARY_BIN))
    os.environ["PATH"] = str(_LIBRARY_BIN) + os.pathsep + os.environ.get("PATH", "")

import arcpy
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from arcpy.ia import ExportTrainingDataForDeepLearning
from arcpy.sa import SetNull

# =========================================================
# FIXED MODEL / EXPORT SETTINGS
# These values are intentionally fixed because they must match
# the trained ONNX model and deployment resolution.
# =========================================================
FIXED_TILE_SIZE = 1024
FIXED_CELL_SIZE = 2.0
FIXED_MODEL_INPUT_SIZE = 1024
SAVE_INTERMEDIATE_PNG = False


def _next_available_path(base_path):
    """Return base_path, or base_path_1, base_path_2, ... until a free path is found."""
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path

    parent = base_path.parent
    stem = base_path.name
    for i in range(1, 100000):
        candidate = parent / f"{stem}_{i}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find available folder name for: {base_path}")


def _safe_name(name):
    """Create a Windows-safe name from an ArcGIS layer name."""
    name = str(name).strip().strip("'\"")
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "extent"


def _layer_display_name(extent_layer):
    """Return a clean output prefix based on the original input extent layer name."""
    try:
        return _safe_name(arcpy.Describe(extent_layer).name)
    except Exception:
        return _safe_name(extent_layer)


def _sigmoid(x):
    # Stable sigmoid to avoid overflow warnings on large logits.
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _build_onnx_session(model_path):
    available = ort.get_available_providers()
    preferred = []

    # Prefer GPU when available, then fall back to CPU.
    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        preferred.append("CPUExecutionProvider")

    if not preferred:
        raise RuntimeError(
            "No ONNX Runtime execution provider is available. "
            "Make sure onnxruntime or onnxruntime-gpu is installed."
        )

    session = ort.InferenceSession(str(model_path), providers=preferred)

    # active = providers actually used by this ONNX session.
    active = session.get_providers()
    cuda_active = "CUDAExecutionProvider" in active

    print(f"Active session providers: {active}")
    if cuda_active:
        print("RESULT: CUDA / GPU IS BEING USED FOR ONNX INFERENCE.")
    else:
        print("RESULT: CUDA / GPU IS NOT ACTIVE. ONNX INFERENCE IS USING CPU.")

    return session, preferred, active, cuda_active


def _prepare_input_rgb_imagenet_norm(image_rgb, model_input_size):
    """
    Match the training preprocessing exactly.

    Training used Albumentations Normalize with:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        max_pixel_value=255.0
    """
    resized = cv2.resize(
        image_rgb,
        (model_input_size, model_input_size),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    resized = resized / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    resized = (resized - mean) / std

    tensor = resized.transpose(2, 0, 1).astype(np.float32)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor



def _extract_binary_predictions_batch(raw_output, threshold):
    """Convert batched ONNX output logits/probabilities into binary masks [B, H, W]."""
    pred = np.asarray(raw_output)

    # Expected common shape: [B, 1, H, W].
    if pred.ndim == 4:
        if pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        elif pred.shape[-1] == 1:
            pred = pred[:, :, :, 0]
        else:
            raise RuntimeError(f"Unexpected batched ONNX output shape: {np.asarray(raw_output).shape}")
    elif pred.ndim == 3:
        # Already [B, H, W].
        pass
    elif pred.ndim == 2:
        # Single prediction fallback [H, W] -> [1, H, W].
        pred = np.expand_dims(pred, axis=0)
    else:
        raise RuntimeError(f"Unexpected ONNX output shape: {np.asarray(raw_output).shape}")

    pred = _sigmoid(pred)
    pred = (pred > float(threshold)).astype(np.uint8)
    return pred


def _chunks(items, chunk_size):
    """Yield successive chunks with the last chunk allowed to be smaller."""
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _bool_from_tool_value(value):
    """Handle ArcGIS GPBoolean values passed as bool/string."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def run_road_tool(
    extent_layer,
    input_raster_name,
    output_root,
    model_path,
    stride_x,
    stride_y,
    threshold,
    use_extent_buffer=False,
    buffer_pixels=100,
    batch_size=4,
    output_name_prefix=None,
    final_predictions_dir=None,
    add_output_to_map=True,
):
    """
    Run the road segmentation pipeline.

    New edge-handling option
    ------------------------
    If use_extent_buffer=True, the ETD4DL export extent is expanded by
    buffer_pixels * cell_size in map units on all sides. The final mosaic is then
    clipped back to the original extent layer. This helps avoid bad predictions
    caused by tiles sitting exactly on the processing boundary.

    batch_size controls ONNX inference batch size. The last batch is allowed to be
    smaller, e.g. 7 tiles with batch_size=4 runs as 4 + 3.
    """
    # ---------------------------------------------------------
    # DERIVED VARIABLES
    # ---------------------------------------------------------
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if output_name_prefix is None:
        output_name_prefix = _layer_display_name(extent_layer)
    output_name_prefix = _safe_name(output_name_prefix)

    if final_predictions_dir is not None:
        final_predictions_dir = Path(final_predictions_dir)
        final_predictions_dir.mkdir(parents=True, exist_ok=True)

    export_root = _next_available_path(output_root / "export_all_tiles")
    export_img_dir = export_root / "images"
    pred_mask_dir = _next_available_path(output_root / "predictions_png") if SAVE_INTERMEDIATE_PNG else None
    pred_raster_dir = _next_available_path(output_root / "predictions_raster")
    final_dir = _next_available_path(output_root / "final")

    if SAVE_INTERMEDIATE_PNG:
        pred_mask_dir.mkdir(parents=True, exist_ok=True)
    pred_raster_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    final_raster = final_dir / "roads_prediction_buffered_mosaic.tif"
    final_clipped_raster = final_dir / "roads_prediction_clipped_to_extent.tif"
    final_clean_raster = final_dir / f"{output_name_prefix}_roads_prediction_clean.tif"

    tile_size = FIXED_TILE_SIZE
    stride_x = int(stride_x)
    stride_y = int(stride_y)
    cell_size = FIXED_CELL_SIZE
    model_input_size = FIXED_MODEL_INPUT_SIZE
    threshold = float(threshold)
    use_extent_buffer = _bool_from_tool_value(use_extent_buffer)
    buffer_pixels = int(buffer_pixels)
    batch_size = int(batch_size)

    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("Stride X and Stride Y must both be > 0.")
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if buffer_pixels < 0:
        raise ValueError("Buffer pixels must be >= 0.")
    if batch_size <= 0:
        raise ValueError("Batch size must be > 0.")

    # ---------------------------------------------------------
    # BASIC SETUP
    # ---------------------------------------------------------
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster = None
    arcpy.env.extent = None
    arcpy.env.mask = None
    arcpy.env.cellSize = None
    arcpy.env.outputCoordinateSystem = None

    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")

    print("ArcPy version:", arcpy.GetInstallInfo()["Version"])
    print("Extent layer:", extent_layer)
    print("Input raster name:", input_raster_name)
    print("Output root:", output_root)
    print("Output prefix:", output_name_prefix)
    print("ONNX model path:", model_path)
    print("Expected model: Unet + resnet101 + ImageNet normalization")
    print("Tile size (fixed):", tile_size)
    print("Stride X:", stride_x)
    print("Stride Y:", stride_y)
    print("Cell size (fixed):", cell_size)
    print("Model input size (fixed):", model_input_size)
    print("Threshold:", threshold)
    print("Use extent buffer:", use_extent_buffer)
    print("Buffer pixels:", buffer_pixels)
    print("ONNX batch size:", batch_size)
    print("Export root:", export_root)

    # ---------------------------------------------------------
    # VALIDATE EXTENT LAYER
    # ---------------------------------------------------------
    if not arcpy.Exists(extent_layer):
        raise ValueError(f"Extent layer not found: {extent_layer}")

    desc_extent = arcpy.Describe(extent_layer)
    extent_fc = desc_extent.catalogPath
    spatial_ref = desc_extent.spatialReference
    original_extent = desc_extent.extent

    print("Extent datasource:", extent_fc)
    print("Original extent:", original_extent.XMin, original_extent.YMin, original_extent.XMax, original_extent.YMax)

    # ---------------------------------------------------------
    # RESOLVE INPUT RASTER FROM CURRENT MAP
    # ---------------------------------------------------------
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    m = aprx.activeMap
    if m is None:
        raise RuntimeError("No active map found in ArcGIS Pro.")

    layers = m.listLayers(input_raster_name)
    if not layers:
        print("\nAvailable layers in current map:")
        for lyr in m.listLayers():
            print(" -", lyr.name)
        raise RuntimeError(
            f"Could not find layer named '{input_raster_name}' in the active map."
        )

    input_raster = layers[0]
    print("Using input raster layer:", input_raster.name)

    # ---------------------------------------------------------
    # LOAD ONNX MODEL
    # ---------------------------------------------------------
    print("\nLoading ONNX model...")
    session, requested_providers, active_providers, cuda_active = _build_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("ONNX input name:", input_name)
    print("ONNX output name:", output_name)
    # ---------------------------------------------------------
    # STAGE 1 - EXPORT ALL TILES DIRECTLY FROM EXTENT
    # ---------------------------------------------------------
    print("\n===================================")
    print("STAGE 1 - EXPORT ALL TILES DIRECTLY FROM EXTENT")
    print("===================================")

    if use_extent_buffer and buffer_pixels > 0:
        buffer_map_units = buffer_pixels * cell_size
        export_extent = arcpy.Extent(
            original_extent.XMin - buffer_map_units,
            original_extent.YMin - buffer_map_units,
            original_extent.XMax + buffer_map_units,
            original_extent.YMax + buffer_map_units,
        )
        print("Buffered export enabled.")
        print("Buffer distance:", buffer_map_units, "map units")
    else:
        buffer_map_units = 0.0
        export_extent = original_extent
        print("Buffered export disabled.")

    export_extent_str = (
        f"{export_extent.XMin} {export_extent.YMin} "
        f"{export_extent.XMax} {export_extent.YMax}"
    )
    print("Export extent:", export_extent_str)

    with arcpy.EnvManager(
        extent=export_extent_str,
        cellSize=cell_size,
        snapRaster=None,
        mask=None,
        outputCoordinateSystem=spatial_ref,
    ):
        ExportTrainingDataForDeepLearning(
            in_raster=input_raster,
            out_folder=str(export_root),
            image_chip_format="TIFF",
            tile_size_x=tile_size,
            tile_size_y=tile_size,
            stride_x=stride_x,
            stride_y=stride_y,
            output_nofeature_tiles="ALL_TILES",
            metadata_format="Export_Tiles",
            start_index=0,
        )

    if not export_img_dir.exists():
        raise RuntimeError(f"ETD4DL did not create image folder: {export_img_dir}")

    exported_tifs = sorted(export_img_dir.glob("*.tif"))
    print("Exported TIFF count:", len(exported_tifs))

    if len(exported_tifs) == 0:
        raise RuntimeError("ETD4DL exported no images.")

    tile_info = []
    for idx, tif in enumerate(exported_tifs, start=1):
        raster_obj = arcpy.Raster(str(tif))
        ext = raster_obj.extent
        tile_info.append({
            "uid": tif.stem,
            "tile_id": idx,
            "tile_name": tif.stem,
            "extent": (ext.XMin, ext.YMin, ext.XMax, ext.YMax),
            "cell_w": raster_obj.meanCellWidth,
            "cell_h": raster_obj.meanCellHeight,
            "spatial_ref": raster_obj.spatialReference,
            "image_path": tif,
        })

    print("Tiles loaded from ETD4DL export:", len(tile_info))
    print("Export image folder:", export_img_dir)

    # ---------------------------------------------------------
    # STAGE 2 - RUN PREDICTION ON EXPORTED TILES
    # ---------------------------------------------------------
    pred_rasters = []
    pred_success = 0
    pred_fail = 0

    print("\n===================================")
    print("STAGE 2 - RUN PREDICTION")
    print("===================================")

    total_tiles = len(tile_info)
    batch_index = 0

    for batch_items in _chunks(tile_info, batch_size):
        batch_index += 1
        batch_count = len(batch_items)
        first_i = (batch_index - 1) * batch_size + 1
        last_i = first_i + batch_count - 1
        print(f"\nRunning batch {batch_index}: tiles {first_i}-{last_i} of {total_tiles} ({batch_count} tile(s))")

        prepared_items = []
        batch_tensors = []

        # Load and preprocess all tiles in this batch. If one tile fails to load,
        # it is counted as failed but the rest of the batch still runs.
        for item in batch_items:
            uid = item["uid"]
            tile_name = item["tile_name"]
            tif = item["image_path"]

            try:
                image = np.array(Image.open(tif).convert("RGB"))
                orig_h, orig_w = image.shape[:2]
                onnx_input = _prepare_input_rgb_imagenet_norm(image, model_input_size)

                prepared_items.append({
                    "item": item,
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                    "image": image,
                })
                batch_tensors.append(onnx_input)

            except Exception as e:
                print(f"PREP FAILED for {tile_name}: {e}")
                pred_fail += 1

        if not batch_tensors:
            print("Skipping batch because no tile was prepared successfully.")
            continue

        try:
            batch_input = np.concatenate(batch_tensors, axis=0).astype(np.float32)
            print("Batch input shape:", batch_input.shape)

            raw_output = session.run([output_name], {input_name: batch_input})[0]
            batch_preds = _extract_binary_predictions_batch(raw_output, threshold)

            if batch_preds.shape[0] != len(prepared_items):
                raise RuntimeError(
                    f"Batch output count mismatch: got {batch_preds.shape[0]}, "
                    f"expected {len(prepared_items)}"
                )

        except Exception as e:
            print("BATCH INFERENCE FAILED:", e)
            print("Falling back to single-tile inference for this batch.")
            batch_preds = []
            fallback_items = []

            for prepared in prepared_items:
                item = prepared["item"]
                tile_name = item["tile_name"]
                try:
                    single_input = _prepare_input_rgb_imagenet_norm(
                        prepared["image"],
                        model_input_size,
                    )
                    raw_output = session.run([output_name], {input_name: single_input})[0]
                    single_pred = _extract_binary_predictions_batch(raw_output, threshold)[0]
                    batch_preds.append(single_pred)
                    fallback_items.append(prepared)
                except Exception as single_e:
                    print(f"SINGLE-TILE FALLBACK FAILED for {tile_name}: {single_e}")
                    pred_fail += 1

            prepared_items = fallback_items
            if not batch_preds:
                continue
            batch_preds = np.stack(batch_preds, axis=0)

        # Save each prediction in the batch using its own original extent/georeference.
        for pred, prepared in zip(batch_preds, prepared_items):
            item = prepared["item"]
            uid = item["uid"]
            tile_name = item["tile_name"]
            xmin, ymin, xmax, ymax = item["extent"]
            cell_w = item["cell_w"]
            cell_h = item["cell_h"]
            tile_spatial_ref = item["spatial_ref"]
            orig_h = prepared["orig_h"]
            orig_w = prepared["orig_w"]

            print(f"Saving {tile_name} -> {uid}.tif")

            try:
                pred_big = cv2.resize(
                    pred.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                if SAVE_INTERMEDIATE_PNG:
                    out_png = pred_mask_dir / f"{uid}.png"
                    Image.fromarray(pred_big.astype(np.uint8) * 255).save(out_png)

                raster = arcpy.NumPyArrayToRaster(
                    pred_big.astype(np.uint8) * 255,
                    arcpy.Point(xmin, ymin),
                    cell_w,
                    cell_h,
                    value_to_nodata=0,
                )

                out_raster = str(pred_raster_dir / f"{uid}.tif")
                raster.save(out_raster)

                try:
                    arcpy.management.DefineProjection(out_raster, tile_spatial_ref)
                except Exception as e:
                    print("Warning: could not define projection for", out_raster, "->", e)

                pred_rasters.append(out_raster)
                pred_success += 1

            except Exception as e:
                print(f"SAVE FAILED for {tile_name}: {e}")
                pred_fail += 1

    print("\nPrediction complete.")
    print("Prediction success:", pred_success)
    print("Prediction fail:", pred_fail)
    if SAVE_INTERMEDIATE_PNG:
        print("Prediction PNG folder:", pred_mask_dir)
    print("Prediction raster folder:", pred_raster_dir)

    if len(pred_rasters) == 0:
        raise RuntimeError("No prediction rasters created.")

    # ---------------------------------------------------------
    # STAGE 3 - MOSAIC GEOREFERENCED PREDICTION RASTERS
    # ---------------------------------------------------------
    print("\n===================================")
    print("STAGE 3 - FAST MOSAIC")
    print("===================================")

    final_raster_str = str(final_raster)
    final_clipped_raster_str = str(final_clipped_raster)
    final_clean_raster_str = str(final_clean_raster)

    for old_path in [final_raster_str, final_clipped_raster_str, final_clean_raster_str]:
        if arcpy.Exists(old_path):
            arcpy.management.Delete(old_path)

    first_pred = arcpy.Raster(pred_rasters[0])
    first_spref = first_pred.spatialReference
    first_cell = first_pred.meanCellWidth

    input_rasters = ";".join(pred_rasters)

    arcpy.management.MosaicToNewRaster(
        input_rasters=input_rasters,
        output_location=str(final_dir),
        raster_dataset_name_with_extension=final_raster.name,
        coordinate_system_for_the_raster=first_spref,
        pixel_type="8_BIT_UNSIGNED",
        cellsize=first_cell,
        number_of_bands=1,
        mosaic_method="MAXIMUM",
        mosaic_colormap_mode="FIRST",
    )

    print("Buffered/raw mosaic raster created:", final_raster)

    try:
        arcpy.management.CalculateStatistics(final_raster_str)
        arcpy.management.BuildPyramids(final_raster_str)
        print("Statistics and pyramids built for raw mosaic.")
    except Exception as e:
        print("Warning while building raw mosaic statistics/pyramids:", e)

    # ---------------------------------------------------------
    # STAGE 4 - CLIP FINAL MOSAIC BACK TO ORIGINAL EXTENT LAYER
    # ---------------------------------------------------------
    print("\n===================================")
    print("STAGE 4 - CLIP FINAL MOSAIC BACK TO ORIGINAL EXTENT")
    print("===================================")

    original_extent_str = (
        f"{original_extent.XMin} {original_extent.YMin} "
        f"{original_extent.XMax} {original_extent.YMax}"
    )

    # Clip to the original polygon geometry, not just the bounding box.
    # The rectangle is still required by the tool, but the extent layer is used as the clipping geometry.
    arcpy.management.Clip(
        in_raster=final_raster_str,
        rectangle=original_extent_str,
        out_raster=final_clipped_raster_str,
        in_template_dataset=extent_fc,
        nodata_value="0",
        clipping_geometry="ClippingGeometry",
        maintain_clipping_extent="MAINTAIN_EXTENT",
    )

    arcpy.management.DefineProjection(final_clipped_raster_str, first_spref)
    print("Clipped raster created:", final_clipped_raster)

    try:
        arcpy.management.CalculateStatistics(final_clipped_raster_str)
        arcpy.management.BuildPyramids(final_clipped_raster_str)
        print("Statistics and pyramids built for clipped raster.")
    except Exception as e:
        print("Warning while building clipped raster statistics/pyramids:", e)

    # ---------------------------------------------------------
    # STAGE 5 - CLEAN NODATA
    # ---------------------------------------------------------
    print("\n===================================")
    print("STAGE 5 - CLEAN NODATA")
    print("===================================")

    SetNull(final_clipped_raster_str, final_clipped_raster_str, "VALUE = 0").save(final_clean_raster_str)
    arcpy.management.DefineProjection(final_clean_raster_str, first_spref)

    print("Clean clipped raster created:", final_clean_raster)

    # ---------------------------------------------------------
    # COPY FINAL CLEAN RASTER TO SHARED FINAL PREDICTIONS FOLDER
    # ---------------------------------------------------------
    copied_final_clean = None

    if final_predictions_dir is not None:
        copied_final_clean = final_predictions_dir / final_clean_raster.name
        copied_final_clean_str = str(copied_final_clean)

        if arcpy.Exists(copied_final_clean_str):
            arcpy.management.Delete(copied_final_clean_str)

        arcpy.management.CopyRaster(final_clean_raster_str, copied_final_clean_str)
        arcpy.management.DefineProjection(copied_final_clean_str, first_spref)
        print("Copied clean prediction to:", copied_final_clean)

    # ---------------------------------------------------------
    # OPTIONAL - ADD ONLY THE CLEAN FINAL RESULT TO MAP
    # ---------------------------------------------------------
    if add_output_to_map:
        try:
            map_output = str(copied_final_clean) if copied_final_clean is not None else final_clean_raster_str
            m.addDataFromPath(map_output)
        except Exception as e:
            print("Warning: could not add final clean output to map:", e)

    # ---------------------------------------------------------
    # DONE
    # ---------------------------------------------------------
    print("\n===================================")
    print("DONE")
    print("===================================")
    print("Export success:", len(tile_info))
    print("Prediction success:", pred_success)
    print("Prediction fail:", pred_fail)
    print("Total prediction rasters:", len(pred_rasters))
    print("Raw buffered mosaic:", final_raster)
    print("Clipped raster:", final_clipped_raster)
    print("Final clean raster:", final_clean_raster)
    if copied_final_clean is not None:
        print("Final clean copy:", copied_final_clean)

    return str(copied_final_clean) if copied_final_clean is not None else final_clean_raster_str


def run_road_tool_multiple(
    extent_layers,
    input_raster_name,
    output_root,
    model_path,
    stride_x,
    stride_y,
    threshold,
    use_extent_buffer=False,
    buffer_pixels=100,
    batch_size=4,
):
    """Run the road segmentation pipeline on multiple extent layers one by one."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    final_predictions_dir = output_root / "final_predictions"
    final_predictions_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    failures = []

    for idx, extent_layer in enumerate(extent_layers, start=1):
        extent_name = _layer_display_name(extent_layer)
        extent_output_root = output_root / extent_name

        print("\n===================================")
        print(f"PROCESSING EXTENT {idx}/{len(extent_layers)}")
        print("===================================")
        print("Extent layer:", extent_layer)
        print("Extent output folder:", extent_output_root)

        try:
            output = run_road_tool(
                extent_layer=extent_layer,
                input_raster_name=input_raster_name,
                output_root=extent_output_root,
                model_path=model_path,
                stride_x=stride_x,
                stride_y=stride_y,
                threshold=threshold,
                use_extent_buffer=use_extent_buffer,
                buffer_pixels=buffer_pixels,
                batch_size=batch_size,
                output_name_prefix=extent_name,
                final_predictions_dir=final_predictions_dir,
                add_output_to_map=True,
            )
            outputs.append(output)
        except Exception as e:
            print("FAILED EXTENT:", extent_layer)
            print(e)
            failures.append((extent_layer, str(e)))

    print("\n===================================")
    print("MULTI-EXTENT SUMMARY")
    print("===================================")
    print("Total extents:", len(extent_layers))
    print("Successful:", len(outputs))
    print("Failed:", len(failures))
    print("Final predictions folder:", final_predictions_dir)

    if failures:
        print("\nFailed extents:")
        for layer, error in failures:
            print(" -", layer, "->", error)

    return outputs, failures


if __name__ == "__main__":
    run_road_tool(
        extent_layer="extent_file",
        input_raster_name="World Imagery",
        output_root=r"C:\temp\arcgis_roads_notebook3",
        model_path=r"E:\Desktop\ccnb\Project_files\best_unet_resnet101_norm_1024.onnx",
        stride_x=1024,
        stride_y=1024,
        threshold=0.5,
        use_extent_buffer=True,
        buffer_pixels=100,
        batch_size=4,
    )
