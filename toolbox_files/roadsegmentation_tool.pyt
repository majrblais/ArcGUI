# -*- coding: utf-8 -*-
# ============================================================
# ArcGUI v5.4 FINAL r7 - ArcGIS Pro Python Toolbox
# ============================================================
# Supported ONNX model families:
#   - SegFormer MiT-B5 filenames containing 'segformer'
#   - U-Net ResNet101 filenames containing 'unet' and 'resnet101'/'r101'
# Execution target is inferred from filename: CPU -> CPU only, CUDA/FP16 -> CUDA,
# otherwise CUDA is preferred with CPU fallback.
#
# Model input size is fixed at 1024x1024.
# The user selects overlap as a percentage.
#
# Optimized Large-Area Export:
#   ExportTrainingDataForDeepLearning extracts balanced,
#   NON-OVERLAPPING World Imagery blocks. Every block is
#   <=10240x10240 px. Candidate large blocks with zero intersection
#   with the requested extent are skipped before imagery
#   export. All large-block planning is performed in the active MAP_SPACE export CRS
#   so coordinates and TIFF georeferencing remain consistent. Python then creates overlapping 1024 model windows only
#   inside the retained, already-exported TIFFs.
#
# Extent policy:
#   r7 uses the exact selected polygon only.
#
# Reusable imagery cache:
#   matching extent/export settings reuse existing georeferenced TIFFs across models.
#   Model/threshold/batch changes do not require another World Imagery export.
#
# Debug Footprint Preview Only:
#   no imagery export and no inference; creates transparent
#   footprint layers showing the planned large blocks and crops.
#
# v5.4 hardened final r7:
#   validates CRS/geometry/model contracts, rejects undefined source CRS,
#   filters black duplicate imagery, preserves selections/definition queries,
#   and can retry optimized image-service exports using smaller non-overlapping blocks.
# ============================================================

from pathlib import Path
import importlib.util
import re
import json
import hashlib
import math
import os

import arcpy


SEGFORMER_CUDA_MODEL_FILENAME = "segformer_cuda_fp16.onnx"
SEGFORMER_CPU_MODEL_FILENAME = "segformer_cpu_fp32_optimized.onnx"
SEGFORMER_LAST_CUDA_MODEL_FILENAME = "segformer_last_cuda_fp16.onnx"
UNET_BEST_CUDA_MODEL_FILENAME = "best_unet_resnet101_norm_1024_cuda_fp16.onnx"
UNET_LAST_CUDA_MODEL_FILENAME = "last_unet_resnet101_norm_1024_cuda_fp16.onnx"
UNET_OLD_MILESTONE_MODEL_FILENAME = "old_milestone_unet_resnet101_norm_1024.onnx"
UNET_LEGACY_MODEL_FILENAME = "best_unet_resnet101_norm_1024.onnx"

KNOWN_MODEL_EXAMPLES = (
    SEGFORMER_CUDA_MODEL_FILENAME,
    SEGFORMER_CPU_MODEL_FILENAME,
    SEGFORMER_LAST_CUDA_MODEL_FILENAME,
    UNET_BEST_CUDA_MODEL_FILENAME,
    UNET_LAST_CUDA_MODEL_FILENAME,
    UNET_OLD_MILESTONE_MODEL_FILENAME,
    UNET_LEGACY_MODEL_FILENAME,
)
FIXED_TILE_SIZE = 1024
MAX_LARGE_EXPORT_PIXELS = 10240
EXPORT_CACHE_SCHEMA_VERSION = 2
EXPORT_CACHE_IDENTITY_VERSION = "arcgui-r7-exact-polygon-cache-v2"
EXPORT_CACHE_DIRNAME = "export_cache"
FIXED_CELL_SIZE = 2.0
MAX_ONNX_BATCH_SIZE = 64
HIGH_OVERLAP_WARNING_PERCENT = 75.0


def _safe_name_tool(name):
    name = str(name).strip().strip("'\"")
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "extent")[:80]


def _extent_name_tool(extent_layer):
    try:
        return _safe_name_tool(arcpy.Describe(extent_layer).name)
    except Exception:
        return _safe_name_tool(extent_layer)


def _sha256_text_tool(value):
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _sr_fingerprint_tool(sr):
    if sr is None:
        return "unknown"
    try:
        text = sr.exportToString()
        if text:
            return _sha256_text_tool(text)
    except Exception:
        pass
    try:
        return _sha256_text_tool(
            f"{sr.name}|{sr.factoryCode}|{getattr(sr, 'type', '')}"
        )
    except Exception:
        return "unknown"


def _geometry_fingerprint_tool(extent_layer):
    values = []
    with arcpy.da.SearchCursor(extent_layer, ["SHAPE@JSON"]) as cursor:
        for (shape_json,) in cursor:
            if shape_json:
                values.append(str(shape_json))
    if not values:
        return None
    values.sort()
    return _sha256_text_tool("\n".join(values))


def _layer_source_identity_tool(layer):
    values = []
    try:
        value = str(layer.dataSource).strip()
        if value:
            values.append(value)
    except Exception:
        pass
    try:
        values.append(json.dumps(layer.connectionProperties, sort_keys=True, default=str))
    except Exception:
        pass
    try:
        values.append(str(layer.name))
    except Exception:
        values.append(str(layer))
    return _sha256_text_tool("|".join(values))


def _cache_config_tool(
    extent_layer,
    input_raster,
    input_raster_name,
    active_map,
    overlap_percent,
    optimized_large_crop,
):
    desc = arcpy.Describe(extent_layer)
    geometry_hash = _geometry_fingerprint_tool(extent_layer)
    if not geometry_hash:
        return None
    try:
        map_sr = active_map.spatialReference
    except Exception:
        map_sr = None
    return {
        "schema_version": EXPORT_CACHE_SCHEMA_VERSION,
        "identity_version": EXPORT_CACHE_IDENTITY_VERSION,
        "extent_name": _extent_name_tool(extent_layer),
        "extent_geometry_sha256": geometry_hash,
        "extent_source_crs_sha256": _sr_fingerprint_tool(desc.spatialReference),
        "map_crs_sha256": _sr_fingerprint_tool(map_sr),
        "input_raster_name": str(input_raster_name or "").strip(),
        "input_raster_source_sha256": _layer_source_identity_tool(input_raster),
        "optimized_large_crop": bool(_tool_bool(optimized_large_crop)),
        "overlap_percent": round(float(overlap_percent), 6),
        "extent_policy": "exact-selected-polygon",
        "fixed_tile_size": int(FIXED_TILE_SIZE),
        "fixed_cell_size": float(FIXED_CELL_SIZE),
        "max_large_export_pixels": int(MAX_LARGE_EXPORT_PIXELS),
    }


def _cache_id_tool(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _cache_tag_tool(config):
    cache_id = _cache_id_tool(config)
    mode = "optimized" if config.get("optimized_large_crop") else "direct"
    overlap = float(config.get("overlap_percent", 0.0))
    overlap_text = str(int(round(overlap))) if abs(overlap-round(overlap)) < 1e-9 else str(overlap).replace(".", "p")
    return f"{mode}_ov{overlap_text}_{cache_id}"


def _cache_status_for_extent(
    extent_layer,
    output_root,
    input_raster,
    input_raster_name,
    active_map,
    overlap_percent,
    optimized_large_crop,
    extent_output_name=None,
):
    config = _cache_config_tool(
        extent_layer,
        input_raster,
        input_raster_name,
        active_map,
        overlap_percent,
        optimized_large_crop,
    )
    if not config:
        return "unknown", None
    cache_tag = _cache_tag_tool(config)
    extent_folder = _safe_name_tool(extent_output_name or _extent_name_tool(extent_layer))
    cache_root = Path(output_root) / extent_folder / EXPORT_CACHE_DIRNAME / cache_tag
    manifest_path = cache_root / "export_config.json"
    if cache_root.exists() and not manifest_path.exists():
        return "incomplete", cache_tag
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("complete") is True and manifest.get("config") == config:
                exports = manifest.get("exports") or []
                valid_files = True
                if not exports:
                    valid_files = False
                for item in exports:
                    stored = Path(str(item.get("relative_path", "")))
                    if stored.is_absolute() or ".." in stored.parts:
                        valid_files = False
                        break
                    candidate = cache_root / stored
                    if not candidate.exists() or not candidate.is_file() or candidate.stat().st_size <= 0:
                        valid_files = False
                        break
                if valid_files:
                    return "hit", cache_tag
                return "incomplete", cache_tag
        except Exception:
            return "incomplete", cache_tag

    # Other cache configurations may already exist for this extent. Old r6
    # manifests are intentionally not reused because older versions could use
    # a different processing-extent policy.
    cache_parent = Path(output_root) / extent_folder / EXPORT_CACHE_DIRNAME
    if cache_parent.exists():
        children = [x for x in cache_parent.iterdir() if x.is_dir()]
        if children:
            older_schema = False
            for child in children:
                mp = child / "export_config.json"
                try:
                    data = json.loads(mp.read_text(encoding="utf-8"))
                    old_cfg = data.get("config") or {}
                    if old_cfg.get("identity_version") != EXPORT_CACHE_IDENTITY_VERSION:
                        older_schema = True
                        break
                except Exception:
                    continue
            return ("legacy_incompatible" if older_schema else "different"), cache_tag

    # Pre-cache/legacy export folders cannot prove that they used the exact
    # current processing-extent policy, so r7 refuses to auto-adopt them.
    extent_root = Path(output_root) / extent_folder
    if any(extent_root.glob("large_exports_max_*px*")):
        return "legacy_incompatible", cache_tag

    return "none", cache_tag


def _checkpoint_label_from_filename(name):
    stem = Path(str(name)).stem.lower()
    if "old_milestone" in stem or "old-milestone" in stem:
        return "Old milestone"
    if re.search(r"(^|[_\-])last([_\-]|$)", stem):
        return "Last checkpoint"
    if re.search(r"(^|[_\-])best([_\-]|$)", stem):
        return "Best checkpoint"
    return "Selected model"


def _selected_model_profile(model_path):
    if not model_path:
        return None

    path = Path(str(model_path))
    name = path.name.lower()
    stem = path.stem.lower()

    if path.suffix.lower() != ".onnx":
        return {
            "key": "unknown",
            "target": "unknown",
            "display": "Unknown",
            "default_batch": 1,
        }

    is_segformer = "segformer" in stem
    is_unet_r101 = (
        "unet" in stem
        and ("resnet101" in stem or "r101" in stem)
    )

    if not is_segformer and not is_unet_r101:
        return {
            "key": "unknown",
            "target": "unknown",
            "display": "Unknown",
            "default_batch": 1,
        }

    if "cpu" in stem:
        target = "cpu"
    elif "cuda" in stem or "fp16" in stem:
        target = "cuda"
    else:
        target = "auto"

    checkpoint_label = _checkpoint_label_from_filename(name)
    target_label = {
        "cuda": "CUDA",
        "cpu": "CPU",
        "auto": "CUDA preferred / CPU fallback",
    }[target]

    if is_segformer:
        return {
            "key": "segformer",
            "target": target,
            "display": f"SegFormer MiT-B5 - {checkpoint_label} ({target_label})",
            "default_batch": 1 if target == "cpu" else 2,
        }

    return {
        "key": "unet_resnet101",
        "target": target,
        "display": f"U-Net ResNet101 - {checkpoint_label} ({target_label})",
        "default_batch": 1 if target == "cpu" else 4,
    }

def _overlap_to_stride(overlap_percent):
    overlap_percent = float(overlap_percent)
    stride = int(round(FIXED_TILE_SIZE * (1.0 - overlap_percent / 100.0)))
    return max(1, stride)


def _parse_multivalue(value_as_text):
    if not value_as_text:
        return []

    values = []
    for item in str(value_as_text).split(";"):
        item = item.strip().strip("'\"")
        if item:
            values.append(item)
    return values


def _tool_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"Boolean option must be True or False; received {value!r}.")


def _valid_sr_tool(sr):
    if sr is None:
        return False
    try:
        return str(sr.name).strip().lower() not in ("", "unknown", "none")
    except Exception:
        return False


def _resolve_input_layer_tool(active_map, input_name):
    if active_map is None:
        raise RuntimeError("No active ArcGIS Pro map is available.")
    requested = str(input_name or "").strip()
    if not requested:
        raise ValueError("Input Raster Layer Name cannot be empty.")
    layers = []
    for lyr in active_map.listLayers():
        try:
            if str(lyr.name).strip().casefold() == requested.casefold():
                layers.append(lyr)
        except Exception:
            pass
    if not layers:
        raise RuntimeError(f"No exact map layer named '{requested}' was found.")
    if len(layers) > 1:
        raise RuntimeError(
            f"{len(layers)} map layers are named '{requested}'. Rename the intended "
            "imagery layer so its name is unique."
        )
    layer = layers[0]
    try:
        if bool(layer.isBroken):
            raise RuntimeError(f"Imagery layer '{requested}' is broken/unavailable.")
    except AttributeError:
        pass
    try:
        if bool(layer.isFeatureLayer):
            raise RuntimeError(f"Layer '{requested}' is a feature layer, not imagery/raster data.")
    except AttributeError:
        pass
    try:
        if bool(layer.isGroupLayer):
            raise RuntimeError(
                f"Layer '{requested}' is a group layer. Use the actual imagery layer inside that group."
            )
    except AttributeError:
        pass
    return layer


class Toolbox(object):
    def __init__(self):
        self.label = "ArcGUI Road Segmentation"
        self.alias = "arcgui_roads"
        self.tools = [RoadSegmentation]


class RoadSegmentation(object):
    def __init__(self):
        self.label = "ArcGUI Road Segmentation"
        self.description = (
            "Road segmentation using a fixed 1024x1024 model input and the exact selected polygon extent. "
            "Choose overlap as a percentage. In optimized mode, Export Training Data For Deep Learning "
            "extracts balanced, non-overlapping World Imagery blocks up to 10240x10240 pixels; "
            "large blocks that do not intersect the requested extent are skipped before export; "
            "if an image service rejects a large request ArcGUI can retry with smaller balanced blocks. "
            "Matching export configurations can reuse previously downloaded georeferenced imagery, "
            "so different ONNX models can be compared without repeating World Imagery export. "
            "Debug mode previews the exact planned large-block and model-crop footprints without export or ONNX inference."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="Extent Layer(s)", name="extent_layers", datatype="GPFeatureLayer",
            parameterType="Required", direction="Input", multiValue=True,
        )
        p0.filter.list = ["Polygon"]

        p1 = arcpy.Parameter(
            displayName="Input Raster Layer Name", name="input_raster_name", datatype="GPString",
            parameterType="Required", direction="Input",
        )
        p1.value = "World Imagery"

        p2 = arcpy.Parameter(
            displayName="Output Folder", name="output_root", datatype="DEFolder",
            parameterType="Required", direction="Input",
        )

        p3 = arcpy.Parameter(
            displayName="Road Segmentation ONNX Model", name="model_path", datatype="DEFile",
            parameterType="Optional", direction="Input",
        )
        p3.filter.list = ["onnx"]

        p4 = arcpy.Parameter(
            displayName="Tile Overlap (%)", name="overlap_percent", datatype="GPDouble",
            parameterType="Required", direction="Input",
        )
        p4.value = 50.0
        p4.filter.type = "Range"
        p4.filter.list = [0.0, 95.0]

        p5 = arcpy.Parameter(
            displayName="Optimized Large-Area Export + Python Cropping",
            name="optimized_large_crop", datatype="GPBoolean",
            parameterType="Required", direction="Input",
        )
        p5.value = True

        p6 = arcpy.Parameter(
            displayName="Prediction Threshold", name="threshold", datatype="GPDouble",
            parameterType="Required", direction="Input",
        )
        p6.value = 0.5
        p6.filter.type = "Range"
        p6.filter.list = [0.0, 1.0]

        p7 = arcpy.Parameter(
            displayName="ONNX Batch Size", name="batch_size", datatype="GPLong",
            parameterType="Required", direction="Input",
        )
        p7.value = 2
        p7.filter.type = "Range"
        p7.filter.list = [1, MAX_ONNX_BATCH_SIZE]

        p8 = arcpy.Parameter(
            displayName="Debug Footprint Preview Only", name="debug_preview_only", datatype="GPBoolean",
            parameterType="Required", direction="Input",
        )
        p8.value = False

        p9 = arcpy.Parameter(
            displayName="Reuse Matching Exported Imagery", name="reuse_matching_exports",
            datatype="GPBoolean", parameterType="Required", direction="Input",
        )
        p9.value = True

        return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        debug_only = _tool_bool(parameters[8].value)

        parameters[1].enabled = True
        parameters[3].enabled = not debug_only
        parameters[6].enabled = not debug_only
        parameters[7].enabled = not debug_only
        parameters[9].enabled = not debug_only

        model_text = parameters[3].valueAsText
        profile = _selected_model_profile(model_text)
        if (
            parameters[3].altered
            and not parameters[7].altered
            and profile
            and profile["key"] != "unknown"
        ):
            parameters[7].value = profile["default_batch"]
        return

    def updateMessages(self, parameters):
        debug_only = _tool_bool(parameters[8].value)
        optimized = _tool_bool(parameters[5].value)

        # Extent validation.
        extent_values = _parse_multivalue(parameters[0].valueAsText)
        if extent_values:
            normalized = [x.strip().casefold() for x in extent_values]
            if len(set(normalized)) != len(normalized):
                parameters[0].setErrorMessage(
                    "The same extent layer is listed more than once. Remove duplicate entries."
                )
            else:
                try:
                    for extent in extent_values:
                        if not arcpy.Exists(extent):
                            raise ValueError(f"Extent layer not found: {extent}")
                        desc = arcpy.Describe(extent)
                        if str(getattr(desc, "shapeType", "")).strip().lower() != "polygon":
                            raise ValueError(f"Extent must be polygon geometry: {extent}")
                        if not _valid_sr_tool(getattr(desc, "spatialReference", None)):
                            raise ValueError(
                                f"Extent has an Unknown/undefined CRS: {extent}. Define the correct CRS before running."
                            )
                except Exception as exc:
                    parameters[0].setErrorMessage(str(exc))

        # Input imagery / active-map validation.
        input_name = parameters[1].valueAsText
        input_layer = None
        active_map = None
        if not input_name or not str(input_name).strip():
            parameters[1].setErrorMessage("Input Raster Layer Name is required.")
        else:
            try:
                aprx = arcpy.mp.ArcGISProject("CURRENT")
                active_map = aprx.activeMap
                input_layer = _resolve_input_layer_tool(active_map, input_name)
                map_sr = active_map.spatialReference
                if not _valid_sr_tool(map_sr):
                    raise RuntimeError("The active map has an Unknown coordinate system.")
                sr_type = str(getattr(map_sr, "type", "")).strip().lower()
                meters_per_unit = float(getattr(map_sr, "metersPerUnit", float("nan")))
                if sr_type and sr_type != "projected":
                    raise RuntimeError(
                        "ArcGUI requires a projected active-map CRS because export resolution is fixed at 2 m/pixel."
                    )
                if not math.isfinite(meters_per_unit) or abs(meters_per_unit - 1.0) > 1e-6:
                    raise RuntimeError(
                        "ArcGUI requires active-map units in metres (metersPerUnit must equal 1)."
                    )
            except Exception as exc:
                parameters[1].setErrorMessage(str(exc))

        # Output folder basic sanity. Production performs an actual write probe.
        output_root_text = parameters[2].valueAsText
        if not output_root_text or not str(output_root_text).strip():
            parameters[2].setErrorMessage("Output Folder is required.")
        else:
            try:
                out_path = Path(output_root_text)
                if out_path.exists() and not out_path.is_dir():
                    parameters[2].setErrorMessage("Output Folder points to a file, not a directory.")
                elif len(str(out_path)) > 220:
                    parameters[2].setWarningMessage(
                        "The selected output path is very long. ArcGIS/Windows raster tools may fail on deeply nested paths; consider a shorter root folder."
                    )
            except Exception:
                pass

        # Overlap.
        if parameters[4].value is not None:
            try:
                overlap = float(parameters[4].value)
                if not math.isfinite(overlap) or overlap < 0.0 or overlap > 95.0:
                    parameters[4].setErrorMessage("Tile overlap must be a finite value between 0 and 95 percent.")
                else:
                    stride = _overlap_to_stride(overlap)
                    if overlap >= HIGH_OVERLAP_WARNING_PERCENT:
                        parameters[4].setWarningMessage(
                            f"1024x1024 input -> stride {stride}px. {overlap:.1f}% overlap creates many more model windows and can greatly increase runtime/memory use."
                        )
                    else:
                        parameters[4].setWarningMessage(
                            f"Fixed 1024x1024 model input -> derived X/Y stride = {stride} pixels."
                        )
            except Exception:
                parameters[4].setErrorMessage("Tile overlap must be numeric.")

        # Model.
        model_text = parameters[3].valueAsText
        if not debug_only:
            if not model_text:
                parameters[3].setErrorMessage(
                    "Select a supported SegFormer MiT-B5 or U-Net ResNet101 ONNX road model."
                )
            else:
                try:
                    model_file = Path(model_text)
                    if not model_file.exists() or not model_file.is_file():
                        raise ValueError("Selected ONNX model does not exist or is not a file.")
                    if model_file.stat().st_size <= 0:
                        raise ValueError("Selected ONNX model file is empty.")
                    profile = _selected_model_profile(model_text)
                    if profile["key"] == "unknown":
                        raise ValueError(
                            "Unsupported ONNX filename. Use a SegFormer filename containing 'segformer', or a U-Net ResNet101 filename containing 'unet' and 'resnet101'/'r101'."
                        )
                    if profile["target"] == "cuda":
                        parameters[3].setWarningMessage(
                            f"{profile['display']} selected; CUDAExecutionProvider is required."
                        )
                    elif profile["target"] == "cpu":
                        parameters[3].setWarningMessage(
                            f"{profile['display']} selected; CPUExecutionProvider is forced."
                        )
                    else:
                        parameters[3].setWarningMessage(
                            f"{profile['display']} selected; CUDA is preferred and CPU fallback is allowed."
                        )
                except Exception as exc:
                    parameters[3].setErrorMessage(str(exc))

        # Threshold.
        if not debug_only and parameters[6].value is not None:
            try:
                threshold = float(parameters[6].value)
                if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
                    parameters[6].setErrorMessage("Prediction threshold must be a finite value between 0 and 1.")
                elif threshold in (0.0, 1.0):
                    parameters[6].setWarningMessage(
                        "A threshold at exactly 0 or 1 is an extreme operating point and may produce nearly all-road or all-background output."
                    )
            except Exception:
                parameters[6].setErrorMessage("Prediction threshold must be numeric.")

        # Batch size.
        if not debug_only and parameters[7].value is not None:
            try:
                batch_size = int(parameters[7].value)
                if batch_size <= 0 or batch_size > MAX_ONNX_BATCH_SIZE:
                    parameters[7].setErrorMessage(
                        f"ONNX batch size must be between 1 and {MAX_ONNX_BATCH_SIZE}."
                    )
                elif batch_size > 8:
                    parameters[7].setWarningMessage(
                        "A large batch size can cause CPU/GPU out-of-memory errors for 1024x1024 models. Use the model-specific default unless you have tested the hardware."
                    )
            except Exception:
                parameters[7].setErrorMessage("ONNX batch size must be an integer.")

        # Cache status is informative; production revalidates every hit.
        if not debug_only:
            reuse_cache = _tool_bool(parameters[9].value)
            if not reuse_cache:
                parameters[9].setWarningMessage(
                    "Cache reuse is disabled. ArcGUI will export imagery again even if an exact r7 cache exists."
                )
            elif (
                output_root_text and extent_values and input_layer is not None and active_map is not None
                and parameters[4].value is not None
            ):
                try:
                    statuses = []
                    used_names = {}
                    for extent_value in extent_values:
                        base_name = _extent_name_tool(extent_value)
                        occurrence = used_names.get(base_name, 0) + 1
                        used_names[base_name] = occurrence
                        extent_output_name = (
                            base_name if occurrence == 1 else f"{base_name}_{occurrence}"
                        )
                        status, _ = _cache_status_for_extent(
                            extent_layer=extent_value,
                            output_root=output_root_text,
                            input_raster=input_layer,
                            input_raster_name=input_name,
                            active_map=active_map,
                            overlap_percent=float(parameters[4].value),
                            optimized_large_crop=optimized,
                            extent_output_name=extent_output_name,
                        )
                        statuses.append(status)
                    hits = sum(x == "hit" for x in statuses)
                    different = sum(x == "different" for x in statuses)
                    incomplete = sum(x == "incomplete" for x in statuses)
                    incompatible = sum(x == "legacy_incompatible" for x in statuses)
                    if hits:
                        parameters[9].setWarningMessage(
                            f"CACHE HIT: matching r7 exported imagery found for {hits}/{len(statuses)} extent(s). Production will validate it again before ONNX inference; valid TIFFs will be reused and imagery export skipped."
                        )
                    elif incomplete:
                        parameters[9].setWarningMessage(
                            "A matching cache folder exists but is incomplete or missing files. ArcGUI will reject it and create a fresh cache."
                        )
                    elif incompatible:
                        parameters[9].setWarningMessage(
                            "Older pre-r7 exports were found. Their cache schema is not compatible with the r7 exact-polygon policy, so they will not be reused automatically. A fresh r7 cache will be created once."
                        )
                    elif different:
                        parameters[9].setWarningMessage(
                            "Cached imagery exists for this extent, but the overlap/mode/geometry configuration differs. A separate cache will be created."
                        )
                except Exception:
                    pass

        if debug_only:
            if optimized:
                parameters[8].setWarningMessage(
                    "Debug only: no imagery export and no ONNX inference. ArcGUI will draw retained/skipped large blocks and exact 1024 model-window footprints for the selected polygon."
                )
            else:
                parameters[8].setWarningMessage(
                    "Debug only: no imagery export and no ONNX inference. Cyan direct 1024 tiles are the exact retained windows; completely outside candidates are skipped."
                )
        elif optimized:
            parameters[5].setWarningMessage(
                f"Optimized mode: balanced non-overlapping imagery blocks <= {MAX_LARGE_EXPORT_PIXELS}x{MAX_LARGE_EXPORT_PIXELS}px are exported only where they intersect the exact selected polygon. Service-size failures can retry with smaller blocks."
            )
        return

    def execute(self, parameters, messages):
        extent_layers = _parse_multivalue(parameters[0].valueAsText)
        if not extent_layers:
            raise arcpy.ExecuteError("At least one extent layer must be selected.")

        normalized = [x.strip().casefold() for x in extent_layers]
        if len(set(normalized)) != len(normalized):
            raise arcpy.ExecuteError("The same extent layer was supplied more than once.")

        input_raster_name = str(parameters[1].valueAsText or "").strip()
        output_root = parameters[2].valueAsText
        model_path = parameters[3].valueAsText

        if parameters[4].value is None:
            raise arcpy.ExecuteError("Tile Overlap (%) is required.")
        overlap_percent = float(parameters[4].value)
        if not math.isfinite(overlap_percent) or not 0.0 <= overlap_percent <= 95.0:
            raise arcpy.ExecuteError("Tile overlap must be between 0 and 95 percent.")

        optimized_large_crop = _tool_bool(parameters[5].value)
        threshold = float(parameters[6].value) if parameters[6].value is not None else 0.5
        batch_size = int(parameters[7].value) if parameters[7].value is not None else 1
        debug_only = _tool_bool(parameters[8].value)
        reuse_matching_exports = _tool_bool(parameters[9].value)

        if not output_root:
            raise arcpy.ExecuteError("Output Folder is required.")
        if not input_raster_name:
            raise arcpy.ExecuteError("Input Raster Layer Name is required.")
        if not debug_only:
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise arcpy.ExecuteError("Prediction threshold must be between 0 and 1.")
            if batch_size <= 0 or batch_size > MAX_ONNX_BATCH_SIZE:
                raise arcpy.ExecuteError(
                    f"ONNX batch size must be between 1 and {MAX_ONNX_BATCH_SIZE}."
                )

        stride_pixels = _overlap_to_stride(overlap_percent)
        toolbox_folder = Path(__file__).resolve().parent
        backend_path = toolbox_folder / "roadsegmentation_python.py"
        if not backend_path.exists() or not backend_path.is_file():
            raise arcpy.ExecuteError(f"ArcGUI backend not found beside toolbox: {backend_path}")

        module_name = "arcgui_roadsegmentation_backend_" + str(abs(hash(str(backend_path.resolve()))))
        spec = importlib.util.spec_from_file_location(module_name, str(backend_path))
        if spec is None or spec.loader is None:
            raise arcpy.ExecuteError(f"Could not load ArcGUI backend module: {backend_path}")
        roadsegmentation_python = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(roadsegmentation_python)
        except Exception as exc:
            arcpy.AddError(f"ArcGUI backend import failed: {exc}")
            raise

        if debug_only:
            arcpy.AddMessage("============================================")
            arcpy.AddMessage("ArcGUI DEBUG FOOTPRINT PREVIEW ONLY")
            arcpy.AddMessage("============================================")
            arcpy.AddMessage("No imagery export will run and no ONNX model will be loaded.")
            arcpy.AddMessage("Extent policy: exact selected polygon")
            arcpy.AddMessage(f"Fixed model crop: {FIXED_TILE_SIZE} x {FIXED_TILE_SIZE} px")
            arcpy.AddMessage(f"Overlap: {overlap_percent:.2f}%")
            arcpy.AddMessage(f"Derived stride: {stride_pixels} px")
            arcpy.AddMessage(f"Optimized large-area mode: {optimized_large_crop}")
            outputs, failures = roadsegmentation_python.run_debug_preview_multiple(
                extent_layers=extent_layers,
                output_root=output_root,
                overlap_percent=overlap_percent,
                optimized_large_crop=optimized_large_crop,
                input_raster_name=input_raster_name,
            )
            arcpy.AddMessage(f"Debug preview layers created: {len(outputs)}")
            arcpy.AddMessage(f"Debug failures: {len(failures)}")
            if failures:
                if not outputs:
                    details = "; ".join(f"{layer}: {error}" for layer, error in failures[:5])
                    raise arcpy.ExecuteError("All debug previews failed. Details: " + details)
                arcpy.AddWarning("One or more debug previews failed; review the geoprocessing messages.")
            return

        profile = _selected_model_profile(model_path)
        if not profile or profile["key"] == "unknown":
            raise arcpy.ExecuteError(
                "Invalid model filename. Accepted families are SegFormer MiT-B5 and U-Net ResNet101."
            )
        model_file = Path(str(model_path))
        if not model_file.exists() or not model_file.is_file() or model_file.stat().st_size <= 0:
            raise arcpy.ExecuteError("Selected ONNX model file is missing, invalid, or empty.")

        arcpy.AddMessage("============================================")
        arcpy.AddMessage("ArcGUI Road Segmentation")
        arcpy.AddMessage("============================================")
        arcpy.AddMessage(f"Detected model: {profile['display']}")
        arcpy.AddMessage("Extent policy: exact selected polygon")
        arcpy.AddMessage("Fixed tile/model input: 1024 x 1024 px")
        arcpy.AddMessage(f"Overlap: {overlap_percent:.2f}%")
        arcpy.AddMessage(f"Derived stride: {stride_pixels} px")
        arcpy.AddMessage(f"Optimized large-area export: {optimized_large_crop}")
        arcpy.AddMessage(f"Extent count: {len(extent_layers)}")
        arcpy.AddMessage(f"Batch size: {batch_size}")
        arcpy.AddMessage(f"Reuse matching exported imagery: {reuse_matching_exports}")

        try:
            outputs, failures = roadsegmentation_python.run_road_tool_multiple(
                extent_layers=extent_layers,
                input_raster_name=input_raster_name,
                output_root=output_root,
                model_path=model_path,
                overlap_percent=overlap_percent,
                threshold=threshold,
                optimized_large_crop=optimized_large_crop,
                batch_size=batch_size,
                reuse_matching_exports=reuse_matching_exports,
            )
        except Exception as exc:
            arcpy.AddError(f"ArcGUI job failed: {exc}")
            raise

        arcpy.AddMessage("")
        arcpy.AddMessage("============================================")
        arcpy.AddMessage("ArcGUI job complete")
        arcpy.AddMessage("============================================")
        arcpy.AddMessage(f"Successful extents: {len(outputs)}")
        arcpy.AddMessage(f"Failed extents: {len(failures)}")
        arcpy.AddMessage(
            f"Final predictions root: {Path(output_root) / 'final_predictions'} (organized by model and cache configuration)"
        )
        if failures:
            if not outputs:
                details = "; ".join(f"{layer}: {error}" for layer, error in failures[:5])
                raise arcpy.ExecuteError("All requested extents failed. Details: " + details)
            arcpy.AddWarning("One or more extents failed; review the geoprocessing messages.")

