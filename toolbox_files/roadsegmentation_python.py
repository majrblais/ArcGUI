# =========================================================
# ArcGUI v5.4 FINAL - road segmentation backend
# =========================================================
# Supported ONNX model families:
#   - SegFormer MiT-B5 models whose filename contains 'segformer'
#   - U-Net ResNet101 models whose filename contains 'unet' and 'resnet101'/'r101'
#
# Execution target is inferred from the filename:
#   - contains 'cpu'              -> CPU only
#   - contains 'cuda' or 'fp16'  -> CUDA required
#   - otherwise                  -> CUDA preferred, CPU fallback
#
# This accepts the current best/last/final/old-milestone naming variants
# without tying ArcGUI to one exact checkpoint filename.
#
# Model input size is always 1024x1024.
# The GUI exposes overlap as a percentage and the stride is
# derived automatically.
#
# Two export modes are available:
#   1) Direct 1024 export: ArcGUI first removes 1024 windows with zero
#      intersection with the requested polygon, then ArcGIS exports
#      only the retained model windows. Boundary-intersecting windows remain.
#   2) Optimized large-area export: ArcGIS exports one or more
#      BALANCED, NON-OVERLAPPING large imagery blocks. Every block
#      is <=10240x10240 pixels. Any large block that does not
#      intersect the requested extent is skipped
#      BEFORE World Imagery export. Python then creates overlapping
#      1024x1024 model crops only inside the retained large blocks.
#
# If the working bounding rectangle fits within 10240x10240, it is
# exported in one shot. Larger rectangles are split uniformly per
# axis (for example, 12000 px -> about 6000 + 6000; 20000 px ->
# 10000 + 10000). Large blocks never overlap each other.
#
# Matching imagery exports are reusable across models through a configuration-aware cache.
# Model, threshold and batch size do not force a World Imagery re-export.
# r7 uses the exact selected polygon only.
#
# Both modes merge overlapping MODEL-CROP predictions by MEAN
# PROBABILITY before applying the final threshold.
#
# In optimized mode, if ArcGIS emits multiple TIFFs for one planned large
# block, completely black duplicate/edge TIFFs are ignored before the
# usable World Imagery TIFF is selected for inference.
#
# Debug preview mode performs no imagery export and no ONNX
# inference. It creates transparent polygon footprint layers
# showing the exact planned large export blocks and/or model
# crops instead of an opaque fishnet.
#
# Debug/production extent planning includes a geometry-based fallback for
# ArcGIS cases where a newly projected in-memory irregular polygon reports
# NaN Describe(...).extent metadata.
#
# Hardened safeguards validate polygon geometry, reject undefined source CRS,
# require projected metric map units, check image-service request limits, black/unreadable
# imagery, TIFF CRS/cell size/alignment, ONNX input/output contracts, duplicate
# output names, available ArcGIS licenses, writable disk space, and excessively
# large debug previews. Selections/definition queries are materialized first.
# =========================================================

from pathlib import Path
import os
import sys
import glob
import re
import math
import uuid
import shutil
import json
import hashlib
from datetime import datetime, timezone

# =========================================================
# CUDA / ONNX DLL DISCOVERY
# Must happen before importing onnxruntime.
# =========================================================

_ENV = sys.prefix
_NVIDIA_ROOT = Path(_ENV) / "Lib" / "site-packages" / "nvidia"

if hasattr(os, "add_dll_directory"):
    for _dll_dir in glob.glob(str(_NVIDIA_ROOT / "*" / "bin")):
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

    _LIBRARY_BIN = Path(_ENV) / "Library" / "bin"
    if _LIBRARY_BIN.exists():
        try:
            os.add_dll_directory(str(_LIBRARY_BIN))
        except Exception:
            pass
        os.environ["PATH"] = str(_LIBRARY_BIN) + os.pathsep + os.environ.get("PATH", "")

# =========================================================
# IMPORTS
# =========================================================

import arcpy
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from arcpy.ia import ExportTrainingDataForDeepLearning
from arcpy.sa import Con

# =========================================================
# FIXED SETTINGS
# =========================================================

ARCGUI_VERSION = "5.4 FINAL r7"

FIXED_TILE_SIZE = 1024
FIXED_MODEL_INPUT_SIZE = 1024
FIXED_CELL_SIZE = 2.0

# Maximum ArcGIS-exported block used by optimized mode.
# 10240 = 10 x 1024, which keeps the limit easy to reason about
# relative to the fixed model input size.
MAX_LARGE_EXPORT_PIXELS = 10240
MIN_LARGE_EXPORT_PIXELS = 1024

# ArcGIS image services can impose maxImageWidth/maxImageHeight limits.
# Optimized export can fall back to progressively smaller non-overlapping
# blocks when a request-size failure is detected.
MAX_LARGE_EXPORT_SIZE_RETRIES = 4

# Export-validation tolerances. These are intentionally much smaller than a
# genuine CRS/georeferencing failure, while allowing normal raster alignment
# and service rounding.
CELL_SIZE_RELATIVE_TOLERANCE = 0.02
EXPORT_ALIGNMENT_TOLERANCE_PIXELS = 16
EXPORT_ALIGNMENT_TOLERANCE_FRACTION = 0.005
EXPORT_DIMENSION_TOLERANCE_FRACTION = 0.03

# Prevent debug-preview vector generation from accidentally creating an
# enormous feature class that can freeze ArcGIS Pro.
MAX_DEBUG_CROP_FEATURES = 500000

# Defensive parameter caps. These are intentionally generous and exist to catch
# accidental GUI/API values that would otherwise cause extreme memory or runtime use.
MAX_ONNX_BATCH_SIZE = 64
HIGH_OVERLAP_WARNING_PERCENT = 75.0

# Basic output-storage safeguards.
MIN_FREE_DISK_GB_ERROR = 0.5
MIN_FREE_DISK_GB_WARNING = 5.0

# Reusable imagery-cache metadata. Only settings that affect the exported
# imagery/crop geometry belong in the cache identity. Model, threshold and
# batch size are deliberately excluded so different models can reuse the same
# already-downloaded imagery.
EXPORT_CACHE_SCHEMA_VERSION = 2
EXPORT_CACHE_IDENTITY_VERSION = "arcgui-r7-exact-polygon-cache-v2"
EXPORT_CACHE_DIRNAME = "export_cache"
MODEL_OUTPUTS_DIRNAME = "model_outputs"

# Canonical/current deployment filenames.
# Model recognition below is intentionally family-based rather than exact-name-only.
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


class ArcGUIError(RuntimeError):
    """Base exception for ArcGUI runtime safeguards."""


class ArcGUIExportError(ArcGUIError):
    """ArcGIS imagery export failed."""


class ArcGUIExportSizeError(ArcGUIExportError):
    """Imagery export appears to exceed a service/request-size limit."""


class ArcGUIImageryContentError(ArcGUIError):
    """Exported imagery is unreadable, missing, or completely black."""


class ArcGUIGeoreferenceError(ArcGUIError):
    """Exported raster georeferencing is inconsistent with the plan."""


# =========================================================
# LOGGING
# =========================================================


def _log(message=""):
    text = str(message)
    print(text)
    try:
        arcpy.AddMessage(text)
    except Exception:
        pass


def _warn(message):
    text = str(message)
    print("WARNING:", text)
    try:
        arcpy.AddWarning(text)
    except Exception:
        pass

# =========================================================
# BASIC HELPERS
# =========================================================


def _next_available_path(base_path):
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path

    for i in range(1, 100000):
        candidate = base_path.parent / f"{base_path.name}_{i}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available path for: {base_path}")


def _next_available_file_path(base_path):
    """Return a non-conflicting file path while preserving the extension."""
    base_path = Path(base_path)

    if not base_path.exists() and not arcpy.Exists(str(base_path)):
        return base_path

    for i in range(1, 100000):
        candidate = base_path.with_name(
            f"{base_path.stem}_{i}{base_path.suffix}"
        )
        if not candidate.exists() and not arcpy.Exists(str(candidate)):
            return candidate

    raise RuntimeError(
        f"Could not find an available output filename for: {base_path}"
    )



def _ensure_writable_output_directory(path):
    """Create the output directory, verify write access, and report low disk space."""
    if path is None or not str(path).strip():
        raise ValueError("Output Folder is required.")

    path = Path(path)
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Output Folder points to a file, not a directory: {path}")

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Could not create/access Output Folder {path}: {exc}") from exc

    if len(str(path.resolve())) > 220:
        _warn(
            "The output root path is long. ArcGIS/Windows raster tools can fail "
            "on deeply nested paths; consider a shorter output root if errors occur."
        )

    probe = path / f".arcgui_write_test_{uuid.uuid4().hex}.tmp"

    try:
        probe.write_text("ArcGUI", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"Output folder is not writable: {path}. Details: {exc}"
        )

    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024.0 ** 3)

        if free_gb < MIN_FREE_DISK_GB_ERROR:
            raise RuntimeError(
                f"Output drive has only {free_gb:.2f} GB free. "
                "ArcGUI may create large temporary imagery/probability rasters; "
                "free additional disk space before running."
            )

        if free_gb < MIN_FREE_DISK_GB_WARNING:
            _warn(
                f"Output drive has only {free_gb:.2f} GB free. "
                "Large-area processing may run out of disk space."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        _warn(f"Could not query free disk space for {path}: {exc}")

    return path


def _ensure_required_extensions():
    """
    Check out ArcGIS extensions needed by the production pipeline.

    Debug preview does not call this because it performs no imagery export,
    raster algebra, or ONNX inference.
    """
    checked_out = []

    # ExportTrainingDataForDeepLearning specifically requires Image Analyst.
    # Con is also licensed by Image Analyst, so requiring a separate
    # Spatial Analyst license would unnecessarily reject valid installations.
    for extension in ("ImageAnalyst",):
        try:
            status = arcpy.CheckExtension(extension)
        except Exception as exc:
            raise RuntimeError(
                f"Could not check ArcGIS {extension} license status: {exc}"
            )

        if str(status).lower() not in ("available", "checkedout", "alreadyinitialized"):
            raise RuntimeError(
                f"ArcGIS extension '{extension}' is not available "
                f"(status={status})."
            )

        try:
            result = arcpy.CheckOutExtension(extension)
            checked_out.append(extension)
            _log(f"ArcGIS extension ready: {extension} ({result})")
        except Exception as exc:
            raise RuntimeError(
                f"Could not check out ArcGIS extension '{extension}': {exc}"
            )

    return checked_out


def _checkin_extensions(extensions):
    for extension in extensions or []:
        try:
            arcpy.CheckInExtension(extension)
        except Exception:
            pass


def _safe_name(name):
    """Create a conservative Windows/ArcGIS-safe short name."""
    name = str(name).strip().strip("'\"")
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    if not name:
        name = "extent"

    # Avoid Windows reserved device names and overlong output paths.
    if name.upper() in {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }:
        name = f"extent_{name}"

    return name[:80]


def _layer_display_name(extent_layer):
    try:
        return _safe_name(arcpy.Describe(extent_layer).name)
    except Exception:
        return _safe_name(extent_layer)


def _resolve_input_imagery_layer(active_map, input_raster_name):
    """Resolve one unambiguous, usable imagery layer from the active map."""
    if active_map is None:
        raise RuntimeError("No active ArcGIS Pro map was found.")

    requested = str(input_raster_name or "").strip()
    if not requested:
        raise ValueError("Input Raster Layer Name cannot be empty.")

    try:
        all_layers = list(active_map.listLayers())
    except Exception as exc:
        raise RuntimeError(f"Could not enumerate layers in the active map: {exc}")

    exact = []
    for layer in all_layers:
        try:
            if str(layer.name).strip().casefold() == requested.casefold():
                exact.append(layer)
        except Exception:
            continue

    if not exact:
        available = []
        for layer in all_layers[:50]:
            try:
                available.append(str(layer.name))
            except Exception:
                pass
        preview = ", ".join(available) if available else "<none>"
        raise RuntimeError(
            f"Could not find an exact map layer named '{requested}'. "
            f"Available layer names (first 50): {preview}"
        )

    if len(exact) > 1:
        raise RuntimeError(
            f"The active map contains {len(exact)} layers named '{requested}'. "
            "ArcGUI will not guess which imagery source to use. Rename the intended "
            "imagery layer so its name is unique, then run again."
        )

    layer = exact[0]
    try:
        if bool(layer.isBroken):
            raise RuntimeError(
                f"The selected imagery layer '{requested}' is broken/unavailable."
            )
    except AttributeError:
        pass

    try:
        if bool(layer.isFeatureLayer):
            raise RuntimeError(
                f"Layer '{requested}' is a feature layer, not an imagery/raster layer."
            )
    except AttributeError:
        pass

    try:
        if bool(layer.isGroupLayer):
            raise RuntimeError(
                f"Layer '{requested}' is a group layer. Select/name the actual imagery layer inside the group."
            )
    except AttributeError:
        pass

    return layer


def _bool_from_tool_value(value):
    """Strictly coerce ArcGIS/Python boolean values instead of guessing."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(f"Boolean option must be True/False or 0/1; received {value!r}.")
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n", ""):
        return False
    raise ValueError(f"Boolean option must be True or False; received {value!r}.")


def _chunks(items, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]



def _utc_now_text():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _spatial_reference_fingerprint(spatial_ref):
    if spatial_ref is None:
        return "unknown"
    try:
        text = spatial_ref.exportToString()
        if text:
            return _sha256_text(text)
    except Exception:
        pass
    try:
        return _sha256_text(
            f"{spatial_ref.name}|{spatial_ref.factoryCode}|"
            f"{getattr(spatial_ref, 'type', '')}"
        )
    except Exception:
        return "unknown"


def _extent_geometry_fingerprint(extent_layer):
    """Stable fingerprint of the currently selected/filtered polygon geometry."""
    geometries = []
    try:
        with arcpy.da.SearchCursor(extent_layer, ["SHAPE@JSON"]) as cursor:
            for (shape_json,) in cursor:
                if shape_json:
                    geometries.append(str(shape_json))
    except Exception as exc:
        raise RuntimeError(
            f"Could not fingerprint extent geometry for export-cache matching: {exc}"
        )

    if not geometries:
        raise RuntimeError(
            "No usable polygon geometry was available for export-cache matching."
        )

    geometries.sort()
    return _sha256_text("\n".join(geometries))


def _layer_source_identity(layer):
    """Best-effort stable identity for the selected imagery source."""
    candidates = []
    try:
        value = str(layer.dataSource).strip()
        if value:
            candidates.append(value)
    except Exception:
        pass
    try:
        value = json.dumps(
            layer.connectionProperties,
            sort_keys=True,
            default=str,
        )
        if value:
            candidates.append(value)
    except Exception:
        pass
    try:
        candidates.append(str(layer.name))
    except Exception:
        candidates.append(str(layer))
    return _sha256_text("|".join(candidates))


def _build_export_cache_config(
    extent_layer,
    input_raster,
    input_raster_name,
    active_map,
    overlap_percent,
    optimized_large_crop,
):
    """
    Build the machine-comparable identity for reusable imagery.

    Threshold, ONNX model and batch size do not affect imagery export and are
    intentionally omitted. This is what allows a new model to reuse imagery
    exported for a previous model run.
    """
    desc = arcpy.Describe(extent_layer)
    source_sr = getattr(desc, "spatialReference", None)
    try:
        map_sr = active_map.spatialReference
    except Exception:
        map_sr = None

    config = {
        "schema_version": EXPORT_CACHE_SCHEMA_VERSION,
        "identity_version": EXPORT_CACHE_IDENTITY_VERSION,
        "extent_name": _layer_display_name(extent_layer),
        "extent_geometry_sha256": _extent_geometry_fingerprint(extent_layer),
        "extent_source_crs_sha256": _spatial_reference_fingerprint(source_sr),
        "map_crs_sha256": _spatial_reference_fingerprint(map_sr),
        "input_raster_name": str(input_raster_name or "").strip(),
        "input_raster_source_sha256": _layer_source_identity(input_raster),
        "optimized_large_crop": bool(_bool_from_tool_value(optimized_large_crop)),
        "overlap_percent": round(float(overlap_percent), 6),
        "extent_policy": "exact-selected-polygon",
        "fixed_tile_size": int(FIXED_TILE_SIZE),
        "fixed_cell_size": float(FIXED_CELL_SIZE),
        "max_large_export_pixels": int(MAX_LARGE_EXPORT_PIXELS),
    }
    return config


def _cache_id_from_config(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _cache_tag_from_config(config, cache_id=None):
    cache_id = cache_id or _cache_id_from_config(config)
    mode = "optimized" if config.get("optimized_large_crop") else "direct"
    overlap = float(config.get("overlap_percent", 0.0))
    overlap_text = (
        str(int(round(overlap)))
        if abs(overlap - round(overlap)) < 1e-9
        else str(overlap).replace(".", "p")
    )
    return f"{mode}_ov{overlap_text}_{cache_id}"


def _model_output_slug(model_path, profile=None):
    """Readable folder suffix such as segformer_last_cuda or unet_best_cuda."""
    profile = profile or detect_model_profile(model_path)
    stem = Path(str(model_path)).stem.lower()
    target = str(profile.get("provider_mode", "auto")).lower()
    target_label = target if target in ("cuda", "cpu") else "auto"

    if "old_milestone" in stem or "old-milestone" in stem:
        checkpoint = "old_milestone"
    elif re.search(r"(^|[_\-])last([_\-]|$)", stem):
        checkpoint = "last"
    elif re.search(r"(^|[_\-])best([_\-]|$)", stem):
        checkpoint = "best"
    elif profile.get("architecture") == "SegFormer MiT-B5" and (
        stem == "segformer_cuda_fp16" or stem.startswith("segformer_cpu_fp32")
    ):
        # The canonical SegFormer deployment files are the selected/best model.
        checkpoint = "best"
    else:
        checkpoint = "selected"

    family = (
        "segformer"
        if profile.get("architecture") == "SegFormer MiT-B5"
        else "unet"
    )
    return _safe_name(f"{family}_{checkpoint}_{target_label}")


def _cache_root_paths(extent_output_root, config):
    cache_id = _cache_id_from_config(config)
    cache_tag = _cache_tag_from_config(config, cache_id)
    cache_root = Path(extent_output_root) / EXPORT_CACHE_DIRNAME / cache_tag
    return {
        "cache_id": cache_id,
        "cache_tag": cache_tag,
        "cache_root": cache_root,
        "imagery_root": cache_root / "imagery",
        "manifest_path": cache_root / "export_config.json",
        "text_path": cache_root / "export_config.txt",
    }


def _manifest_relative_path(cache_root, path):
    """Store only paths contained inside the cache root."""
    path = Path(path).resolve()
    cache_root = Path(cache_root).resolve()
    try:
        relative = path.relative_to(cache_root)
    except Exception as exc:
        raise RuntimeError(
            f"Cache export path is outside its cache root: {path}"
        ) from exc
    return relative.as_posix()


def _manifest_absolute_path(cache_root, stored_path):
    stored = Path(str(stored_path))
    if stored.is_absolute():
        raise RuntimeError(
            "Absolute export paths are not allowed in r7 cache manifests."
        )
    root = Path(cache_root).resolve()
    candidate = (root / stored).resolve()
    try:
        candidate.relative_to(root)
    except Exception as exc:
        raise RuntimeError(
            f"Cache manifest path escapes its cache root: {stored_path}"
        ) from exc
    return candidate


def _quick_file_fingerprint(path, sample_bytes=65536):
    """Cheap integrity fingerprint: size plus first/last byte samples."""
    path = Path(path)
    size = int(path.stat().st_size)
    if size <= 0:
        raise RuntimeError(f"Cannot fingerprint empty file: {path}")
    sample_bytes = max(1024, int(sample_bytes))
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()

def _read_export_cache_manifest(cache_root):
    manifest_path = Path(cache_root) / "export_config.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn(f"Could not read export cache manifest {manifest_path}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_export_cache_text(cache_context, manifest):
    config = manifest.get("config", {})
    lines = [
        "ArcGUI reusable imagery export cache",
        "==================================",
        f"ArcGUI version: {ARCGUI_VERSION}",
        f"Cache ID: {cache_context['cache_id']}",
        f"Cache tag: {cache_context['cache_tag']}",
        f"Cache complete: {bool(manifest.get('complete'))}",
        f"Created UTC: {manifest.get('created_utc', '')}",
        f"Updated UTC: {manifest.get('updated_utc', '')}",
        "",
        "EXPORT-AFFECTING CONFIGURATION",
        f"Extent name: {config.get('extent_name', '')}",
        f"Extent geometry SHA256: {config.get('extent_geometry_sha256', '')}",
        f"Input raster layer: {config.get('input_raster_name', '')}",
        f"Optimized large-area export: {config.get('optimized_large_crop')}",
        f"Tile overlap (%): {config.get('overlap_percent')}",
        f"Extent policy: {config.get('extent_policy', '')}",
        f"Fixed model tile size: {config.get('fixed_tile_size')} px",
        f"Fixed export cell size: {config.get('fixed_cell_size')} map units/pixel",
        f"Configured maximum large export: {config.get('max_large_export_pixels')} px",
        f"Export mode recorded: {manifest.get('mode', '')}",
        f"Usable cached imagery TIFFs: {len(manifest.get('exports', []))}",
        f"Actual large-block cap used: {manifest.get('actual_large_export_limit', '')}",
        "",
        "NOT PART OF THE IMAGERY CACHE IDENTITY",
        "ONNX model, prediction threshold and batch size may change without",
        "requiring World Imagery to be exported again.",
        "",
        "A cache is reused only when the extent geometry and all export-affecting",
        "settings above match the current ArcGUI job and the referenced TIFFs exist.",
        "r7 does not reuse older pre-r7 cache schemas automatically.",
    ]
    text_path = Path(cache_context["text_path"])
    temp_path = text_path.with_suffix(text_path.suffix + ".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(temp_path), str(text_path))


def _write_export_cache_manifest(cache_context, manifest):
    cache_context["cache_root"].mkdir(parents=True, exist_ok=True)
    manifest["schema_version"] = EXPORT_CACHE_SCHEMA_VERSION
    manifest["cache_id"] = cache_context["cache_id"]
    manifest["cache_tag"] = cache_context["cache_tag"]
    manifest["updated_utc"] = _utc_now_text()
    if not manifest.get("created_utc"):
        manifest["created_utc"] = manifest["updated_utc"]

    manifest_path = Path(cache_context["manifest_path"])
    temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(str(temp_manifest), str(manifest_path))
    _write_export_cache_text(cache_context, manifest)


def _cache_manifest_files_exist(cache_root, manifest):
    exports = manifest.get("exports") or []
    if not exports:
        return False
    for item in exports:
        try:
            path = _manifest_absolute_path(
                cache_root, item.get("relative_path", "")
            )
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                return False
        except Exception:
            return False
    return True

def _archive_cache_root(cache_root, reason="archived"):
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = cache_root.with_name(f"{cache_root.name}_{reason}_{stamp}")
    counter = 1
    while candidate.exists():
        candidate = cache_root.with_name(
            f"{cache_root.name}_{reason}_{stamp}_{counter}"
        )
        counter += 1
    cache_root.rename(candidate)
    return candidate


def _prepare_export_cache_context(
    extent_output_root,
    config,
    reuse_matching_exports=True,
):
    paths = _cache_root_paths(extent_output_root, config)
    paths["config"] = config
    paths["extent_output_root"] = Path(extent_output_root)
    paths["reuse"] = False
    paths["manifest"] = None

    manifest = _read_export_cache_manifest(paths["cache_root"])
    exact_match = bool(
        manifest
        and manifest.get("complete") is True
        and manifest.get("config") == config
        and _cache_manifest_files_exist(paths["cache_root"], manifest)
    )

    if _bool_from_tool_value(reuse_matching_exports) and exact_match:
        paths["reuse"] = True
        paths["manifest"] = manifest
        return paths

    # Reuse disabled or the canonical cache is incomplete/corrupt. Keep a prior
    # complete cache as an archive when the user explicitly requests a refresh;
    # incomplete caches can simply be removed.
    if paths["cache_root"].exists():
        if manifest and manifest.get("complete") is True:
            reason = "refresh" if not _bool_from_tool_value(reuse_matching_exports) else "invalid"
            archived = _archive_cache_root(paths["cache_root"], reason=reason)
            if archived:
                _warn(f"Previous export cache archived to: {archived}")
        else:
            shutil.rmtree(paths["cache_root"], ignore_errors=True)

    paths["cache_root"].mkdir(parents=True, exist_ok=True)
    paths["imagery_root"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "complete": False,
        "config": config,
        "mode": "optimized" if config.get("optimized_large_crop") else "direct",
        "exports": [],
        "created_utc": _utc_now_text(),
    }
    paths["manifest"] = manifest
    _write_export_cache_manifest(paths, manifest)
    return paths


def _complete_direct_export_cache(
    cache_context,
    exported_items,
    direct_plan,
    spatial_ref,
    skipped_count=0,
):
    if not cache_context:
        return

    entries = []
    for item in exported_items:
        path = Path(item["path"])
        raster = _validate_exported_tif_basic(path, spatial_ref)
        coverage_key = list(_direct_coverage_key(raster))
        entries.append({
            "uid": str(item["uid"]),
            "relative_path": _manifest_relative_path(
                cache_context["cache_root"], path
            ),
            "coverage_key": coverage_key,
            "size_bytes": int(path.stat().st_size),
            "quick_sha256": _quick_file_fingerprint(path),
        })

    manifest = dict(cache_context.get("manifest") or {})
    manifest.update({
        "complete": True,
        "config": cache_context["config"],
        "mode": "direct",
        "exports": entries,
        "direct_plan_sha256": _direct_plan_fingerprint(direct_plan),
        "skipped_outside_count": int(skipped_count),
    })
    cache_context["manifest"] = manifest
    _write_export_cache_manifest(cache_context, manifest)

def _optimized_plan_fingerprint(blocks):
    """Stable hash of the exact retained optimized large-block plan."""
    values = []
    for block in sorted(blocks or [], key=lambda x: int(x.get("block_id", 0))):
        e = block["extent"]
        values.append({
            "block_id": int(block["block_id"]),
            "width_px": int(block["width_px"]),
            "height_px": int(block["height_px"]),
            "extent": [
                round(float(e.XMin), 6),
                round(float(e.YMin), 6),
                round(float(e.XMax), 6),
                round(float(e.YMax), 6),
            ],
        })
    return _sha256_text(json.dumps(values, sort_keys=True, separators=(",", ":")))


def _complete_optimized_export_cache(
    cache_context,
    large_exports,
    blocks,
    skipped_large_blocks,
    actual_large_export_limit,
):
    if not cache_context:
        return
    by_id = {int(block["block_id"]): block for block in blocks}
    export_ids = [int(item["block_id"]) for item in large_exports]
    if len(export_ids) != len(blocks) or len(set(export_ids)) != len(export_ids) or set(export_ids) != set(by_id):
        raise ArcGUIExportError(
            "Optimized export cache cannot be finalized because the usable exported "
            "block set does not exactly match the retained large-block plan. "
            f"Planned block ids={sorted(by_id)}, exported block ids={sorted(export_ids)}."
        )
    entries = []
    for item in large_exports:
        block_id = int(item["block_id"])
        block = by_id[block_id]
        extent = block["extent"]
        entries.append({
            "block_id": block_id,
            "relative_path": _manifest_relative_path(
                cache_context["cache_root"], item["path"]
            ),
            "planned_width_px": int(block["width_px"]),
            "planned_height_px": int(block["height_px"]),
            "planned_extent": [
                float(extent.XMin),
                float(extent.YMin),
                float(extent.XMax),
                float(extent.YMax),
            ],
            "size_bytes": int(Path(item["path"]).stat().st_size),
            "quick_sha256": _quick_file_fingerprint(item["path"]),
        })
    manifest = dict(cache_context.get("manifest") or {})
    manifest.update({
        "complete": True,
        "config": cache_context["config"],
        "mode": "optimized",
        "exports": entries,
        "optimized_plan_sha256": _optimized_plan_fingerprint(blocks),
        "skipped_large_block_count": int(len(skipped_large_blocks)),
        "actual_large_export_limit": int(actual_large_export_limit),
    })
    cache_context["manifest"] = manifest
    _write_export_cache_manifest(cache_context, manifest)


def _restore_direct_export_items(
    cache_context,
    spatial_ref=None,
    direct_plan=None,
):
    manifest = cache_context.get("manifest") or {}
    if manifest.get("mode") != "direct":
        raise RuntimeError("Cached imagery mode is not direct 1024 export.")

    if direct_plan is not None:
        expected_hash = _direct_plan_fingerprint(direct_plan)
        stored_hash = str(manifest.get("direct_plan_sha256") or "")
        if not stored_hash or stored_hash != expected_hash:
            raise RuntimeError(
                "Direct imagery cache does not match the exact current retained-tile plan."
            )

    items = []
    seen_keys = set()
    for entry in manifest.get("exports") or []:
        path = _manifest_absolute_path(
            cache_context["cache_root"], entry.get("relative_path", "")
        )
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Cached direct TIFF is missing: {path}")
        try:
            size_bytes = int(path.stat().st_size)
        except OSError as exc:
            raise RuntimeError(f"Could not inspect cached direct TIFF {path}: {exc}")
        if size_bytes <= 0:
            raise RuntimeError(f"Cached direct TIFF is empty: {path}")
        stored_size = entry.get("size_bytes")
        if stored_size is None or int(stored_size) != size_bytes:
            raise RuntimeError(
                f"Cached direct TIFF size does not match its manifest: {path}"
            )
        stored_hash = str(entry.get("quick_sha256") or "")
        if not stored_hash or stored_hash != _quick_file_fingerprint(path):
            raise RuntimeError(
                f"Cached direct TIFF content fingerprint changed: {path}"
            )

        if spatial_ref is not None:
            raster = _validate_exported_tif_basic(path, spatial_ref)
            actual_key = tuple(_direct_coverage_key(raster))
            stored_key = entry.get("coverage_key")
            if not stored_key:
                raise RuntimeError(
                    "Direct cache manifest is missing footprint metadata and cannot be safely reused."
                )
            normalized_stored = tuple(stored_key)
            if normalized_stored != actual_key:
                raise RuntimeError(
                    f"Cached direct TIFF footprint changed since cache creation: {path}"
                )
            if actual_key in seen_keys:
                raise RuntimeError(
                    f"Direct cache contains duplicate geographic coverage: {path}"
                )
            seen_keys.add(actual_key)

        items.append({"path": path, "uid": str(entry.get("uid") or path.stem)})

    if not items:
        raise RuntimeError("Direct imagery cache contains no usable TIFF entries.")
    return items

def _restore_optimized_large_exports(cache_context, spatial_ref, expected_blocks=None):
    manifest = cache_context.get("manifest") or {}
    if manifest.get("mode") != "optimized":
        raise RuntimeError("Cached imagery mode is not optimized large-area export.")

    if expected_blocks is not None:
        expected_hash = _optimized_plan_fingerprint(expected_blocks)
        stored_hash = str(manifest.get("optimized_plan_sha256") or "")
        if not stored_hash or stored_hash != expected_hash:
            raise RuntimeError(
                "Optimized imagery cache does not match the exact current large-block plan."
            )
        if len(manifest.get("exports") or []) != len(expected_blocks):
            raise RuntimeError(
                "Optimized imagery cache block count does not match the current plan."
            )

    items = []
    seen_block_ids = set()
    expected_by_id = {
        int(block["block_id"]): block for block in (expected_blocks or [])
    }

    for entry in manifest.get("exports") or []:
        block_id = int(entry.get("block_id", -1))
        if block_id <= 0 or block_id in seen_block_ids:
            raise RuntimeError(
                f"Optimized cache contains an invalid/duplicate block id: {block_id}"
            )
        seen_block_ids.add(block_id)

        if expected_blocks is not None:
            if block_id not in expected_by_id:
                raise RuntimeError(
                    f"Optimized cache contains block {block_id} that is not in the current plan."
                )
            expected_block = expected_by_id[block_id]
            expected_extent = expected_block["extent"]
            stored_extent = entry.get("planned_extent") or []
            expected_values = [
                round(float(expected_extent.XMin), 6),
                round(float(expected_extent.YMin), 6),
                round(float(expected_extent.XMax), 6),
                round(float(expected_extent.YMax), 6),
            ]
            if len(stored_extent) != 4 or [round(float(v), 6) for v in stored_extent] != expected_values:
                raise RuntimeError(
                    f"Optimized cache block {block_id} extent does not match the current plan."
                )
            if (
                int(entry.get("planned_width_px", -1)) != int(expected_block["width_px"])
                or int(entry.get("planned_height_px", -1)) != int(expected_block["height_px"])
            ):
                raise RuntimeError(
                    f"Optimized cache block {block_id} dimensions do not match the current plan."
                )

        path = _manifest_absolute_path(
            cache_context["cache_root"], entry.get("relative_path", "")
        )
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Cached optimized TIFF is missing: {path}")
        size_bytes = int(path.stat().st_size)
        if size_bytes <= 0 or entry.get("size_bytes") is None or int(entry["size_bytes"]) != size_bytes:
            raise RuntimeError(f"Cached optimized TIFF size does not match its manifest: {path}")
        stored_hash = str(entry.get("quick_sha256") or "")
        if not stored_hash or stored_hash != _quick_file_fingerprint(path):
            raise RuntimeError(f"Cached optimized TIFF content fingerprint changed: {path}")
        planned_extent = entry.get("planned_extent") or []
        if len(planned_extent) != 4:
            raise RuntimeError(f"Cached block has no valid planned extent: {path}")
        extent = arcpy.Extent(*[float(v) for v in planned_extent])
        _validate_exported_tif_georeference(
            path,
            extent,
            spatial_ref,
            expected_width_px=int(entry["planned_width_px"]),
            expected_height_px=int(entry["planned_height_px"]),
        )
        if _tif_has_nonblack_pixels(path) is not True:
            raise ArcGUIImageryContentError(
                f"Cached optimized TIFF is black or unreadable: {path}"
            )
        items.append({
            "block_id": block_id,
            "path": path,
            "planned_width_px": int(entry["planned_width_px"]),
            "planned_height_px": int(entry["planned_height_px"]),
        })
    if not items:
        raise RuntimeError("Optimized imagery cache contains no usable TIFF entries.")
    return items




def _refresh_cache_context_after_runtime_validation_failure(cache_context, reason):
    """Archive a bad cache discovered during production preflight and create a fresh one."""
    _warn(
        "Existing export cache failed runtime validation and will NOT be reused. "
        f"A fresh World Imagery export will be created. Details: {reason}"
    )
    try:
        archived = _archive_cache_root(cache_context["cache_root"], reason="invalid_runtime")
        if archived:
            _warn(f"Invalid cache archived to: {archived}")
    except Exception as exc:
        _warn(f"Could not archive invalid cache; it will be replaced: {exc}")
        shutil.rmtree(cache_context["cache_root"], ignore_errors=True)

    return _prepare_export_cache_context(
        extent_output_root=cache_context["extent_output_root"],
        config=cache_context["config"],
        reuse_matching_exports=False,
    )


def _runtime_preflight_cached_imagery(
    cache_context,
    optimized_large_crop,
    spatial_ref,
    direct_plan=None,
    export_extent=None,
    mask_geometries=None,
):
    """
    Validate the reusable cache before ONNX loading.

    Optimized caches are small enough to fully validate georeferencing here.
    Direct caches can contain thousands of 1024 TIFFs, so the preflight checks
    every file path, size, lightweight content fingerprint, georeference footprint,
    and the exact retained-tile plan before ONNX loading. Full RGB content is
    decoded naturally during inference.
    """
    if not cache_context or not cache_context.get("reuse"):
        return cache_context

    try:
        if _bool_from_tool_value(optimized_large_crop):
            manifest = cache_context.get("manifest") or {}
            limit = int(manifest.get("actual_large_export_limit") or MAX_LARGE_EXPORT_PIXELS)
            if export_extent is None or not mask_geometries:
                raise RuntimeError(
                    "Optimized cache preflight is missing the current extent geometry/plan context."
                )
            all_blocks = _plan_large_export_blocks(
                export_extent,
                spatial_ref,
                mask_geometries=mask_geometries,
                include_nonintersecting=True,
                maximum_pixels=limit,
            )
            expected_blocks = [
                block for block in all_blocks if block["intersects_requested_extent"]
            ]
            if not expected_blocks:
                raise RuntimeError(
                    "The current optimized plan contains no block intersecting the requested extent."
                )
            expected_skipped = len(all_blocks) - len(expected_blocks)
            stored_skipped = int(manifest.get("skipped_large_block_count", -1))
            if stored_skipped != expected_skipped:
                raise RuntimeError(
                    "Optimized cache skipped-block count does not match the current plan."
                )
            _restore_optimized_large_exports(
                cache_context,
                spatial_ref,
                expected_blocks=expected_blocks,
            )
        else:
            items = _restore_direct_export_items(
                cache_context,
                spatial_ref=spatial_ref,
                direct_plan=direct_plan,
            )
            if direct_plan is not None:
                expected = len(direct_plan.get("active_crops") or [])
                if len(items) != expected:
                    raise RuntimeError(
                        f"direct cache contains {len(items)} TIFF(s), but the "
                        f"current exact plan requires {expected}."
                    )
        return cache_context
    except Exception as exc:
        return _refresh_cache_context_after_runtime_validation_failure(
            cache_context,
            exc,
        )


def _write_model_run_info(
    run_root,
    model_path,
    model_profile,
    effective_target,
    threshold,
    batch_size,
    cache_context,
):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "ArcGUI model run",
        "================",
        f"ArcGUI version: {ARCGUI_VERSION}",
        f"Created UTC: {_utc_now_text()}",
        f"Model file: {Path(str(model_path)).name}",
        f"Model path: {model_path}",
        f"Model size bytes: {Path(str(model_path)).stat().st_size}",
        f"Model quick SHA256: {_quick_file_fingerprint(model_path)}",
        f"Architecture: {model_profile.get('architecture', '')}",
        f"Execution target: {str(effective_target).upper()}",
        f"Prediction threshold: {float(threshold):.6f}",
        f"ONNX batch size: {int(batch_size)}",
        f"Export cache tag: {cache_context.get('cache_tag', '')}",
        f"Export cache reused: {bool(cache_context.get('reuse'))}",
        f"Export config: {cache_context.get('text_path', '')}",
    ]
    (run_root / "model_run_info.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def overlap_percent_to_stride(overlap_percent):
    overlap_percent = float(overlap_percent)

    if not 0.0 <= overlap_percent <= 95.0:
        raise ValueError("Tile overlap must be between 0 and 95 percent.")

    stride = int(round(FIXED_TILE_SIZE * (1.0 - overlap_percent / 100.0)))
    return max(1, stride)


def _sliding_starts(length_pixels, window_pixels, stride_pixels):
    """
    Sliding-window starts that cover the full axis without creating
    unnecessary windows extending beyond the far edge.

    The final start is forced to length-window when the normal stride
    sequence would otherwise miss the edge.
    """
    length_pixels = max(1, int(math.ceil(float(length_pixels))))
    window_pixels = max(1, int(window_pixels))
    stride_pixels = max(1, int(stride_pixels))

    if length_pixels <= window_pixels:
        return [0]

    starts = list(range(0, length_pixels - window_pixels + 1, stride_pixels))
    last = length_pixels - window_pixels

    if starts[-1] != last:
        starts.append(last)

    return starts


def _extent_pixel_size(extent):
    """
    Convert a map extent to pixel dimensions at FIXED_CELL_SIZE.

    ArcGIS can occasionally report NaN extent metadata for a newly projected
    in-memory feature class even though the feature geometries themselves are
    valid. Detect that explicitly instead of allowing the cryptic
    "cannot convert float NaN to integer" exception.
    """
    values = [
        float(extent.XMin),
        float(extent.YMin),
        float(extent.XMax),
        float(extent.YMax),
    ]

    if not all(math.isfinite(v) for v in values):
        raise RuntimeError(
            "ArcGIS returned a non-finite planning extent "
            f"(XMin={values[0]}, YMin={values[1]}, "
            f"XMax={values[2]}, YMax={values[3]})."
        )

    width_map = values[2] - values[0]
    height_map = values[3] - values[1]

    if width_map <= 0 or height_map <= 0:
        raise RuntimeError(
            "ArcGIS returned an invalid planning extent with non-positive "
            f"size: width={width_map}, height={height_map}."
        )

    width = int(math.ceil(width_map / FIXED_CELL_SIZE))
    height = int(math.ceil(height_map / FIXED_CELL_SIZE))

    return max(1, width), max(1, height)


def _rect_polygon(xmin, ymin, xmax, ymax, spatial_ref):
    ring = arcpy.Array([
        arcpy.Point(xmin, ymin),
        arcpy.Point(xmin, ymax),
        arcpy.Point(xmax, ymax),
        arcpy.Point(xmax, ymin),
        arcpy.Point(xmin, ymin),
    ])
    return arcpy.Polygon(ring, spatial_ref)


def _validate_polygon_extent_layer(extent_layer):
    """Validate that the processing extent is a polygon feature layer/class."""
    desc = arcpy.Describe(extent_layer)

    try:
        shape_type = str(desc.shapeType).strip().lower()
    except Exception:
        shape_type = ""

    if shape_type != "polygon":
        raise ValueError(
            "Extent Layer(s) must be polygon geometry. "
            f"Received shape type: {shape_type or 'unknown'} for {extent_layer}."
        )

    spatial_ref = getattr(desc, "spatialReference", None)
    if not _is_valid_spatial_reference(spatial_ref):
        raise ValueError(
            "The extent layer has an Unknown/undefined coordinate system. "
            "For deployment safety ArcGUI r7 no longer guesses the CRS. "
            "Define the correct coordinate system on the extent dataset before running."
        )

    return desc



def _materialize_extent_layer(extent_layer, scratch_tag):
    """
    Copy the input feature layer to an in-memory feature class.

    CopyFeatures preserves the current selection and definition query.
    The temporary copy is repaired so null/self-inconsistent polygon geometry
    cannot silently break intersection tests later in the pipeline.
    """
    temp_name = (
        f"arcgui_input_extent_{_safe_name(scratch_tag)}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    temp_fc = rf"in_memory\{temp_name}"

    arcpy.management.CopyFeatures(extent_layer, temp_fc)

    try:
        arcpy.management.RepairGeometry(
            temp_fc,
            delete_null="DELETE_NULL",
        )
    except TypeError:
        # Compatibility with ArcGIS builds that expose the older positional
        # signature.
        try:
            arcpy.management.RepairGeometry(temp_fc, "DELETE_NULL")
        except Exception as exc:
            try:
                arcpy.management.Delete(temp_fc)
            except Exception:
                pass
            raise RuntimeError(
                "ArcGUI could not repair/validate the temporary extent geometry. "
                f"Repair the source polygon dataset and retry. Details: {exc}"
            ) from exc
    except Exception as exc:
        try:
            arcpy.management.Delete(temp_fc)
        except Exception:
            pass
        raise RuntimeError(
            "ArcGUI could not repair/validate the temporary extent geometry. "
            f"Repair the source polygon dataset and retry. Details: {exc}"
        ) from exc

    try:
        count = int(arcpy.management.GetCount(temp_fc).getOutput(0))
    except Exception as exc:
        try:
            arcpy.management.Delete(temp_fc)
        except Exception:
            pass
        raise RuntimeError(
            f"Could not count polygons in the materialized extent: {exc}"
        )

    if count <= 0:
        try:
            arcpy.management.Delete(temp_fc)
        except Exception:
            pass
        raise ValueError(
            f"Extent layer contains no usable polygons to process: {extent_layer}"
        )

    return temp_fc, temp_fc


def _geometry_intersects_any(polygon, geometries):
    """
    Return True when a footprint intersects any requested polygon.

    Missing or unreadable extent geometries are treated as a hard error rather
    than "intersects everything"; otherwise an extent-reading failure could
    cause ArcGUI to export/infer an entire bounding rectangle unintentionally.
    """
    if not geometries:
        raise RuntimeError(
            "No usable requested-extent geometries are available for "
            "intersection testing."
        )

    for geom in geometries:
        try:
            if not polygon.disjoint(geom):
                return True
        except Exception as exc:
            raise RuntimeError(
                "ArcGIS could not evaluate a polygon intersection relation. "
                f"Repair/validate the extent geometry and retry. Details: {exc}"
            )

    return False

def _spatial_reference_label(spatial_ref):
    if spatial_ref is None:
        return "<None>"

    try:
        code = int(spatial_ref.factoryCode)
    except Exception:
        code = 0

    try:
        name = str(spatial_ref.name)
    except Exception:
        name = "Unknown"

    if code > 0:
        return f"{name} (EPSG/WKID {code})"
    return name


def _spatial_references_equivalent(a, b):
    """
    Best-effort ArcGIS spatial-reference equivalence test.

    Do not immediately reject two spatial references only because their WKIDs
    differ. ArcGIS can expose equivalent CRS aliases with different historical
    WKIDs (for example Web Mercator aliases).
    """
    if a is None or b is None:
        return False

    try:
        a_code = int(a.factoryCode)
        b_code = int(b.factoryCode)
        if a_code > 0 and b_code > 0 and a_code == b_code:
            return True
    except Exception:
        pass

    try:
        if a.exportToString() == b.exportToString():
            return True
    except Exception:
        pass

    try:
        return str(a.name).strip().lower() == str(b.name).strip().lower()
    except Exception:
        return False



def _is_valid_spatial_reference(spatial_ref):
    """Return True when ArcPy supplied a usable, non-Unknown spatial reference."""
    if spatial_ref is None:
        return False

    try:
        name = str(spatial_ref.name).strip().lower()
        if name in ("", "unknown", "none"):
            return False
    except Exception:
        pass

    try:
        # A projected/geographic SR may have factoryCode 0 when custom, so do
        # not reject only because factoryCode is zero.
        spatial_ref.exportToString()
    except Exception:
        # Some ArcPy SR objects still work even if exportToString is unavailable.
        pass

    return True


def _validate_fixed_cell_export_crs(spatial_ref):
    """
    ArcGUI's fixed 2.0 cell size is defined in metres.

    Fail fast for geographic (degree) maps or projected CRSs whose map units
    are not metres. Otherwise a nominal "2.0" cell could mean 2 degrees or
    2 feet and silently produce the wrong ground resolution.
    """
    if not _is_valid_spatial_reference(spatial_ref):
        raise RuntimeError("The selected map/export CRS is not usable.")

    try:
        sr_type = str(spatial_ref.type).strip().lower()
    except Exception:
        sr_type = ""

    if sr_type and sr_type != "projected":
        raise RuntimeError(
            "ArcGUI requires a projected map/export coordinate system because "
            "the deployment uses a fixed 2.0 metre cell size. "
            f"Current CRS: {_spatial_reference_label(spatial_ref)}. "
            "Change the active map to a projected metric CRS such as UTM or "
            "Web Mercator before running ArcGUI."
        )

    try:
        meters_per_unit = float(spatial_ref.metersPerUnit)
    except Exception:
        meters_per_unit = float("nan")

    if (
        not math.isfinite(meters_per_unit)
        or abs(meters_per_unit - 1.0) > 1e-6
    ):
        raise RuntimeError(
            "ArcGUI requires map units in metres because FIXED_CELL_SIZE=2.0. "
            f"Current CRS: {_spatial_reference_label(spatial_ref)}; "
            f"metersPerUnit={meters_per_unit}. "
            "Use a projected metric map CRS."
        )

    return spatial_ref


def _get_export_spatial_reference(
    raster_layer,
    active_map=None,
    fallback_spatial_ref=None,
):
    """
    Choose the CRS used for ArcGUI MAP_SPACE imagery export and block planning.

    Priority:
      1) Active map spatial reference.
         This is the most appropriate CRS for reference_system="MAP_SPACE",
         and basemap/web-tile layers such as World Imagery often do not expose
         a normal raster spatialReference through Describe/Raster.
      2) Spatial reference exposed directly by the imagery layer.
      3) Explicit fallback spatial reference (normally the extent layer CRS).
      4) For a layer named World Imagery, Web Mercator Auxiliary Sphere
         (EPSG/WKID 3857) as a final, service-specific fallback.

    The selected extent is projected into this CRS before any block coordinates
    are computed. The same CRS is also passed as outputCoordinateSystem during
    ExportTrainingDataForDeepLearning, so coordinate numbers and CRS tags stay
    consistent.
    """
    # 1) Active map CRS -- preferred for MAP_SPACE exports.
    if active_map is not None:
        try:
            map_sr = active_map.spatialReference
            if _is_valid_spatial_reference(map_sr):
                _log(
                    "Using active map spatial reference for imagery/export "
                    f"planning: {_spatial_reference_label(map_sr)}"
                )
                return map_sr
        except Exception:
            pass

    # 2) Imagery layer-reported CRS.
    candidates = []

    try:
        candidates.append(arcpy.Describe(raster_layer).spatialReference)
    except Exception:
        pass

    try:
        candidates.append(arcpy.Raster(raster_layer).spatialReference)
    except Exception:
        pass

    for sr in candidates:
        if _is_valid_spatial_reference(sr):
            _log(
                "Using imagery-layer spatial reference for export planning: "
                f"{_spatial_reference_label(sr)}"
            )
            return sr

    # 3) Extent/source CRS fallback. World Imagery can be reprojected to the
    # requested MAP_SPACE output CRS by ArcGIS.
    if _is_valid_spatial_reference(fallback_spatial_ref):
        _warn(
            "The imagery layer did not expose a spatial reference. "
            "Using the extent layer spatial reference as the MAP_SPACE export "
            f"CRS: {_spatial_reference_label(fallback_spatial_ref)}"
        )
        return fallback_spatial_ref

    # 4) Service-specific final fallback for Esri World Imagery.
    try:
        layer_name = str(raster_layer.name).strip().lower()
    except Exception:
        layer_name = str(raster_layer).strip().lower()

    if "world imagery" in layer_name:
        sr = arcpy.SpatialReference(3857)
        _warn(
            "Neither the active map nor the World Imagery layer exposed a usable "
            "spatial reference. Falling back to Web Mercator Auxiliary Sphere "
            "(EPSG/WKID 3857) for World Imagery."
        )
        return sr

    raise RuntimeError(
        "Could not determine a usable map/export spatial reference. "
        "Open the imagery in an ArcGIS Pro map with a defined coordinate system, "
        "or use an extent layer with a defined coordinate system."
    )


def _project_feature_class_to_spatial_ref(
    feature_class,
    target_spatial_ref,
    scratch_tag,
):
    """
    Project the temporary extent feature class to the map/export CRS.

    The source CRS must be defined. ArcGUI r7 refuses to guess an Unknown CRS
    because a wrong assumption can produce correctly shaped but geographically
    misplaced exports and predictions.
    """
    desc = arcpy.Describe(feature_class)
    source_sr = desc.spatialReference

    if not _is_valid_spatial_reference(target_spatial_ref):
        raise RuntimeError(
            "The selected map/export coordinate system is unknown or invalid."
        )

    if not _is_valid_spatial_reference(source_sr):
        raise RuntimeError(
            "The extent layer coordinate system is Unknown/undefined. ArcGUI r7 "
            "does not assume that unknown coordinates already match the map CRS. "
            "Define the correct source CRS before processing."
        )

    if _spatial_references_equivalent(source_sr, target_spatial_ref):
        return feature_class, None

    temp_name = (
        f"arcgui_projected_extent_{_safe_name(scratch_tag)}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    temp_fc = rf"in_memory\{temp_name}"

    transformation = None
    try:
        transformations = arcpy.ListTransformations(
            source_sr,
            target_spatial_ref,
            desc.extent,
        )
        if transformations:
            transformation = transformations[0]
    except Exception as exc:
        _warn(
            "Could not enumerate geographic transformations; ArcGIS will use "
            f"its default projection behavior. Details: {exc}"
        )

    _log(
        "Projecting requested extent into imagery/export CRS: "
        f"{_spatial_reference_label(source_sr)} -> "
        f"{_spatial_reference_label(target_spatial_ref)}"
    )

    if transformation:
        _log(f"Geographic transformation: {transformation}")

    try:
        arcpy.management.Project(
            in_dataset=feature_class,
            out_dataset=temp_fc,
            out_coor_system=target_spatial_ref,
            transform_method=transformation,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not project the requested extent into the map/export CRS. "
            f"Source={_spatial_reference_label(source_sr)}, "
            f"target={_spatial_reference_label(target_spatial_ref)}. "
            f"Details: {exc}"
        )

    return temp_fc, temp_fc


def _validate_exported_tif_georeference(
    tif_path,
    planned_extent,
    expected_spatial_ref,
    expected_width_px=None,
    expected_height_px=None,
):
    """
    Validate an optimized ArcGIS export without being brittle to harmless
    service/grid alignment.

    The check is deliberately two-tiered:
      * cell size / dimensions / coverage must be close to the 2 m plan;
      * raster centre and outer bounds may differ by a small alignment tolerance.

    Large CRS-tag mistakes or UTM-vs-Web-Mercator coordinate mismatches still
    fail by orders of magnitude.
    """
    raster = arcpy.Raster(str(tif_path))
    actual_sr = raster.spatialReference

    if not _is_valid_spatial_reference(actual_sr):
        raise ArcGUIGeoreferenceError(
            f"Exported TIFF has an unknown CRS: {tif_path}."
        )

    if not _spatial_references_equivalent(actual_sr, expected_spatial_ref):
        raise ArcGUIGeoreferenceError(
            "Exported TIFF spatial reference mismatch. "
            f"Expected {_spatial_reference_label(expected_spatial_ref)}, got "
            f"{_spatial_reference_label(actual_sr)} for {tif_path}."
        )

    try:
        cell_w = abs(float(raster.meanCellWidth))
        cell_h = abs(float(raster.meanCellHeight))
        actual_width = int(raster.width)
        actual_height = int(raster.height)
        actual = raster.extent
        actual_values = [
            float(actual.XMin),
            float(actual.YMin),
            float(actual.XMax),
            float(actual.YMax),
        ]
    except Exception as exc:
        raise ArcGUIGeoreferenceError(
            f"Could not read georeference metadata from exported TIFF "
            f"{tif_path}: {exc}"
        )

    if (
        actual_width <= 0
        or actual_height <= 0
        or not math.isfinite(cell_w)
        or not math.isfinite(cell_h)
        or cell_w <= 0
        or cell_h <= 0
        or not all(math.isfinite(v) for v in actual_values)
    ):
        raise ArcGUIGeoreferenceError(
            f"Exported TIFF has invalid/non-finite georeference metadata: "
            f"{tif_path}."
        )

    max_cell_error = FIXED_CELL_SIZE * CELL_SIZE_RELATIVE_TOLERANCE
    if (
        abs(cell_w - FIXED_CELL_SIZE) > max_cell_error
        or abs(cell_h - FIXED_CELL_SIZE) > max_cell_error
    ):
        raise ArcGUIGeoreferenceError(
            "Exported TIFF resolution is too different from the fixed "
            f"{FIXED_CELL_SIZE} m/pixel deployment resolution. "
            f"Got {cell_w:.6f} x {cell_h:.6f} for {tif_path}. "
            "Check the map CRS, image-service scale, and geoprocessing "
            "cell-size environment."
        )

    planned_values = [
        float(planned_extent.XMin),
        float(planned_extent.YMin),
        float(planned_extent.XMax),
        float(planned_extent.YMax),
    ]
    if not all(math.isfinite(v) for v in planned_values):
        raise ArcGUIGeoreferenceError(
            "Planned export extent contains non-finite coordinates."
        )

    planned_w = planned_values[2] - planned_values[0]
    planned_h = planned_values[3] - planned_values[1]
    actual_w_map = actual_values[2] - actual_values[0]
    actual_h_map = actual_values[3] - actual_values[1]

    if planned_w <= 0 or planned_h <= 0 or actual_w_map <= 0 or actual_h_map <= 0:
        raise ArcGUIGeoreferenceError(
            f"Invalid planned/actual raster extent for {tif_path}."
        )

    x_tol = max(
        cell_w * EXPORT_ALIGNMENT_TOLERANCE_PIXELS,
        planned_w * EXPORT_ALIGNMENT_TOLERANCE_FRACTION,
    )
    y_tol = max(
        cell_h * EXPORT_ALIGNMENT_TOLERANCE_PIXELS,
        planned_h * EXPORT_ALIGNMENT_TOLERANCE_FRACTION,
    )

    planned_cx = (planned_values[0] + planned_values[2]) / 2.0
    planned_cy = (planned_values[1] + planned_values[3]) / 2.0
    actual_cx = (actual_values[0] + actual_values[2]) / 2.0
    actual_cy = (actual_values[1] + actual_values[3]) / 2.0

    coverage_errors = {
        "center_x": abs(actual_cx - planned_cx),
        "center_y": abs(actual_cy - planned_cy),
        "width": abs(actual_w_map - planned_w),
        "height": abs(actual_h_map - planned_h),
    }

    if (
        coverage_errors["center_x"] > x_tol
        or coverage_errors["center_y"] > y_tol
        or coverage_errors["width"] > 2.0 * x_tol
        or coverage_errors["height"] > 2.0 * y_tol
    ):
        raise ArcGUIGeoreferenceError(
            "Exported TIFF coverage does not match the planned block within "
            "the allowed raster-alignment tolerance. "
            f"Planned extent=({planned_values[0]}, {planned_values[1]}, "
            f"{planned_values[2]}, {planned_values[3]}); "
            f"actual extent=({actual_values[0]}, {actual_values[1]}, "
            f"{actual_values[2]}, {actual_values[3]}); "
            f"CRS={_spatial_reference_label(actual_sr)}."
        )

    if expected_width_px is not None:
        width_tol_px = max(
            8,
            int(math.ceil(
                int(expected_width_px) * EXPORT_DIMENSION_TOLERANCE_FRACTION
            )),
        )
        if abs(actual_width - int(expected_width_px)) > width_tol_px:
            raise ArcGUIGeoreferenceError(
                f"Exported TIFF width differs too much from the planned block: "
                f"planned {int(expected_width_px)} px, got {actual_width} px "
                f"for {tif_path}."
            )

    if expected_height_px is not None:
        height_tol_px = max(
            8,
            int(math.ceil(
                int(expected_height_px) * EXPORT_DIMENSION_TOLERANCE_FRACTION
            )),
        )
        if abs(actual_height - int(expected_height_px)) > height_tol_px:
            raise ArcGUIGeoreferenceError(
                f"Exported TIFF height differs too much from the planned block: "
                f"planned {int(expected_height_px)} px, got {actual_height} px "
                f"for {tif_path}."
            )

    # Informative warning for exports that needed the relaxed alignment margin.
    strict_x = cell_w * 4.0
    strict_y = cell_h * 4.0
    bound_errors = [
        abs(actual_values[0] - planned_values[0]),
        abs(actual_values[1] - planned_values[1]),
        abs(actual_values[2] - planned_values[2]),
        abs(actual_values[3] - planned_values[3]),
    ]
    if (
        bound_errors[0] > strict_x
        or bound_errors[2] > strict_x
        or bound_errors[1] > strict_y
        or bound_errors[3] > strict_y
    ):
        _warn(
            f"ArcGIS aligned {Path(tif_path).name} slightly differently from "
            "the exact planned bounds, but the raster remains within the safe "
            "coverage/CRS tolerance."
        )

    return raster


def _validate_exported_tif_basic(tif_path, expected_spatial_ref):
    """Basic CRS/resolution/finite-extent validation for direct-export tiles."""
    raster = arcpy.Raster(str(tif_path))

    actual_sr = raster.spatialReference
    if (
        not _is_valid_spatial_reference(actual_sr)
        or not _spatial_references_equivalent(
            actual_sr,
            expected_spatial_ref,
        )
    ):
        raise ArcGUIGeoreferenceError(
            f"Direct-export TIFF has an unexpected/unknown CRS: {tif_path}."
        )

    try:
        width = int(raster.width)
        height = int(raster.height)
        values = [
            float(raster.extent.XMin),
            float(raster.extent.YMin),
            float(raster.extent.XMax),
            float(raster.extent.YMax),
            abs(float(raster.meanCellWidth)),
            abs(float(raster.meanCellHeight)),
        ]
    except Exception as exc:
        raise ArcGUIGeoreferenceError(
            f"Could not read direct-export TIFF metadata for {tif_path}: {exc}"
        )

    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in values):
        raise ArcGUIGeoreferenceError(
            f"Direct-export TIFF has invalid/non-finite metadata: {tif_path}."
        )

    max_cell_error = FIXED_CELL_SIZE * CELL_SIZE_RELATIVE_TOLERANCE
    if (
        values[4] <= 0
        or values[5] <= 0
        or abs(values[4] - FIXED_CELL_SIZE) > max_cell_error
        or abs(values[5] - FIXED_CELL_SIZE) > max_cell_error
    ):
        raise ArcGUIGeoreferenceError(
            f"Direct-export TIFF has unexpected cell size "
            f"{values[4]} x {values[5]} instead of approximately "
            f"{FIXED_CELL_SIZE} x {FIXED_CELL_SIZE}: {tif_path}."
        )

    return raster

# =========================================================
# MODEL DETECTION / SESSION
# =========================================================


def _checkpoint_label_from_filename(name):
    """Return a short human-readable checkpoint label derived from a filename."""
    stem = Path(str(name)).stem.lower()
    if "old_milestone" in stem or "old-milestone" in stem:
        return "Old milestone"
    if re.search(r"(^|[_\-])last([_\-]|$)", stem):
        return "Last checkpoint"
    if re.search(r"(^|[_\-])best([_\-]|$)", stem):
        return "Best checkpoint"
    return "Selected model"


def detect_model_profile(model_path):
    """
    Detect the model family and intended execution target from the ONNX filename.

    ArcGUI no longer requires one exact filename.  The ONNX graph is still validated
    after loading, so accepting a filename family does not weaken the input/output
    contract checks.

    Recognized families:
      * SegFormer MiT-B5: filename contains "segformer"
      * U-Net ResNet101: filename contains "unet" and "resnet101" or "r101"

    Execution target inferred from filename:
      * contains "cpu"             -> CPU only
      * contains "cuda" or "fp16" -> CUDA required
      * otherwise                  -> CUDA preferred, CPU fallback
    """
    if model_path is None or not str(model_path).strip():
        raise ValueError("No ONNX model was selected.")

    path = Path(str(model_path))
    name = path.name.lower()
    stem = path.stem.lower()

    if path.suffix.lower() != ".onnx":
        raise ValueError(
            f"ArcGUI requires an ONNX model (.onnx); received: {path.name}"
        )

    is_segformer = "segformer" in stem
    is_unet_r101 = (
        "unet" in stem
        and ("resnet101" in stem or "r101" in stem)
    )

    if not is_segformer and not is_unet_r101:
        examples = "\n".join(f"  {item}" for item in KNOWN_MODEL_EXAMPLES)
        raise ValueError(
            "Unrecognized ArcGUI ONNX filename.\n"
            "Accepted filename families are:\n"
            "  * SegFormer MiT-B5: filename contains 'segformer'\n"
            "  * U-Net ResNet101: filename contains 'unet' and 'resnet101' or 'r101'\n"
            "Examples currently supported:\n"
            f"{examples}"
        )

    # Explicit CPU marker wins over all other filename hints.
    if "cpu" in stem:
        provider_mode = "cpu"
    elif "cuda" in stem or "fp16" in stem:
        provider_mode = "cuda"
    else:
        provider_mode = "auto"

    checkpoint_label = _checkpoint_label_from_filename(name)

    if is_segformer:
        target_label = {
            "cuda": "CUDA",
            "cpu": "CPU",
            "auto": "CUDA preferred / CPU fallback",
        }[provider_mode]
        return {
            "key": "segformer",
            "architecture": "SegFormer MiT-B5",
            "display_name": f"SegFormer MiT-B5 - {checkpoint_label} ({target_label})",
            "provider_mode": provider_mode,
            "default_batch_size": 1 if provider_mode == "cpu" else 2,
            "output_semantics": "logits",
            "filename": name,
        }

    target_label = {
        "cuda": "CUDA",
        "cpu": "CPU",
        "auto": "CUDA preferred / CPU fallback",
    }[provider_mode]
    return {
        "key": "unet_resnet101",
        "architecture": "U-Net ResNet101",
        "display_name": f"U-Net ResNet101 - {checkpoint_label} ({target_label})",
        "provider_mode": provider_mode,
        "default_batch_size": 1 if provider_mode == "cpu" else 4,
        "output_semantics": "logits",
        "filename": name,
    }

def detect_model_target(model_path):
    return detect_model_profile(model_path)["provider_mode"]



def _build_onnx_session(model_path):
    model_path = Path(str(model_path))

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not model_path.is_file():
        raise ValueError(f"Selected ONNX path is not a file: {model_path}")
    try:
        if model_path.stat().st_size <= 0:
            raise ValueError(f"Selected ONNX model file is empty: {model_path}")
    except OSError as exc:
        raise RuntimeError(f"Could not inspect selected ONNX model file: {exc}")

    profile = detect_model_profile(model_path)
    provider_mode = profile["provider_mode"]
    available = ort.get_available_providers()

    _log(f"Detected model: {profile['display_name']}")
    _log(f"Available ONNX Runtime providers: {available}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider_mode in ("cuda", "auto") and "CUDAExecutionProvider" in available:
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception as exc:
            _warn(f"CUDA DLL preload warning: {exc}")

    if provider_mode == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"{model_path.name} is marked as a CUDA model and requires CUDAExecutionProvider."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        effective_target = "cuda"

    elif provider_mode == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is unavailable.")
        providers = ["CPUExecutionProvider"]
        effective_target = "cpu"

    else:
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            effective_target = "cuda"
        elif "CPUExecutionProvider" in available:
            providers = ["CPUExecutionProvider"]
            effective_target = "cpu"
            _warn(f"{profile['display_name']} is running on CPU because CUDA is unavailable.")
        else:
            raise RuntimeError(
                "No supported ONNX Runtime execution provider is available."
            )

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not create ONNX Runtime session for {model_path.name}: {exc}"
        )

    active = session.get_providers()
    _log(f"Requested providers: {providers}")
    _log(f"Active providers: {active}")

    if provider_mode == "cuda" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"CUDA model {model_path.name} did not activate CUDAExecutionProvider."
        )

    if provider_mode == "cpu" and (
        not active or active[0] != "CPUExecutionProvider"
    ):
        raise RuntimeError(
            f"CPU model {model_path.name} did not activate CPUExecutionProvider."
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"ArcGUI expects exactly one ONNX image input; model exposes "
            f"{len(inputs)} inputs."
        )

    if len(outputs) != 1:
        raise RuntimeError(
            f"ArcGUI expects exactly one ONNX segmentation output; model exposes "
            f"{len(outputs)} outputs."
        )

    input_meta = inputs[0]
    output_meta = outputs[0]

    _log(f"ONNX input:  {input_meta.name}, {input_meta.type}, {input_meta.shape}")
    _log(f"ONNX output: {output_meta.name}, {output_meta.type}, {output_meta.shape}")
    _log(
        "ONNX output contract: single-channel LOGITS; ArcGUI applies sigmoid "
        "before overlap averaging."
    )

    if input_meta.type != "tensor(float)":
        raise RuntimeError(
            f"Expected FP32 ONNX input tensor(float); got {input_meta.type}."
        )

    in_shape = input_meta.shape
    if len(in_shape) != 4:
        raise RuntimeError(
            f"Expected NCHW [B,3,H,W] ONNX input; got {in_shape}."
        )

    if isinstance(in_shape[1], int) and in_shape[1] != 3:
        raise RuntimeError(
            f"Expected 3 RGB input channels; got shape {in_shape}."
        )

    if isinstance(in_shape[2], int) and in_shape[2] != FIXED_MODEL_INPUT_SIZE:
        raise RuntimeError(
            f"Expected input height 1024; got {in_shape[2]}."
        )

    if isinstance(in_shape[3], int) and in_shape[3] != FIXED_MODEL_INPUT_SIZE:
        raise RuntimeError(
            f"Expected input width 1024; got {in_shape[3]}."
        )

    if isinstance(in_shape[0], int):
        if in_shape[0] <= 0:
            raise RuntimeError(
                f"Invalid static ONNX batch dimension: {in_shape[0]}."
            )
        if in_shape[0] != 1:
            raise RuntimeError(
                "This ArcGUI build supports dynamic batch models or static "
                "batch=1 models. "
                f"The selected model requires static batch={in_shape[0]}."
            )
        profile["static_batch_size"] = 1
    else:
        profile["static_batch_size"] = None

    allowed_output_types = {
        "tensor(float)",
        "tensor(float16)",
        "tensor(double)",
    }
    if output_meta.type not in allowed_output_types:
        raise RuntimeError(
            "Expected a floating-point segmentation output tensor; got "
            f"{output_meta.type}."
        )

    out_shape = output_meta.shape
    if len(out_shape) not in (2, 3, 4):
        raise RuntimeError(
            "ArcGUI expects a single-channel segmentation output with rank "
            f"2, 3, or 4; got shape {out_shape}."
        )

    if len(out_shape) == 4:
        channel_first_ok = (
            not isinstance(out_shape[1], int)
            or out_shape[1] == 1
        )
        channel_last_ok = (
            not isinstance(out_shape[-1], int)
            or out_shape[-1] == 1
        )
        if not channel_first_ok and not channel_last_ok:
            raise RuntimeError(
                "ArcGUI supports a single-channel segmentation logit output. "
                f"Got 4-D output shape {out_shape}; a two-class/multi-channel "
                "output is not supported by this deployment wrapper."
            )

    if len(out_shape) == 2:
        _warn(
            "The ONNX model exposes a 2-D batchless output. ArcGUI will run "
            "single-image inference for compatibility."
        )
        profile["static_batch_size"] = 1

    return session, profile, effective_target

# =========================================================
# PREPROCESS / OUTPUT EXTRACTION
# =========================================================


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _prepare_input_rgb_imagenet_norm(image_rgb):
    resized = cv2.resize(
        image_rgb,
        (FIXED_MODEL_INPUT_SIZE, FIXED_MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    resized /= 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    resized = (resized - mean) / std
    return resized.transpose(2, 0, 1).astype(np.float32)



def _extract_probability_batch(raw_output, output_semantics="logits"):
    pred = np.asarray(raw_output)

    if pred.ndim == 4:
        if pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        elif pred.shape[-1] == 1:
            pred = pred[:, :, :, 0]
        else:
            raise RuntimeError(
                f"Unexpected multi-channel 4-D ONNX output shape: {pred.shape}"
            )
    elif pred.ndim == 3:
        pass
    elif pred.ndim == 2:
        pred = pred[None, ...]
    else:
        raise RuntimeError(
            f"Unexpected ONNX output shape: {pred.shape}"
        )

    pred = np.asarray(pred, dtype=np.float32)

    if not np.all(np.isfinite(pred)):
        raise RuntimeError(
            "ONNX output contains NaN or infinite values before probability "
            "conversion."
        )

    if output_semantics == "logits":
        probabilities = _sigmoid(pred).astype(np.float32)
    elif output_semantics == "probabilities":
        probabilities = pred.astype(np.float32)
        if np.min(probabilities) < -1e-5 or np.max(probabilities) > 1.00001:
            raise RuntimeError(
                "Model output was declared as probabilities but contains values "
                "outside [0, 1]."
            )
        probabilities = np.clip(probabilities, 0.0, 1.0)
    else:
        raise RuntimeError(
            f"Unsupported ONNX output semantics: {output_semantics}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            "ONNX output contains NaN or infinite probability values."
        )

    return probabilities


def _run_batch_probabilities(
    session,
    input_name,
    output_name,
    prepared_tensors,
    output_semantics="logits",
):
    """Run a prepared list of CHW tensors; fall back to one image at a time."""
    if not prepared_tensors:
        raise RuntimeError("Inference was requested with an empty tensor batch.")

    try:
        batch_input = np.stack(prepared_tensors, axis=0).astype(np.float32)
        raw = session.run([output_name], {input_name: batch_input})[0]
        probabilities = _extract_probability_batch(
            raw,
            output_semantics=output_semantics,
        )

        if probabilities.shape[0] != len(prepared_tensors):
            raise RuntimeError(
                f"Batch output count mismatch: {probabilities.shape[0]} vs "
                f"{len(prepared_tensors)}"
            )

        return probabilities

    except Exception as exc:
        if len(prepared_tensors) == 1:
            raise RuntimeError(
                f"Single-image ONNX inference failed: {exc}"
            ) from exc

        _warn(
            f"Batch inference failed ({exc}); falling back to single-image "
            "inference for this batch."
        )

        outputs = []
        for index, tensor in enumerate(prepared_tensors, start=1):
            try:
                raw = session.run(
                    [output_name],
                    {input_name: tensor[None, ...].astype(np.float32)},
                )[0]
                outputs.append(
                    _extract_probability_batch(
                        raw,
                        output_semantics=output_semantics,
                    )[0]
                )
            except Exception as single_exc:
                raise RuntimeError(
                    f"Single-image ONNX fallback failed for item {index}/"
                    f"{len(prepared_tensors)}: {single_exc}"
                ) from single_exc

        return np.stack(outputs, axis=0)

# =========================================================
# MASK / GEOMETRY PREPARATION
# =========================================================



def _geometry_is_usable(geom):
    """ArcGIS-version-safe geometry validity check."""
    if geom is None:
        return False

    # Some ArcPy Polygon builds do not expose .isEmpty.
    try:
        is_empty = geom.isEmpty
        if callable(is_empty):
            is_empty = is_empty()
        if is_empty:
            return False
    except Exception:
        pass

    try:
        if int(geom.pointCount) <= 0:
            return False
    except Exception:
        pass

    try:
        area = float(geom.area)
        if math.isfinite(area) and area <= 0:
            return False
    except Exception:
        pass

    return True



def _read_geometries(feature_class, label="requested extent"):
    geometries = []

    try:
        with arcpy.da.SearchCursor(feature_class, ["SHAPE@"]) as cursor:
            for (geom,) in cursor:
                if _geometry_is_usable(geom):
                    geometries.append(geom)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {label} geometries for spatial planning: {exc}"
        )

    if not geometries:
        raise RuntimeError(
            f"No usable polygon geometries were found for {label}."
        )

    return geometries

def _extent_is_finite(extent):
    if extent is None:
        return False

    try:
        values = [
            float(extent.XMin),
            float(extent.YMin),
            float(extent.XMax),
            float(extent.YMax),
        ]
    except Exception:
        return False

    return (
        all(math.isfinite(v) for v in values)
        and values[2] > values[0]
        and values[3] > values[1]
    )


def _feature_extent_safe(feature_class, label="feature class"):
    """
    Return a reliable extent for a polygon feature class.

    First use Describe(...).extent when it is finite. If ArcGIS returns NaN
    metadata (seen occasionally for newly projected in-memory irregular
    polygons), calculate the bounding rectangle directly from the actual
    SHAPE@ geometries.

    This does not modify the polygon and does not approximate the geometry
    itself; it only reconstructs its outer bounding rectangle from valid
    geometry extents.
    """
    try:
        described_extent = arcpy.Describe(feature_class).extent
        if _extent_is_finite(described_extent):
            return described_extent
    except Exception:
        described_extent = None

    _warn(
        f"ArcGIS reported invalid/NaN extent metadata for {label}. "
        "Rebuilding the planning extent from the actual polygon geometries."
    )

    xmin = None
    ymin = None
    xmax = None
    ymax = None
    valid_geometry_count = 0

    with arcpy.da.SearchCursor(feature_class, ["SHAPE@"]) as cursor:
        for (geom,) in cursor:
            if not _geometry_is_usable(geom):
                continue

            try:
                e = geom.extent
                vals = [
                    float(e.XMin),
                    float(e.YMin),
                    float(e.XMax),
                    float(e.YMax),
                ]
            except Exception:
                continue

            if not all(math.isfinite(v) for v in vals):
                continue

            if vals[2] <= vals[0] or vals[3] <= vals[1]:
                continue

            valid_geometry_count += 1

            xmin = vals[0] if xmin is None else min(xmin, vals[0])
            ymin = vals[1] if ymin is None else min(ymin, vals[1])
            xmax = vals[2] if xmax is None else max(xmax, vals[2])
            ymax = vals[3] if ymax is None else max(ymax, vals[3])

    if valid_geometry_count == 0:
        raise RuntimeError(
            f"Could not determine a finite extent for {label}. "
            "The projected polygon geometries did not contain a usable extent."
        )

    rebuilt = arcpy.Extent(xmin, ymin, xmax, ymax)

    _log(
        f"Rebuilt {label} extent from {valid_geometry_count} polygon "
        f"geometry/geometries: "
        f"{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}"
    )

    return rebuilt

# =========================================================
# EXPORT PLANNING
# =========================================================


def _balanced_axis_partitions(total_pixels, maximum_pixels):
    """
    Split an axis into the minimum number of non-overlapping, near-uniform
    integer pixel segments, each <= maximum_pixels.

    Examples:
      12000 -> [6000, 6000]
      20000 -> [10000, 10000]
      25000 -> [8334, 8333, 8333]

    Segment sizes differ by at most one pixel.
    """
    total_pixels = max(1, int(math.ceil(float(total_pixels))))
    maximum_pixels = max(1, int(maximum_pixels))

    count = max(1, int(math.ceil(total_pixels / float(maximum_pixels))))
    base, remainder = divmod(total_pixels, count)

    sizes = [
        base + (1 if i < remainder else 0)
        for i in range(count)
    ]

    starts = []
    cursor = 0
    for size in sizes:
        starts.append(cursor)
        cursor += size

    return list(zip(starts, sizes))



def _plan_large_export_blocks(
    export_extent,
    spatial_ref,
    mask_geometries,
    include_nonintersecting=False,
    maximum_pixels=MAX_LARGE_EXPORT_PIXELS,
):
    """
    Plan a balanced, NON-OVERLAPPING rectangular grid covering the complete
    working bounding rectangle.

    Important behavior:
      * <=10240 px on both axes -> exactly one candidate large export.
      * Larger dimensions are split uniformly per axis.
      * Candidate large blocks never overlap and never leave internal gaps.
      * If extent geometries are available, a candidate block is tested against
        the requested geometry BEFORE any World Imagery export occurs.
      * A candidate block with zero intersection is omitted from production
        export unless include_nonintersecting=True (used by debug preview).
      * Missing/unreadable target geometry is a hard error; ArcGUI never falls
        back to exporting the full bounding rectangle silently.
    """
    width_px, height_px = _extent_pixel_size(export_extent)

    maximum_pixels = max(
        MIN_LARGE_EXPORT_PIXELS,
        min(int(maximum_pixels), MAX_LARGE_EXPORT_PIXELS),
    )

    x_parts = _balanced_axis_partitions(
        width_px,
        maximum_pixels,
    )
    y_parts = _balanced_axis_partitions(
        height_px,
        maximum_pixels,
    )

    blocks = []
    block_id = 0

    for row, (y0, block_h) in enumerate(y_parts):
        y_top = export_extent.YMax - y0 * FIXED_CELL_SIZE
        y_bottom = y_top - block_h * FIXED_CELL_SIZE

        for col, (x0, block_w) in enumerate(x_parts):
            x_left = export_extent.XMin + x0 * FIXED_CELL_SIZE
            x_right = x_left + block_w * FIXED_CELL_SIZE

            block_id += 1

            polygon = _rect_polygon(
                x_left,
                y_bottom,
                x_right,
                y_top,
                spatial_ref,
            )

            intersects_requested_extent = _geometry_intersects_any(
                polygon,
                mask_geometries,
            )

            block = {
                "block_id": block_id,
                "row": row,
                "col": col,
                "grid_rows": len(y_parts),
                "grid_cols": len(x_parts),
                "x_start_px": x0,
                "y_start_px": y0,
                "width_px": block_w,
                "height_px": block_h,
                "extent": arcpy.Extent(
                    x_left,
                    y_bottom,
                    x_right,
                    y_top,
                ),
                "polygon": polygon,
                "intersects_requested_extent": bool(
                    intersects_requested_extent
                ),
            }

            if (
                intersects_requested_extent
                or include_nonintersecting
            ):
                blocks.append(block)

    return blocks

def _plan_crop_footprints_for_extent(extent, width_px, height_px, spatial_ref, stride_pixels):
    x_starts = _sliding_starts(width_px, FIXED_TILE_SIZE, stride_pixels)
    y_starts = _sliding_starts(height_px, FIXED_TILE_SIZE, stride_pixels)

    crops = []
    crop_id = 0
    tile_map = FIXED_TILE_SIZE * FIXED_CELL_SIZE

    for row, y0 in enumerate(y_starts):
        y_top = extent.YMax - y0 * FIXED_CELL_SIZE
        y_bottom = y_top - tile_map

        for col, x0 in enumerate(x_starts):
            x_left = extent.XMin + x0 * FIXED_CELL_SIZE
            x_right = x_left + tile_map

            crop_id += 1
            crop_extent = arcpy.Extent(
                x_left,
                y_bottom,
                x_right,
                y_top,
            )

            crops.append({
                "crop_id": crop_id,
                "row": row,
                "col": col,
                "x_start_px": x0,
                "y_start_px": y0,
                "extent": crop_extent,
                "polygon": _rect_polygon(
                    x_left,
                    y_bottom,
                    x_right,
                    y_top,
                    spatial_ref,
                ),
            })

    return crops


def _partition_model_crops_by_extent_intersection(
    crops,
    mask_geometries,
):
    """
    Split planned model-window footprints into:
      * active/intersecting windows; and
      * windows completely outside the requested polygon.

    A window is retained when it touches the requested geometry at all.
    Only windows with zero intersection are skipped.

    This is deliberately different from ExportTrainingDataForDeepLearning's
    ``in_mask_polygons`` behavior. Esri only creates chips that fall completely
    inside an input mask polygon, which would incorrectly discard useful
    boundary windows that only partially intersect the target.
    """
    active = []
    skipped = []

    for crop in crops:
        if _geometry_intersects_any(
            crop["polygon"],
            mask_geometries,
        ):
            active.append(crop)
        else:
            skipped.append(crop)

    return active, skipped


def _build_direct_tile_plan(
    export_extent,
    spatial_ref,
    mask_geometries,
    stride_pixels,
):
    """
    Build the authoritative direct/non-optimized 1024-tile plan.

    This helper is shared by:
      * debug preview; and
      * production export.

    Therefore the cyan debug footprints are the exact set of planned model
    windows that production is allowed to export/infer.

    A tile is active when it intersects the requested polygon at all.
    A tile with zero intersection is skipped before World Imagery export.
    """
    width_px, height_px = _extent_pixel_size(export_extent)

    candidate_crops = _plan_crop_footprints_for_extent(
        export_extent,
        width_px,
        height_px,
        spatial_ref,
        stride_pixels,
    )

    active_crops, skipped_crops = (
        _partition_model_crops_by_extent_intersection(
            candidate_crops,
            mask_geometries,
        )
    )

    if not active_crops:
        raise ArcGUIImageryContentError(
            "The direct 1024 planning grid contains no tile that intersects "
            "the requested extent."
        )

    return {
        "width_px": width_px,
        "height_px": height_px,
        "stride_pixels": int(stride_pixels),
        "candidate_crops": candidate_crops,
        "active_crops": active_crops,
        "skipped_crops": skipped_crops,
    }


def _direct_plan_fingerprint(direct_plan):
    """Stable hash of the exact retained direct-tile geographic plan."""
    if not direct_plan:
        return ""
    values = []
    for crop in direct_plan.get("active_crops") or []:
        e = crop["extent"]
        values.append(
            [
                int(crop.get("crop_id", 0)),
                round(float(e.XMin), 6),
                round(float(e.YMin), 6),
                round(float(e.XMax), 6),
                round(float(e.YMax), 6),
            ]
        )
    payload = {
        "stride_pixels": int(direct_plan.get("stride_pixels", 0)),
        "active": values,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


# =========================================================
# DEBUG FOOTPRINT PREVIEW
# =========================================================


def _create_polygon_feature_class(gdb_path, name, spatial_ref, fields, rows):
    fc = str(Path(gdb_path) / _safe_name(name))

    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)

    arcpy.management.CreateFeatureclass(
        out_path=str(gdb_path),
        out_name=Path(fc).name,
        geometry_type="POLYGON",
        spatial_reference=spatial_ref,
    )

    for field in fields:
        field_name, field_type = field[:2]
        kwargs = {}
        if len(field) >= 3 and field[2] is not None:
            kwargs["field_length"] = field[2]
        arcpy.management.AddField(fc, field_name, field_type, **kwargs)

    cursor_fields = ["SHAPE@"] + [field[0] for field in fields]

    with arcpy.da.InsertCursor(fc, cursor_fields) as cursor:
        for row in rows:
            cursor.insertRow(row)

    return fc


def _style_debug_polygon_layer(layer, outline_rgb, outline_width=1.5, fill_alpha=0):
    """Best-effort transparent-fill symbology so overlap remains visible."""
    try:
        sym = layer.symbology
        if hasattr(sym, "renderer") and hasattr(sym.renderer, "symbol"):
            symbol = sym.renderer.symbol
            symbol.color = {"RGB": [255, 255, 255, int(fill_alpha)]}
            if hasattr(symbol, "outlineColor"):
                symbol.outlineColor = {"RGB": [*outline_rgb, 100]}
            if hasattr(symbol, "outlineWidth"):
                symbol.outlineWidth = float(outline_width)
            layer.symbology = sym
    except Exception as exc:
        _warn(f"Could not apply debug outline symbology to {layer.name}: {exc}")


def _add_debug_layers_to_map(extent_name, layers_to_add):
    """
    Add a small number of transparent footprint layers.

    Individual crops are individual polygon FEATURES rather than hundreds of
    individual ArcGIS layers. This preserves overlap visibility without making
    the Contents pane unusably slow.
    """
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.activeMap
        if m is None:
            return

        group = None
        if hasattr(m, "createGroupLayer"):
            try:
                group = m.createGroupLayer(f"ArcGUI Debug - {extent_name}")
            except Exception:
                group = None

        for spec in layers_to_add:
            layer = m.addDataFromPath(spec["path"])
            try:
                layer.name = spec["name"]
            except Exception:
                pass

            try:
                layer.visible = bool(spec.get("visible", True))
            except Exception:
                pass

            _style_debug_polygon_layer(
                layer,
                spec["outline_rgb"],
                spec.get("outline_width", 1.5),
                spec.get("fill_alpha", 0),
            )

            if group is not None:
                try:
                    m.addLayerToGroup(group, layer, "BOTTOM")
                    m.removeLayer(layer)
                except Exception:
                    pass

    except Exception as exc:
        _warn(f"Could not add debug preview layers to the active map: {exc}")


def _prepare_debug_preview_context(
    extent_layer,
    output_root,
    overlap_percent,
    input_raster_name,
):
    """Prepare debug spatial context and clean temporary FCs if preparation fails."""
    temp_source_extent = None
    temp_projected_extent = None
    try:
        if not arcpy.Exists(extent_layer):
            raise ValueError(f"Extent layer not found: {extent_layer}")

        overlap_percent = float(overlap_percent)
        stride_pixels = overlap_percent_to_stride(overlap_percent)
        source_desc = _validate_polygon_extent_layer(extent_layer)
        extent_name = _layer_display_name(extent_layer)
        source_extent_fc, temp_source_extent = _materialize_extent_layer(
            extent_layer,
            extent_name,
        )

        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap
        input_raster = _resolve_input_imagery_layer(active_map, input_raster_name)
        try:
            active_map_sr = active_map.spatialReference
        except Exception as exc:
            raise RuntimeError(f"Could not read the active map coordinate system: {exc}") from exc
        if not _is_valid_spatial_reference(active_map_sr):
            raise RuntimeError(
                "The active ArcGIS Pro map has an Unknown/undefined coordinate system. "
                "Set the map to a projected metric CRS before running debug preview."
            )
        _validate_fixed_cell_export_crs(active_map_sr)
        spatial_ref = _get_export_spatial_reference(
            input_raster,
            active_map=active_map,
            fallback_spatial_ref=source_desc.spatialReference,
        )
        _validate_fixed_cell_export_crs(spatial_ref)

        extent_fc, temp_projected_extent = _project_feature_class_to_spatial_ref(
            source_extent_fc,
            spatial_ref,
            extent_name,
        )
        original_extent = _feature_extent_safe(
            extent_fc,
            label=f"projected debug extent '{extent_name}'",
        )
        export_extent = original_extent
        debug_width_px, debug_height_px = _extent_pixel_size(export_extent)
        mask_geometries = _read_geometries(extent_fc, label="requested extent")

        output_root = _ensure_writable_output_directory(output_root)
        debug_root = _next_available_path(output_root / "debug_preview")
        debug_root.mkdir(parents=True, exist_ok=True)
        gdb_path = debug_root / "debug_preview.gdb"
        if not arcpy.Exists(str(gdb_path)):
            arcpy.management.CreateFileGDB(str(debug_root), gdb_path.name)

        return {
            "overlap_percent": overlap_percent,
            "stride_pixels": stride_pixels,
            "source_desc": source_desc,
            "extent_name": extent_name,
            "temp_source_extent": temp_source_extent,
            "active_map": active_map,
            "input_raster": input_raster,
            "spatial_ref": spatial_ref,
            "extent_fc": extent_fc,
            "temp_projected_extent": temp_projected_extent,
            "original_extent": original_extent,
            "export_extent": export_extent,
            "debug_width_px": debug_width_px,
            "debug_height_px": debug_height_px,
            "mask_geometries": mask_geometries,
            "debug_root": debug_root,
            "gdb_path": gdb_path,
        }
    except Exception:
        for temp_fc in (temp_projected_extent, temp_source_extent):
            if temp_fc and arcpy.Exists(temp_fc):
                try:
                    arcpy.management.Delete(temp_fc)
                except Exception:
                    pass
        raise


def create_debug_preview(
    extent_layer,
    output_root,
    overlap_percent,
    optimized_large_crop=True,
    input_raster_name="World Imagery",
    add_to_map=True,
):
    """Create transparent footprint previews only; no imagery export or inference."""
    context = _prepare_debug_preview_context(
        extent_layer=extent_layer,
        output_root=output_root,
        overlap_percent=overlap_percent,
        input_raster_name=input_raster_name,
    )
    overlap_percent = context["overlap_percent"]
    stride_pixels = context["stride_pixels"]
    extent_name = context["extent_name"]
    temp_source_extent = context["temp_source_extent"]
    input_raster = context["input_raster"]
    spatial_ref = context["spatial_ref"]
    extent_fc = context["extent_fc"]
    temp_projected_extent = context["temp_projected_extent"]
    export_extent = context["export_extent"]
    debug_width_px = context["debug_width_px"]
    debug_height_px = context["debug_height_px"]
    mask_geometries = context["mask_geometries"]
    gdb_path = context["gdb_path"]

    _log(
        f"Debug working extent: "
        f"{export_extent.XMin:.3f}, {export_extent.YMin:.3f}, "
        f"{export_extent.XMax:.3f}, {export_extent.YMax:.3f}"
    )
    _log(
        f"Debug working size at {FIXED_CELL_SIZE:g} map-units/pixel: "
        f"{debug_width_px} x {debug_height_px} px"
    )
    _log(f"Debug/export planning CRS: {_spatial_reference_label(spatial_ref)}")

    layers = []

    try:
        if _bool_from_tool_value(optimized_large_crop):
            debug_large_limit = (
                _get_image_service_pixel_limit(input_raster)
                if input_raster is not None
                else None
            )
            if debug_large_limit is None:
                debug_large_limit = MAX_LARGE_EXPORT_PIXELS
            elif debug_large_limit < MIN_LARGE_EXPORT_PIXELS:
                raise RuntimeError(
                    "The image service reports a request limit smaller than "
                    f"the required {FIXED_TILE_SIZE}px model chip "
                    f"(reported limit={debug_large_limit}px)."
                )

            all_blocks = _plan_large_export_blocks(
                export_extent,
                spatial_ref,
                mask_geometries=mask_geometries,
                include_nonintersecting=True,
                maximum_pixels=debug_large_limit,
            )

            blocks = [
                block
                for block in all_blocks
                if block["intersects_requested_extent"]
            ]

            skipped_large_blocks = [
                block
                for block in all_blocks
                if not block["intersects_requested_extent"]
            ]

            estimated_debug_crops = 0
            for block in blocks:
                estimated_debug_crops += (
                    len(_sliding_starts(
                        block["width_px"],
                        FIXED_TILE_SIZE,
                        stride_pixels,
                    ))
                    *
                    len(_sliding_starts(
                        block["height_px"],
                        FIXED_TILE_SIZE,
                        stride_pixels,
                    ))
                )

            if estimated_debug_crops > MAX_DEBUG_CROP_FEATURES:
                raise RuntimeError(
                    "Debug preview would create "
                    f"{estimated_debug_crops:,} model-crop polygons, which is "
                    "too large for a safe interactive preview. Reduce the area "
                    "or reduce overlap before running Debug Footprint Preview."
                )

            block_rows = []
            skipped_block_rows = []
            active_crop_rows = []
            skipped_crop_rows = []
            crop_global_id = 0

            for block in skipped_large_blocks:
                skipped_block_rows.append([
                    block["polygon"],
                    block["block_id"],
                    block["row"],
                    block["col"],
                    block["width_px"],
                    block["height_px"],
                    block["grid_rows"],
                    block["grid_cols"],
                ])

            # Only retained/intersecting large blocks receive 1024 model crops.
            for block in blocks:
                block_rows.append([
                    block["polygon"],
                    block["block_id"],
                    block["row"],
                    block["col"],
                    block["width_px"],
                    block["height_px"],
                    block["grid_rows"],
                    block["grid_cols"],
                ])

                block_crops = _plan_crop_footprints_for_extent(
                    block["extent"],
                    block["width_px"],
                    block["height_px"],
                    spatial_ref,
                    stride_pixels,
                )

                for crop in block_crops:
                    crop_global_id += 1
                    row = [
                        crop["polygon"],
                        crop_global_id,
                        block["block_id"],
                        crop["row"],
                        crop["col"],
                        FIXED_TILE_SIZE,
                        stride_pixels,
                        overlap_percent,
                    ]

                    if _geometry_intersects_any(crop["polygon"], mask_geometries):
                        active_crop_rows.append(row)
                    else:
                        skipped_crop_rows.append(row)

            block_fc = _create_polygon_feature_class(
                gdb_path,
                f"{extent_name}_large_export_blocks",
                spatial_ref,
                [
                    ("LargeID", "LONG"),
                    ("GridRow", "LONG"),
                    ("GridCol", "LONG"),
                    ("WidthPx", "LONG"),
                    ("HeightPx", "LONG"),
                    ("GridRows", "LONG"),
                    ("GridCols", "LONG"),
                ],
                block_rows,
            )

            skipped_block_fc = None
            if skipped_block_rows:
                skipped_block_fc = _create_polygon_feature_class(
                    gdb_path,
                    f"{extent_name}_skipped_large_export_blocks",
                    spatial_ref,
                    [
                        ("LargeID", "LONG"),
                        ("GridRow", "LONG"),
                        ("GridCol", "LONG"),
                        ("WidthPx", "LONG"),
                        ("HeightPx", "LONG"),
                        ("GridRows", "LONG"),
                        ("GridCols", "LONG"),
                    ],
                    skipped_block_rows,
                )

            active_fc = _create_polygon_feature_class(
                gdb_path,
                f"{extent_name}_active_model_crops",
                spatial_ref,
                [
                    ("CropID", "LONG"),
                    ("LargeID", "LONG"),
                    ("GridRow", "LONG"),
                    ("GridCol", "LONG"),
                    ("TilePx", "LONG"),
                    ("StridePx", "LONG"),
                    ("OverlapPct", "DOUBLE"),
                ],
                active_crop_rows,
            )

            layers.append({
                "path": block_fc,
                "name": "01 - Large exports that WILL run",
                "outline_rgb": [255, 120, 0],
                "outline_width": 2.5,
            })

            if skipped_block_fc:
                layers.append({
                    "path": skipped_block_fc,
                    "name": "02 - Large exports SKIPPED (no extent intersection)",
                    "outline_rgb": [150, 150, 150],
                    "outline_width": 1.4,
                })

            layers.append({
                "path": active_fc,
                "name": "03 - Active 1024 model crops",
                "outline_rgb": [0, 220, 255],
                "outline_width": 1.2,
            })

            skipped_fc = None
            if skipped_crop_rows:
                skipped_fc = _create_polygon_feature_class(
                    gdb_path,
                    f"{extent_name}_outside_model_crops",
                    spatial_ref,
                    [
                        ("CropID", "LONG"),
                        ("LargeID", "LONG"),
                        ("GridRow", "LONG"),
                        ("GridCol", "LONG"),
                        ("TilePx", "LONG"),
                        ("StridePx", "LONG"),
                        ("OverlapPct", "DOUBLE"),
                    ],
                    skipped_crop_rows,
                )
                layers.append({
                    "path": skipped_fc,
                    "name": "04 - 1024 crops skipped outside requested extent",
                    "outline_rgb": [150, 150, 150],
                    "outline_width": 0.8,
                })

            _log("\n===================================")
            _log("DEBUG OPTIMIZED FOOTPRINT PREVIEW")
            _log("===================================")
            _log(f"Extent: {extent_name}")
            _log(f"Balanced large-block candidates: {len(all_blocks)}")
            _log(f"Large blocks that WILL be exported: {len(blocks)}")
            _log(
                f"Large blocks skipped before World Imagery export "
                f"(no requested-extent intersection): {len(skipped_large_blocks)}"
            )

            width_px, height_px = _extent_pixel_size(export_extent)
            grid_rows = (
                max(block["grid_rows"] for block in all_blocks)
                if all_blocks
                else 0
            )
            grid_cols = (
                max(block["grid_cols"] for block in all_blocks)
                if all_blocks
                else 0
            )
            widths = sorted({
                block["width_px"]
                for block in all_blocks
            })
            heights = sorted({
                block["height_px"]
                for block in all_blocks
            })

            _log(f"Working bounding rectangle: {width_px} x {height_px} px")
            _log(f"Large block maximum for this preview: {debug_large_limit} x {debug_large_limit} px")
            _log(f"Balanced large-block grid: {grid_rows} row(s) x {grid_cols} column(s)")
            _log(f"Large block widths: {widths}")
            _log(f"Large block heights: {heights}")
            _log("Large block overlap: 0 px")
            _log(f"Active/intersecting 1024 crops: {len(active_crop_rows)}")
            _log(f"1024 crops skipped outside requested extent: {len(skipped_crop_rows)}")
            _log(f"Model overlap: {overlap_percent:.2f}%")
            _log(f"Model stride: {stride_pixels} px")

            outputs = [block_fc, active_fc]

            if skipped_block_fc:
                outputs.append(skipped_block_fc)

            if skipped_fc:
                outputs.append(skipped_fc)

        else:
            direct_plan = _build_direct_tile_plan(
                export_extent=export_extent,
                spatial_ref=spatial_ref,
                mask_geometries=mask_geometries,
                stride_pixels=stride_pixels,
            )

            crops = direct_plan["candidate_crops"]
            active_crops = direct_plan["active_crops"]
            skipped_crops = direct_plan["skipped_crops"]

            if len(crops) > MAX_DEBUG_CROP_FEATURES:
                raise RuntimeError(
                    "Direct debug preview would create "
                    f"{len(crops):,} model-crop polygons, which is too "
                    "large for a safe interactive preview. Reduce the area or "
                    "reduce overlap."
                )

            active_rows = []
            skipped_rows = []

            for crop in active_crops:
                active_rows.append([
                    crop["polygon"],
                    crop["crop_id"],
                    crop["row"],
                    crop["col"],
                    FIXED_TILE_SIZE,
                    stride_pixels,
                    overlap_percent,
                ])

            for crop in skipped_crops:
                skipped_rows.append([
                    crop["polygon"],
                    crop["crop_id"],
                    crop["row"],
                    crop["col"],
                    FIXED_TILE_SIZE,
                    stride_pixels,
                    overlap_percent,
                ])

            active_fc = _create_polygon_feature_class(
                gdb_path,
                f"{extent_name}_direct_active_model_crops",
                spatial_ref,
                [
                    ("CropID", "LONG"),
                    ("GridRow", "LONG"),
                    ("GridCol", "LONG"),
                    ("TilePx", "LONG"),
                    ("StridePx", "LONG"),
                    ("OverlapPct", "DOUBLE"),
                ],
                active_rows,
            )

            layers.append({
                "path": active_fc,
                "name": "01 - Active direct 1024 tiles (WILL export/infer)",
                "outline_rgb": [0, 220, 255],
                "outline_width": 1.2,
                "visible": True,
            })

            skipped_fc = None
            if skipped_rows:
                skipped_fc = _create_polygon_feature_class(
                    gdb_path,
                    f"{extent_name}_direct_skipped_model_crops",
                    spatial_ref,
                    [
                        ("CropID", "LONG"),
                        ("GridRow", "LONG"),
                        ("GridCol", "LONG"),
                        ("TilePx", "LONG"),
                        ("StridePx", "LONG"),
                        ("OverlapPct", "DOUBLE"),
                    ],
                    skipped_rows,
                )
                layers.append({
                    "path": skipped_fc,
                    "name": "02 - Direct 1024 tiles SKIPPED outside extent",
                    "outline_rgb": [150, 150, 150],
                    "outline_width": 0.8,
                    # Keep available for inspection but OFF so the initial
                    # preview shows only work that production will actually do.
                    "visible": False,
                })

            _log("\\n===================================")
            _log("DEBUG DIRECT-EXPORT FOOTPRINT PREVIEW")
            _log("===================================")
            _log(f"Extent: {extent_name}")
            _log(f"Direct 1024 candidate tiles: {len(crops)}")
            _log(f"Direct tiles that WILL be exported/inferred: {len(active_crops)}")
            _log(
                f"Direct tiles skipped before imagery export "
                f"(zero extent intersection): {len(skipped_crops)}"
            )
            _log(f"Overlap: {overlap_percent:.2f}%")
            _log(f"Stride: {stride_pixels} px")
            if skipped_fc:
                _log(
                    "The skipped-tile debug layer was created but is OFF by "
                    "default. Turn it on only if you want to inspect rejected "
                    "candidate windows."
                )

            outputs = [active_fc]
            if skipped_fc:
                outputs.append(skipped_fc)

        if add_to_map:
            _add_debug_layers_to_map(extent_name, layers)

        _log("No imagery was exported and no ONNX model was loaded or run.")
        return outputs

    finally:
        if temp_projected_extent and arcpy.Exists(temp_projected_extent):
            try:
                arcpy.management.Delete(temp_projected_extent)
            except Exception:
                pass

        if temp_source_extent and arcpy.Exists(temp_source_extent):
            try:
                arcpy.management.Delete(temp_source_extent)
            except Exception:
                pass


def run_debug_preview_multiple(
    extent_layers,
    output_root,
    overlap_percent,
    optimized_large_crop=True,
    input_raster_name="World Imagery",
):
    if not extent_layers:
        raise ValueError("At least one extent layer must be selected.")
    if optimized_large_crop is None:
        raise ValueError("Optimized Large-Area Export option must be True or False.")
    optimized_large_crop = _bool_from_tool_value(optimized_large_crop)
    if output_root is None or not str(output_root).strip():
        raise ValueError("Output Folder is required.")
    if not str(input_raster_name or "").strip():
        raise ValueError("Input Raster Layer Name is required.")
    overlap_percent_to_stride(overlap_percent)

    normalized_extents = [str(x).strip().casefold() for x in extent_layers]
    if len(set(normalized_extents)) != len(normalized_extents):
        raise ValueError(
            "The same extent layer was supplied more than once. Remove duplicate "
            "extent entries before running Debug Footprint Preview."
        )

    outputs = []
    failures = []

    for extent_layer in extent_layers:
        try:
            result = create_debug_preview(
                extent_layer=extent_layer,
                output_root=output_root,
                overlap_percent=overlap_percent,
                optimized_large_crop=optimized_large_crop,
                input_raster_name=input_raster_name,
                add_to_map=True,
            )
            outputs.extend(result)
        except Exception as exc:
            _warn(f"DEBUG PREVIEW FAILED: {extent_layer} -> {exc}")
            failures.append((extent_layer, str(exc)))

    return outputs, failures

# Backward-compatible names from the immediately previous build.
def create_debug_fishnet(*args, **kwargs):
    return create_debug_preview(*args, **kwargs)


def run_debug_fishnet_multiple(*args, **kwargs):
    return run_debug_preview_multiple(*args, **kwargs)


def _positive_int_property(obj, names):
    for name in names:
        try:
            value = getattr(obj, name)
            if callable(value):
                value = value()
            value = int(value)
            if value > 0:
                return value
        except Exception:
            pass
    return None


def _get_image_service_pixel_limit(input_raster):
    """
    Best-effort discovery of image-service request limits.

    ArcGIS services may expose maxImageWidth/maxImageHeight on the layer
    description. If unavailable, ArcGUI starts at 10240 and relies on the
    smaller-block recovery path if a request-size error occurs.
    """
    try:
        desc = arcpy.Describe(input_raster)
    except Exception:
        return None

    max_w = _positive_int_property(
        desc,
        ("maxImageWidth", "maximumImageWidth", "maxWidth"),
    )
    max_h = _positive_int_property(
        desc,
        ("maxImageHeight", "maximumImageHeight", "maxHeight"),
    )

    values = [v for v in (max_w, max_h) if v is not None]
    if not values:
        return None

    limit = int(min(values))

    if limit < MIN_LARGE_EXPORT_PIXELS:
        _warn(
            "The image service reports a request limit smaller than the fixed "
            f"{FIXED_TILE_SIZE}x{FIXED_TILE_SIZE} model chip: "
            f"max width={max_w}, max height={max_h}."
        )
        return limit

    safe_limit = min(MAX_LARGE_EXPORT_PIXELS, limit)

    _log(
        "Detected image-service request limit: "
        f"max width={max_w}, max height={max_h}; "
        f"ArcGUI optimized block cap={safe_limit} px."
    )
    return safe_limit


def _looks_like_image_service_size_error(message):
    text = str(message).lower()
    tokens = (
        "maximagewidth",
        "maximageheight",
        "maximum image",
        "maximum request",
        "request size",
        "request width",
        "request height",
        "too large",
        "exceeds the maximum",
        "exceeded",
        "image width",
        "image height",
    )
    return any(token in text for token in tokens)


def _next_smaller_large_export_limit(current_limit):
    current_limit = int(current_limit)

    if current_limit <= MIN_LARGE_EXPORT_PIXELS:
        return None

    candidate = max(
        MIN_LARGE_EXPORT_PIXELS,
        int(math.floor(current_limit / 2.0)),
    )

    # Keep a useful integer and never return the same limit.
    if candidate >= current_limit:
        candidate = current_limit - 1

    return max(MIN_LARGE_EXPORT_PIXELS, candidate)


# =========================================================
# ARCGIS EXPORT HELPERS
# =========================================================



def _export_training_data(
    input_raster,
    out_folder,
    extent,
    spatial_ref,
    tile_w,
    tile_h,
    stride_x,
    stride_y,
):
    """
    Export imagery chips from the ArcGIS imagery layer.

    In optimized mode the caller supplies stride == tile size, so large ArcGIS
    blocks do not overlap. Direct mode keeps the overlap-derived 1024 stride.
    """
    out_folder = Path(out_folder)

    if out_folder.exists():
        try:
            shutil.rmtree(out_folder)
        except Exception as exc:
            raise ArcGUIExportError(
                f"Could not clear previous ArcGIS export folder "
                f"{out_folder}: {exc}"
            )

    out_folder.mkdir(parents=True, exist_ok=True)

    extent_values = [
        float(extent.XMin),
        float(extent.YMin),
        float(extent.XMax),
        float(extent.YMax),
    ]
    if (
        not all(math.isfinite(v) for v in extent_values)
        or extent_values[2] <= extent_values[0]
        or extent_values[3] <= extent_values[1]
    ):
        raise ArcGUIExportError(
            f"Invalid ArcGIS export extent: {extent_values}"
        )

    extent_str = " ".join(str(v) for v in extent_values)

    tile_w = int(tile_w)
    tile_h = int(tile_h)
    stride_x = int(stride_x)
    stride_y = int(stride_y)

    if tile_w <= 0 or tile_h <= 0 or stride_x <= 0 or stride_y <= 0:
        raise ValueError(
            "ArcGIS export tile size and stride must all be positive integers."
        )

    _log(
        "ArcGIS export via ExportTrainingDataForDeepLearning: "
        f"{tile_w} x {tile_h} px, stride "
        f"{stride_x} x {stride_y} px, metadata=Export_Tiles"
    )

    try:
        with arcpy.EnvManager(
            extent=extent_str,
            cellSize=FIXED_CELL_SIZE,
            snapRaster=None,
            outputCoordinateSystem=spatial_ref,
        ):
            ExportTrainingDataForDeepLearning(
                in_raster=input_raster,
                out_folder=str(out_folder),
                in_class_data=None,
                image_chip_format="TIFF",
                tile_size_x=tile_w,
                tile_size_y=tile_h,
                stride_x=stride_x,
                stride_y=stride_y,
                output_nofeature_tiles="ALL_TILES",
                metadata_format="Export_Tiles",
                start_index=0,
                class_value_field=None,
                in_mask_polygons=None,
                rotation_angle=0,
                reference_system="MAP_SPACE",
                processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
                blacken_around_feature="NO_BLACKEN",
                crop_mode="FIXED_SIZE",
            )
    except Exception as exc:
        message = str(exc)
        error_cls = (
            ArcGUIExportSizeError
            if _looks_like_image_service_size_error(message)
            else ArcGUIExportError
        )
        raise error_cls(
            "ExportTrainingDataForDeepLearning failed for a planned imagery "
            f"request of {tile_w} x {tile_h} px. Details: {message}"
        ) from exc

    try:
        tifs = _collect_export_tifs(out_folder)
    except Exception as exc:
        # A large request that silently produces no TIFF is commonly a service
        # request-limit symptom, so allow optimized mode to retry smaller.
        error_cls = (
            ArcGUIExportSizeError
            if max(tile_w, tile_h) > MIN_LARGE_EXPORT_PIXELS
            else ArcGUIExportError
        )
        raise error_cls(
            "ExportTrainingDataForDeepLearning completed but created no TIFF "
            f"for the {tile_w} x {tile_h} px request. Details: {exc}"
        ) from exc

    return tifs

def _collect_export_tifs(export_root):
    """Find ArcGIS-exported TIFFs across version-dependent folder layouts."""
    export_root = Path(export_root)
    images_dir = export_root / "images"

    candidates = []
    for root in (images_dir, export_root):
        if root.exists():
            candidates.extend(root.rglob("*.tif"))
            candidates.extend(root.rglob("*.tiff"))

    # Preserve order but remove duplicates.
    seen = set()
    tifs = []
    for path in sorted(candidates):
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            tifs.append(path)

    if not tifs:
        raise RuntimeError(
            f"No TIFF images were exported under {export_root}"
        )

    return tifs

# =========================================================
# PROBABILITY RASTER HELPERS
# =========================================================


def _save_probability_array(array, reference_raster_path, output_path, nodata_value=-9999.0):
    reference = arcpy.Raster(str(reference_raster_path))
    output_path = str(output_path)

    if arcpy.Exists(output_path):
        arcpy.management.Delete(output_path)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.size <= 0:
        raise RuntimeError(
            f"Probability output must be a non-empty 2-D array; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            "Probability output contains NaN or infinite values and will not be written."
        )

    expected_h = int(reference.height)
    expected_w = int(reference.width)
    if array.shape != (expected_h, expected_w):
        raise RuntimeError(
            "Probability array dimensions do not match the georeferenced reference raster: "
            f"array={array.shape}, raster={(expected_h, expected_w)}."
        )

    raster = arcpy.NumPyArrayToRaster(
        array,
        arcpy.Point(reference.extent.XMin, reference.extent.YMin),
        reference.meanCellWidth,
        reference.meanCellHeight,
        value_to_nodata=float(nodata_value),
    )
    raster.save(output_path)
    if not arcpy.Exists(output_path):
        raise RuntimeError(f"Probability raster save reported success but output is missing: {output_path}")
    arcpy.management.DefineProjection(output_path, reference.spatialReference)
    return output_path



def _mosaic_probability_rasters(
    probability_rasters,
    final_dir,
    output_name,
    mosaic_method="MEAN",
):
    if not probability_rasters:
        raise RuntimeError(
            "No probability rasters are available to mosaic."
        )

    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_dir / output_name

    if arcpy.Exists(str(output_path)):
        arcpy.management.Delete(str(output_path))

    first = arcpy.Raster(str(probability_rasters[0]))
    first_sr = first.spatialReference
    first_cell_w = abs(float(first.meanCellWidth))
    first_cell_h = abs(float(first.meanCellHeight))

    if (
        not _is_valid_spatial_reference(first_sr)
        or not math.isfinite(first_cell_w)
        or not math.isfinite(first_cell_h)
        or first_cell_w <= 0
        or first_cell_h <= 0
    ):
        raise ArcGUIGeoreferenceError(
            "The first probability raster has invalid georeference metadata."
        )

    for path in probability_rasters[1:]:
        raster = arcpy.Raster(str(path))

        if not _spatial_references_equivalent(
            raster.spatialReference,
            first_sr,
        ):
            raise ArcGUIGeoreferenceError(
                "Probability rasters do not share the same CRS before mosaic: "
                f"{path}"
            )

        cell_w = abs(float(raster.meanCellWidth))
        cell_h = abs(float(raster.meanCellHeight))

        if (
            not math.isfinite(cell_w)
            or not math.isfinite(cell_h)
            or abs(cell_w - first_cell_w)
                > first_cell_w * CELL_SIZE_RELATIVE_TOLERANCE
            or abs(cell_h - first_cell_h)
                > first_cell_h * CELL_SIZE_RELATIVE_TOLERANCE
        ):
            raise ArcGUIGeoreferenceError(
                "Probability rasters do not share a compatible cell size "
                f"before mosaic: {path}"
            )

    with arcpy.EnvManager(
        snapRaster=str(probability_rasters[0]),
        cellSize=first_cell_w,
        outputCoordinateSystem=first_sr,
    ):
        arcpy.management.MosaicToNewRaster(
            input_rasters=";".join(
                str(p) for p in probability_rasters
            ),
            output_location=str(final_dir),
            raster_dataset_name_with_extension=output_path.name,
            coordinate_system_for_the_raster=first_sr,
            pixel_type="32_BIT_FLOAT",
            cellsize=first_cell_w,
            number_of_bands=1,
            mosaic_method=str(mosaic_method).upper(),
            mosaic_colormap_mode="FIRST",
        )

    if not arcpy.Exists(str(output_path)):
        raise RuntimeError(
            f"Probability mosaic was not created: {output_path}"
        )

    return output_path

def _finalize_probability_to_clean_roads(
    probability_mosaic,
    extent_fc,
    original_extent,
    threshold,
    final_dir,
    output_name_prefix,
):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    clipped_probability = final_dir / "roads_probability_clipped_to_extent.tif"
    binary_raster = final_dir / "roads_prediction_binary.tif"
    clean_raster = final_dir / f"{output_name_prefix}_roads_prediction_clean.tif"

    for path in (clipped_probability, binary_raster, clean_raster):
        if arcpy.Exists(str(path)):
            arcpy.management.Delete(str(path))

    extent_str = (
        f"{original_extent.XMin} {original_extent.YMin} "
        f"{original_extent.XMax} {original_extent.YMax}"
    )

    arcpy.management.Clip(
        in_raster=str(probability_mosaic),
        rectangle=extent_str,
        out_raster=str(clipped_probability),
        in_template_dataset=extent_fc,
        nodata_value="-9999",
        clipping_geometry="ClippingGeometry",
        maintain_clipping_extent="NO_MAINTAIN_EXTENT",
    )

    # Preserve the probability grid alignment. NO_MAINTAIN_EXTENT avoids
    # resampling merely to force the exact rectangle boundary.
    prob = arcpy.Raster(str(clipped_probability))

    try:
        all_nodata = str(
            arcpy.management.GetRasterProperties(
                str(clipped_probability),
                "ALLNODATA",
            ).getOutput(0)
        ).strip().lower()
        if all_nodata in ("1", "true"):
            raise RuntimeError(
                "The clipped probability raster contains only NoData. "
                "This usually indicates non-overlapping geometry/georeferencing "
                "or missing imagery coverage."
            )
    except RuntimeError:
        raise
    except Exception:
        # Not every ArcGIS raster source/build exposes ALLNODATA reliably.
        pass

    Con(prob > float(threshold), 255, 0).save(str(binary_raster))

    try:
        max_binary = float(
            arcpy.management.GetRasterProperties(
                str(binary_raster),
                "MAXIMUM",
            ).getOutput(0)
        )
        if math.isfinite(max_binary) and max_binary <= 0:
            _warn(
                f"The selected threshold ({float(threshold):.3f}) produced "
                "no road pixels inside this extent. The output is valid but "
                "the clean road raster will contain only NoData."
            )
    except Exception:
        pass

    # --------------------------------------------------------------
    # Create the display-clean road raster.
    #
    # Older builds used SetNull with the same TIFF as both the condition
    # and the value raster. ArcGIS can throw generic ERROR 999999 on some
    # TIFFs at that cosmetic cleanup stage even though the binary result
    # itself is already valid.
    #
    # Preferred method:
    #   Copy the valid 0/255 binary TIFF and declare value 0 as NoData.
    # --------------------------------------------------------------

    clean_created = False

    try:
        arcpy.management.CopyRaster(
            in_raster=str(binary_raster),
            out_rasterdataset=str(clean_raster),
            nodata_value="0",
            pixel_type="8_BIT_UNSIGNED",
        )
        clean_created = True
        _log(
            "Clean road raster created from the valid binary raster "
            "using CopyRaster (0 -> NoData)."
        )

    except Exception as copy_exc:
        _warn(
            "CopyRaster could not convert binary background 0 to NoData "
            f"({copy_exc}). Trying a direct probability-to-clean Con operation."
        )

        try:
            # When the optional false raster is omitted, false cells are NoData.
            Con(
                prob > float(threshold),
                255,
            ).save(str(clean_raster))

            clean_created = True
            _log(
                "Clean road raster created directly from the clipped "
                "probability raster using Con."
            )

        except Exception as con_exc:
            _warn(
                "Could not create a NoData-background clean road raster "
                f"({con_exc}). The road segmentation itself is already valid "
                "because roads_prediction_binary.tif was created successfully. "
                "ArcGUI will preserve that binary result."
            )

            try:
                arcpy.management.CopyRaster(
                    in_raster=str(binary_raster),
                    out_rasterdataset=str(clean_raster),
                    pixel_type="8_BIT_UNSIGNED",
                )
                clean_created = True
                _warn(
                    "Fallback final raster created successfully. "
                    "Its stored values are 0=background and 255=road; "
                    "the background was not converted to NoData."
                )

            except Exception as fallback_exc:
                _warn(
                    "Could not create the optional clean-display copy "
                    f"({fallback_exc}). Using the already-valid "
                    "roads_prediction_binary.tif as the final road output. "
                    "Values remain 0=background and 255=road."
                )
                clean_raster = binary_raster
                clean_created = True

    if not clean_created:
        clean_raster = binary_raster

    try:
        arcpy.management.DefineProjection(
            str(clean_raster),
            arcpy.Raster(str(probability_mosaic)).spatialReference,
        )
        arcpy.management.CalculateStatistics(str(clean_raster))
        arcpy.management.BuildPyramids(str(clean_raster))
    except Exception as exc:
        _warn(
            "Final road raster was created, but ArcGIS could not complete "
            f"optional projection/statistics/pyramid maintenance: {exc}"
        )

    return clean_raster, clipped_probability, binary_raster


def _raster_footprint_polygon(raster):
    e = raster.extent
    return _rect_polygon(
        float(e.XMin),
        float(e.YMin),
        float(e.XMax),
        float(e.YMax),
        raster.spatialReference,
    )


def _direct_coverage_key(raster):
    """
    Key direct exports by actual geographic footprint, not filename.

    ArcGIS can occasionally emit duplicate black/non-black TIFFs for the same
    chip. Rounding below is far tighter than the 2 m pixel size.
    """
    e = raster.extent
    return (
        round(float(e.XMin), 4),
        round(float(e.YMin), 4),
        round(float(e.XMax), 4),
        round(float(e.YMax), 4),
        int(raster.width),
        int(raster.height),
    )


def _filter_direct_export_tifs(
    exported_tifs,
    spatial_ref,
    mask_geometries,
):
    """
    Filter direct-export TIFFs by georeference, target intersection and content.

    A chip completely outside the requested polygon is skipped even if
    it contains imagery. A black chip is tolerated only when it is outside the
    target or a usable non-black duplicate covers the same geographic footprint.
    """
    groups = {}
    invalid = []

    for tif in exported_tifs:
        try:
            raster = _validate_exported_tif_basic(tif, spatial_ref)
            key = _direct_coverage_key(raster)
            state = _tif_has_nonblack_pixels(tif)
            groups.setdefault(key, []).append(
                {
                    "path": Path(tif),
                    "raster": raster,
                    "state": state,
                }
            )
        except Exception as exc:
            invalid.append((Path(tif), str(exc)))

    if invalid:
        details = "; ".join(
            f"{path.name}: {message}"
            for path, message in invalid[:5]
        )
        raise ArcGUIGeoreferenceError(
            f"{len(invalid)} direct-export TIFF(s) have invalid metadata. "
            f"Examples: {details}"
        )

    selected = []
    ignored_black = 0
    ignored_unreadable = 0
    ignored_outside = 0

    for items in groups.values():
        representative = items[0]["raster"]
        footprint = _raster_footprint_polygon(representative)
        intersects = _geometry_intersects_any(
            footprint,
            mask_geometries,
        )

        if not intersects:
            ignored_outside += len(items)
            continue

        nonblack = [x for x in items if x["state"] is True]
        black = [x for x in items if x["state"] is False]
        unreadable = [x for x in items if x["state"] is None]

        if nonblack:
            nonblack.sort(key=lambda x: str(x["path"]))
            selected.append(nonblack[0]["path"])

            if len(nonblack) > 1:
                _warn(
                    "ArcGIS produced multiple non-black direct TIFFs with the "
                    f"same footprint; using {nonblack[0]['path'].name}."
                )

            ignored_black += len(black)
            ignored_unreadable += len(unreadable)
            continue

        names = ", ".join(x["path"].name for x in items)

        if black:
            raise ArcGUIImageryContentError(
                "Direct export produced only completely black imagery for "
                "a chip that intersects the requested extent. "
                f"Candidates: {names}"
            )

        raise ArcGUIImageryContentError(
            "Direct export produced no readable imagery for a chip that "
            "intersects the requested extent. "
            f"Candidates: {names}"
        )

    if not selected:
        raise ArcGUIImageryContentError(
            "No usable direct-export imagery TIFFs intersect the requested "
            "extent after filtering."
        )

    if ignored_black:
        _log(
            f"Ignored {ignored_black} completely black direct-export TIFF(s) "
            "because usable duplicate coverage existed."
        )

    if ignored_unreadable:
        _warn(
            f"Ignored {ignored_unreadable} unreadable direct-export TIFF(s) "
            "because usable duplicate coverage existed."
        )

    if ignored_outside:
        _log(
            f"Skipped {ignored_outside} direct-export TIFF(s) completely "
            "outside the requested extent."
        )

    return selected, ignored_black, ignored_outside


# =========================================================
# DIRECT 1024 EXPORT MODE
# =========================================================



def _run_direct_export_mode(
    input_raster,
    export_extent,
    spatial_ref,
    mask_geometries,
    session,
    model_profile,
    batch_size,
    overlap_percent,
    output_root,
    final_dir,
    direct_plan=None,
    cache_context=None,
):
    """
    Direct 1024 mode with polygon-aware pre-export filtering.

    The complete rectangular planning grid is generated first. Any 1024x1024
    window with zero intersection with the requested polygon is removed
    BEFORE World Imagery is requested.

    Boundary windows are retained even when only part of the tile intersects the
    polygon. We intentionally do not use ETDFDL ``in_mask_polygons`` because
    Esri defines that parameter as creating only chips that fall completely
    within the polygon; that would remove useful boundary context.

    For a perfectly rectangular extent where every candidate tile intersects,
    ArcGUI keeps the efficient single ExportTrainingDataForDeepLearning call.
    For an irregular extent with outside tiles, ArcGUI exports only the retained
    planned windows, one exact 1024 request per active footprint.
    """
    stride = overlap_percent_to_stride(overlap_percent)
    reuse_cached = bool(cache_context and cache_context.get("reuse"))

    if not reuse_cached:
        service_limit = _get_image_service_pixel_limit(input_raster)
        if service_limit is not None and service_limit < FIXED_TILE_SIZE:
            raise ArcGUIExportError(
                "The input image service reports a maximum request size smaller "
                f"than the required {FIXED_TILE_SIZE}px model chip "
                f"(reported limit={service_limit}px)."
            )

    if direct_plan is None:
        direct_plan = _build_direct_tile_plan(
            export_extent=export_extent,
            spatial_ref=spatial_ref,
            mask_geometries=mask_geometries,
            stride_pixels=stride,
        )

    if int(direct_plan.get("stride_pixels", stride)) != int(stride):
        raise RuntimeError(
            "Internal direct-plan stride mismatch. Refusing to export a plan "
            "that does not match the requested overlap."
        )

    candidate_crops = direct_plan["candidate_crops"]
    active_crops = direct_plan["active_crops"]
    skipped_crops = direct_plan["skipped_crops"]

    _log("\\n===================================")
    _log("STAGE 1 - DIRECT 1024 TILE EXPORT")
    _log("===================================")
    _log("Tile size: 1024 x 1024")
    _log(f"Stride: {stride} x {stride}")
    _log(f"Direct candidate tiles over bounding rectangle: {len(candidate_crops):,}")
    _log(f"Direct tiles retained for export/inference: {len(active_crops):,}")
    _log(
        f"Direct tiles skipped BEFORE World Imagery export "
        f"(zero extent intersection): {len(skipped_crops):,}"
    )

    if cache_context:
        export_root = Path(cache_context["imagery_root"]) / "direct_1024_tiles"
    else:
        export_root = _next_available_path(
            Path(output_root) / "export_1024_tiles"
        )
    export_root.mkdir(parents=True, exist_ok=True)

    exported_items = []
    black_duplicate_count = 0

    if reuse_cached:
        exported_items = _restore_direct_export_items(
            cache_context,
            spatial_ref=spatial_ref,
            direct_plan=direct_plan,
        )
        if len(exported_items) != len(active_crops):
            raise RuntimeError(
                "Cached direct TIFF count does not match the current retained "
                f"tile plan ({len(exported_items)} cached vs {len(active_crops)} planned)."
            )
        _warn(
            f"CACHE HIT: reusing {len(exported_items):,} previously exported "
            "direct imagery TIFF(s). No World Imagery export will run."
        )

    # Fast path: a rectangular/fully covered planning grid contains no
    # irrelevant outside tiles. Keep the efficient one-call ArcGIS export.
    elif not skipped_crops:
        _log(
            "All direct planning tiles intersect the requested extent; using "
            "one ExportTrainingDataForDeepLearning call."
        )

        _export_training_data(
            input_raster=input_raster,
            out_folder=export_root,
            extent=export_extent,
            spatial_ref=spatial_ref,
            tile_w=FIXED_TILE_SIZE,
            tile_h=FIXED_TILE_SIZE,
            stride_x=stride,
            stride_y=stride,
        )

        exported_tifs = _collect_export_tifs(export_root)

        filtered_tifs, ignored_black, ignored_outside = (
            _filter_direct_export_tifs(
                exported_tifs,
                spatial_ref,
                mask_geometries,
            )
        )

        black_duplicate_count += int(ignored_black)

        # ignored_outside should normally be zero on this fast path, but keep
        # the value in the skipped count if ArcGIS generated an extra edge chip.
        if ignored_outside:
            _warn(
                f"ArcGIS emitted {ignored_outside} unexpected direct edge "
                "TIFF(s) outside the requested extent; they were ignored."
            )

        for index, tif in enumerate(filtered_tifs, start=1):
            exported_items.append({
                "path": Path(tif),
                "uid": f"tile_{index:08d}",
            })

    else:
        _log(
            "Irregular extent detected: exporting only intersecting 1024 tiles. "
            "Completely outside tiles will not be downloaded."
        )

        if len(active_crops) > 5000:
            _warn(
                f"Direct polygon-aware mode requires {len(active_crops):,} "
                "individual 1024 ArcGIS imagery requests. This is correct but "
                "can be slow; optimized large-area mode is recommended for "
                "large regions."
            )

        for index, crop in enumerate(active_crops, start=1):
            tile_root = (
                export_root
                / f"tile_{crop['crop_id']:08d}"
            )

            _log(
                f"Exporting active direct tile {index}/{len(active_crops)} "
                f"(crop id {crop['crop_id']}, row {crop['row']}, "
                f"column {crop['col']})"
            )

            _export_training_data(
                input_raster=input_raster,
                out_folder=tile_root,
                extent=crop["extent"],
                spatial_ref=spatial_ref,
                tile_w=FIXED_TILE_SIZE,
                tile_h=FIXED_TILE_SIZE,
                # One exact 1024 request. Model overlap was already encoded
                # in the planned crop origins.
                stride_x=FIXED_TILE_SIZE,
                stride_y=FIXED_TILE_SIZE,
            )

            candidates = _collect_export_tifs(tile_root)

            # Count black duplicate candidates for reporting. A retained tile
            # whose every candidate is black is rejected by the selector below.
            for candidate in candidates:
                if _tif_has_nonblack_pixels(candidate) is False:
                    black_duplicate_count += 1

            selected = _select_large_export_tif(
                candidates,
                FIXED_TILE_SIZE,
                FIXED_TILE_SIZE,
            )

            _validate_exported_tif_georeference(
                selected,
                crop["extent"],
                spatial_ref,
            )

            exported_items.append({
                "path": Path(selected),
                "uid": f"tile_{crop['crop_id']:08d}",
            })

    if not exported_items:
        raise ArcGUIImageryContentError(
            "No usable direct imagery tile remained after polygon-aware export "
            "and imagery validation."
        )

    if len(exported_items) != len(active_crops):
        raise ArcGUIExportError(
            "Direct export did not produce exactly one usable georeferenced TIFF "
            "for every retained model tile. "
            f"Planned={len(active_crops)}, usable={len(exported_items)}. "
            "ArcGUI is stopping instead of producing a partial mosaic."
        )

    if cache_context and not reuse_cached:
        _complete_direct_export_cache(
            cache_context,
            exported_items,
            direct_plan=direct_plan,
            spatial_ref=spatial_ref,
            skipped_count=len(skipped_crops),
        )
        _log(f"Reusable imagery cache completed: {cache_context['text_path']}")

    _log(
        f"Usable direct imagery tiles entering ONNX inference: "
        f"{len(exported_items):,}"
    )

    probability_dir = _next_available_path(
        Path(output_root) / "probability_tiles"
    )
    probability_dir.mkdir(parents=True, exist_ok=True)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output_semantics = model_profile.get(
        "output_semantics",
        "logits",
    )

    probability_rasters = []
    failed = 0
    prepared_total = 0

    for batch_index, batch_items in enumerate(
        _chunks(exported_items, batch_size),
        start=1,
    ):
        prepared = []
        tensors = []

        for item in batch_items:
            tif = item["path"]
            uid = item["uid"]

            try:
                with Image.open(tif) as image_file:
                    image = np.array(image_file.convert("RGB"))

                if not np.any(image):
                    raise ArcGUIImageryContentError(
                        f"Retained direct TIFF is completely black: {tif.name}"
                    )

                prepared.append(
                    (uid, tif, image.shape[0], image.shape[1])
                )
                tensors.append(
                    _prepare_input_rgb_imagenet_norm(image)
                )
                prepared_total += 1

            except Exception as exc:
                _warn(f"Could not prepare {uid} ({tif.name}): {exc}")
                failed += 1

        if not tensors:
            continue

        probabilities = _run_batch_probabilities(
            session,
            input_name,
            output_name,
            tensors,
            output_semantics=output_semantics,
        )

        for (uid, tif, h, w), probability in zip(
            prepared,
            probabilities,
        ):
            try:
                if probability.shape != (h, w):
                    probability = cv2.resize(
                        probability,
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                if not np.all(np.isfinite(probability)):
                    raise RuntimeError(
                        "Probability raster contains NaN/Inf values."
                    )

                out_path = (
                    probability_dir
                    / f"{uid}_probability.tif"
                )

                _save_probability_array(
                    probability,
                    tif,
                    out_path,
                )
                probability_rasters.append(out_path)

            except Exception as exc:
                _warn(
                    f"Could not save probability for {uid}: {exc}"
                )
                failed += 1

        _log(
            f"Direct inference batch {batch_index}: "
            f"{len(probability_rasters)} probability tile(s) saved; "
            f"{failed} failure(s)."
        )

    _log(f"Direct tiles prepared for inference: {prepared_total}")
    _log(
        f"Direct tiles skipped outside extent BEFORE export: "
        f"{len(skipped_crops)}"
    )
    _log(
        f"Black duplicate/edge TIFF candidates ignored: "
        f"{black_duplicate_count}"
    )
    _log(f"Failed direct tiles: {failed}")

    if failed > 0:
        raise RuntimeError(
            f"{failed} direct-export tile(s) could not be prepared, inferred, "
            "or saved. ArcGUI is stopping instead of creating a partial road "
            "mosaic."
        )

    if not probability_rasters:
        raise ArcGUIImageryContentError(
            "No direct probability tiles were produced after extent filtering "
            "and inference."
        )

    probability_mosaic = _mosaic_probability_rasters(
        probability_rasters,
        final_dir,
        "roads_probability_mosaic.tif",
    )

    return (
        probability_mosaic,
        len(exported_items),
        len(skipped_crops) + black_duplicate_count,
    )

# =========================================================
# OPTIMIZED LARGE EXPORT + PYTHON CROP MODE
# =========================================================





def _tif_has_nonblack_pixels(tif_path):
    """
    Inspect an exported RGB TIFF.

    Returns:
      True  -> readable and contains at least one non-zero RGB pixel
      False -> readable but completely black
      None  -> TIFF could not be decoded/read
    """
    try:
        with Image.open(tif_path) as im:
            rgb = im.convert("RGB")
            return rgb.getbbox() is not None
    except Exception as exc:
        _warn(
            f"Could not inspect TIFF content for {Path(tif_path).name}: {exc}"
        )
        return None


def _select_large_export_tif(tifs, expected_w, expected_h):
    """
    Select the actual imagery TIFF for one planned optimized block.

    Black duplicate TIFFs are ignored. Unreadable TIFFs are never selected.
    If ArcGIS returns only partial non-black TIFFs instead of one raster close
    to the planned block dimensions, ArcGUI treats that as a request-size
    problem so optimized mode can retry with smaller blocks.
    """
    if not tifs:
        raise ArcGUIExportError(
            "No TIFF candidates were supplied for the large block."
        )

    inspected = []

    for tif in tifs:
        tif = Path(tif)

        try:
            raster = arcpy.Raster(str(tif))
            w = int(raster.width)
            h = int(raster.height)
            dimension_score = (
                abs(w - int(expected_w))
                + abs(h - int(expected_h))
            )
            area = w * h
        except Exception as exc:
            _warn(
                f"Could not read raster dimensions for {tif.name}: {exc}"
            )
            w = -1
            h = -1
            dimension_score = 10**18
            area = 0

        content_state = _tif_has_nonblack_pixels(tif)

        inspected.append({
            "path": tif,
            "width": w,
            "height": h,
            "dimension_score": dimension_score,
            "area": area,
            "content_state": content_state,
        })

    nonblack_candidates = [
        item for item in inspected
        if item["content_state"] is True
    ]
    black_candidates = [
        item for item in inspected
        if item["content_state"] is False
    ]
    unreadable_candidates = [
        item for item in inspected
        if item["content_state"] is None
    ]

    for item in black_candidates:
        _log(
            f"Ignoring completely black ArcGIS export TIFF: "
            f"{item['path'].name} "
            f"({item['width']} x {item['height']} px)"
        )

    for item in unreadable_candidates:
        _warn(
            f"Ignoring unreadable ArcGIS export TIFF: "
            f"{item['path'].name}"
        )

    if not nonblack_candidates:
        if black_candidates and not unreadable_candidates:
            names = ", ".join(
                item["path"].name for item in black_candidates
            )
            raise ArcGUIImageryContentError(
                "ArcGIS exported TIFF files for a retained block, but every "
                f"candidate is completely black: {names}. "
                "The block intersects the requested extent, so ArcGUI "
                "will not create a prediction with missing imagery coverage."
            )

        names = ", ".join(
            item["path"].name for item in inspected
        )
        raise ArcGUIImageryContentError(
            "ArcGIS did not produce a readable non-black imagery TIFF for the "
            f"retained large block. Candidates: {names}"
        )

    nonblack_candidates.sort(
        key=lambda item: (
            item["dimension_score"],
            -item["area"],
            str(item["path"]),
        )
    )
    selected = nonblack_candidates[0]

    width_tol = max(
        8,
        int(math.ceil(
            int(expected_w) * EXPORT_DIMENSION_TOLERANCE_FRACTION
        )),
    )
    height_tol = max(
        8,
        int(math.ceil(
            int(expected_h) * EXPORT_DIMENSION_TOLERANCE_FRACTION
        )),
    )

    if (
        selected["width"] < 1
        or selected["height"] < 1
        or abs(selected["width"] - int(expected_w)) > width_tol
        or abs(selected["height"] - int(expected_h)) > height_tol
    ):
        raise ArcGUIExportSizeError(
            "ArcGIS returned non-black TIFF imagery, but no candidate covers "
            "the planned block dimensions closely enough. This can occur when "
            "an image service splits/rejects an oversized client request. "
            f"Planned={expected_w} x {expected_h} px; best candidate="
            f"{selected['width']} x {selected['height']} px."
        )

    if len(inspected) > 1:
        _log(
            f"Selected usable large export TIFF: {selected['path'].name} "
            f"({selected['width']} x {selected['height']} px; "
            f"planned {int(expected_w)} x {int(expected_h)} px)"
        )

    return selected["path"]

def _export_large_blocks(
    input_raster,
    blocks,
    spatial_ref,
    export_root,
):
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    exported = []

    for index, block in enumerate(blocks, start=1):
        block_root = export_root / f"block_{block['block_id']:04d}"

        _log(
            f"Exporting large block {index}/{len(blocks)}: "
            f"{block['width_px']} x {block['height_px']} px "
            f"(grid row {block['row'] + 1}/{block['grid_rows']}, "
            f"column {block['col'] + 1}/{block['grid_cols']})"
        )

        _export_training_data(
            input_raster=input_raster,
            out_folder=block_root,
            extent=block["extent"],
            spatial_ref=spatial_ref,
            tile_w=block["width_px"],
            tile_h=block["height_px"],
            stride_x=block["width_px"],
            stride_y=block["height_px"],
        )

        try:
            tifs = _collect_export_tifs(block_root)
        except Exception as exc:
            # _export_training_data already verifies that a TIFF was created,
            # so reaching this branch indicates files disappeared/became
            # unreadable between export and selection.
            raise ArcGUIExportError(
                f"Large block {block['block_id']} produced no readable TIFF "
                f"after ExportTrainingDataForDeepLearning: {exc}"
            )

        selected = _select_large_export_tif(
            tifs,
            block["width_px"],
            block["height_px"],
        )

        if len(tifs) > 1:
            _warn(
                f"ArcGIS created {len(tifs)} TIFFs for large block "
                f"{block['block_id']}. Completely black candidates were ignored; "
                f"using {selected.name} as the usable imagery export."
            )

        # Do not continue with a TIFF that has the wrong CRS tag or map extent.
        # This catches the previous failure mode where UTM-like coordinate values
        # were written into a TIFF tagged as Web Mercator.
        _validate_exported_tif_georeference(
            selected,
            block["extent"],
            spatial_ref,
            expected_width_px=block["width_px"],
            expected_height_px=block["height_px"],
        )

        exported.append({
            "block_id": block["block_id"],
            "path": selected,
            "planned_width_px": block["width_px"],
            "planned_height_px": block["height_px"],
        })

    return exported


def _infer_one_large_block(
    large_tif,
    block_id,
    session,
    model_profile,
    batch_size,
    overlap_percent,
    probability_dir,
    mask_geometries=None,
):
    """
    Create overlapping 1024 model windows from ONE ArcGIS-exported large TIFF
    and average their probabilities.

    The large TIFF itself has already been cropped from World Imagery by
    ExportTrainingDataForDeepLearning. Python only creates the 1024x1024 model
    windows inside that exported TIFF. Windows that do not intersect the
    requested polygon at all are skipped before ONNX inference.
    """
    with Image.open(large_tif) as image_file:
        image = np.array(image_file.convert("RGB"))

    if not np.any(image):
        return None, {
            "large_block_id": block_id,
            "model_crops_total": 0,
            "model_crops_intersecting": 0,
            "model_crops_inferred": 0,
            "model_crops_skipped_outside_extent": 0,
            "model_crops_skipped_black": 0,
            "large_block_skipped_black": 1,
        }

    h, w = image.shape[:2]
    stride = overlap_percent_to_stride(overlap_percent)
    x_starts = _sliding_starts(w, FIXED_TILE_SIZE, stride)
    y_starts = _sliding_starts(h, FIXED_TILE_SIZE, stride)

    raster_ref = arcpy.Raster(str(large_tif))
    raster_extent = raster_ref.extent
    cell_w = float(raster_ref.meanCellWidth)
    cell_h = float(raster_ref.meanCellHeight)
    spatial_ref = raster_ref.spatialReference

    all_crop_items = []
    crop_items = []
    skipped_outside = 0

    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            item = (row, col, y, x)
            all_crop_items.append(item)

            crop_w = min(FIXED_TILE_SIZE, w - x)
            crop_h = min(FIXED_TILE_SIZE, h - y)

            x_left = raster_extent.XMin + x * cell_w
            x_right = x_left + crop_w * cell_w
            y_top = raster_extent.YMax - y * cell_h
            y_bottom = y_top - crop_h * cell_h

            crop_polygon = _rect_polygon(
                x_left,
                y_bottom,
                x_right,
                y_top,
                spatial_ref,
            )

            if _geometry_intersects_any(crop_polygon, mask_geometries):
                crop_items.append(item)
            else:
                skipped_outside += 1

    probability_sum = np.zeros((h, w), dtype=np.float32)
    coverage_count = np.zeros((h, w), dtype=np.uint16)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output_semantics = model_profile.get("output_semantics", "logits")

    inferred = 0
    skipped_black = 0

    _log(
        f"Large block {block_id}: {len(all_crop_items)} total 1024 windows, "
        f"{len(crop_items)} intersect requested extent, "
        f"{skipped_outside} skipped outside extent"
    )

    for batch_index, batch in enumerate(_chunks(crop_items, batch_size), start=1):
        prepared_meta = []
        tensors = []

        for row, col, y, x in batch:
            crop = image[y:y + FIXED_TILE_SIZE, x:x + FIXED_TILE_SIZE]
            crop_h, crop_w = crop.shape[:2]

            if crop_h != FIXED_TILE_SIZE or crop_w != FIXED_TILE_SIZE:
                padded = np.zeros(
                    (FIXED_TILE_SIZE, FIXED_TILE_SIZE, 3),
                    dtype=np.uint8,
                )
                padded[:crop_h, :crop_w] = crop
                crop = padded

            # Still skip a truly empty exported crop if the imagery service
            # returned NoData/black pixels for the whole model window.
            if not np.any(crop):
                skipped_black += 1
                continue

            prepared_meta.append((y, x, crop_h, crop_w))
            tensors.append(_prepare_input_rgb_imagenet_norm(crop))

        if not tensors:
            continue

        probabilities = _run_batch_probabilities(
            session,
            input_name,
            output_name,
            tensors,
            output_semantics=output_semantics,
        )

        for (y, x, crop_h, crop_w), probability in zip(prepared_meta, probabilities):
            if probability.shape != (FIXED_TILE_SIZE, FIXED_TILE_SIZE):
                probability = cv2.resize(
                    probability,
                    (FIXED_TILE_SIZE, FIXED_TILE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

            probability = probability[:crop_h, :crop_w]

            probability_sum[
                y:y + crop_h,
                x:x + crop_w,
            ] += probability

            coverage_count[
                y:y + crop_h,
                x:x + crop_w,
            ] += 1

            inferred += 1

        _log(
            f"Large block {block_id}: crop batch {batch_index}; "
            f"{inferred}/{len(crop_items)} intersecting crops inferred, "
            f"{skipped_black} black crop(s) skipped."
        )

    if skipped_black > 0:
        raise ArcGUIImageryContentError(
            f"Large block {block_id} contains {skipped_black} completely black "
            "model crop(s) that intersect the requested polygon. ArcGUI is "
            "stopping instead of producing a prediction with missing target coverage."
        )

    valid = coverage_count > 0

    if not np.any(valid):
        return None, {
            "large_block_id": block_id,
            "model_crops_total": len(all_crop_items),
            "model_crops_intersecting": len(crop_items),
            "model_crops_inferred": inferred,
            "model_crops_skipped_outside_extent": skipped_outside,
            "model_crops_skipped_black": skipped_black,
            "large_block_skipped_black": 1,
        }

    probability_mean = np.full((h, w), -9999.0, dtype=np.float32)
    np.divide(
        probability_sum,
        coverage_count,
        out=probability_mean,
        where=valid,
    )

    probability_dir = Path(probability_dir)
    probability_dir.mkdir(parents=True, exist_ok=True)
    output_path = probability_dir / f"large_block_{block_id:04d}_probability.tif"

    _save_probability_array(
        probability_mean,
        large_tif,
        output_path,
        nodata_value=-9999.0,
    )

    return output_path, {
        "large_block_id": block_id,
        "model_crops_total": len(all_crop_items),
        "model_crops_intersecting": len(crop_items),
        "model_crops_inferred": inferred,
        "model_crops_skipped_outside_extent": skipped_outside,
        "model_crops_skipped_black": skipped_black,
        "large_block_skipped_black": 0,
    }


def _run_optimized_large_crop_mode(
    input_raster,
    export_extent,
    spatial_ref,
    mask_geometries,
    session,
    model_profile,
    batch_size,
    overlap_percent,
    output_root,
    final_dir,
    cache_context=None,
):
    width_px, height_px = _extent_pixel_size(export_extent)

    if width_px < FIXED_TILE_SIZE or height_px < FIXED_TILE_SIZE:
        _warn(
            "The optimized working extent is smaller than the 1024x1024 model "
            "input on at least one axis. ArcGUI will pad affected model "
            "windows with black pixels. For better model context, use a larger "
            "processing extent when practical."
        )

    reuse_cached = bool(cache_context and cache_context.get("reuse"))

    if reuse_cached:
        large_exports = _restore_optimized_large_exports(
            cache_context,
            spatial_ref,
        )
        cached_skipped = int(
            (cache_context.get("manifest") or {}).get(
                "skipped_large_block_count", 0
            )
        )
        skipped_large_blocks = [None] * cached_skipped
        current_limit = int(
            (cache_context.get("manifest") or {}).get(
                "actual_large_export_limit", MAX_LARGE_EXPORT_PIXELS
            )
        )
        _log("\n===================================")
        _log("STAGE 1 - OPTIMIZED LARGE-AREA EXPORT")
        _log("===================================")
        _warn(
            f"CACHE HIT: reusing {len(large_exports)} previously exported "
            "large imagery TIFF(s). No World Imagery export will run."
        )

    else:
        service_limit = _get_image_service_pixel_limit(input_raster)
        if (
            service_limit is not None
            and service_limit < MIN_LARGE_EXPORT_PIXELS
        ):
            raise ArcGUIExportError(
                "The input image service reports a maximum request size smaller "
                f"than the fixed {FIXED_TILE_SIZE}px model chip "
                f"(reported limit={service_limit}px)."
            )

        current_limit = (
            min(MAX_LARGE_EXPORT_PIXELS, service_limit)
            if service_limit is not None
            else MAX_LARGE_EXPORT_PIXELS
        )
        current_limit = max(
            MIN_LARGE_EXPORT_PIXELS,
            int(current_limit),
        )

        size_retries = 0
        generic_recovery_used = False

        while True:
            all_candidate_blocks = _plan_large_export_blocks(
                export_extent,
                spatial_ref,
                mask_geometries=mask_geometries,
                include_nonintersecting=True,
                maximum_pixels=current_limit,
            )

            blocks = [
                block
                for block in all_candidate_blocks
                if block["intersects_requested_extent"]
            ]

            skipped_large_blocks = [
                block
                for block in all_candidate_blocks
                if not block["intersects_requested_extent"]
            ]

            if not blocks:
                raise RuntimeError(
                    "No optimized large export block intersects the requested extent."
                )

            grid_rows = max(
                block["grid_rows"] for block in all_candidate_blocks
            )
            grid_cols = max(
                block["grid_cols"] for block in all_candidate_blocks
            )
            widths = sorted({
                block["width_px"] for block in blocks
            })
            heights = sorted({
                block["height_px"] for block in blocks
            })

            _log("\n===================================")
            _log("STAGE 1 - OPTIMIZED LARGE-AREA EXPORT")
            _log("===================================")
            _log(
                f"Working bounding rectangle: "
                f"{width_px} x {height_px} px"
            )
            _log(
                f"Current optimized large-block cap: "
                f"{current_limit} px"
            )
            _log(
                f"Balanced large-block candidates: "
                f"{len(all_candidate_blocks)}"
            )
            _log(
                f"Large blocks that WILL be exported: "
                f"{len(blocks)}"
            )
            _log(
                "Large blocks skipped before World Imagery export "
                f"(no requested-extent intersection): "
                f"{len(skipped_large_blocks)}"
            )
            _log(
                f"Balanced large-block grid: "
                f"{grid_rows} row(s) x {grid_cols} column(s)"
            )
            _log(f"Large block widths: {widths}")
            _log(f"Large block heights: {heights}")
            _log("Inter-block overlap: 0 px")
            _log(
                "Large imagery blocks are extracted by "
                "ExportTrainingDataForDeepLearning."
            )
            _log(
                "Python only subdivides retained exported TIFFs into "
                "1024 model windows."
            )

            if cache_context:
                export_root = (
                    Path(cache_context["imagery_root"])
                    / f"large_exports_max_{current_limit}px"
                )
            else:
                export_root = _next_available_path(
                    Path(output_root)
                    / f"large_exports_max_{current_limit}px"
                )

            try:
                large_exports = _export_large_blocks(
                    input_raster=input_raster,
                    blocks=blocks,
                    spatial_ref=spatial_ref,
                    export_root=export_root,
                )
                break

            except ArcGUIImageryContentError:
                # Smaller requests do not repair genuinely black/no-data imagery.
                raise

            except ArcGUIGeoreferenceError:
                # A CRS/resolution mismatch is not a request-size problem.
                raise

            except ArcGUIExportSizeError as exc:
                next_limit = _next_smaller_large_export_limit(
                    current_limit
                )
                if (
                    next_limit is None
                    or size_retries >= MAX_LARGE_EXPORT_SIZE_RETRIES
                ):
                    raise RuntimeError(
                        "ArcGIS image-service export still exceeds/violates the "
                        "request-size behavior after progressively reducing large "
                        f"blocks to {current_limit}px. Last error: {exc}"
                    ) from exc

                size_retries += 1

                try:
                    if export_root.exists():
                        shutil.rmtree(export_root)
                except Exception as cleanup_exc:
                    _warn(
                        f"Could not remove failed export attempt folder "
                        f"{export_root}: {cleanup_exc}"
                    )

                _warn(
                    f"Large imagery request failed as a likely service-size "
                    f"constraint at {current_limit}px. Retrying the entire "
                    f"optimized export with <= {next_limit}px balanced, "
                    "non-overlapping blocks."
                )
                current_limit = next_limit
                continue

            except ArcGUIExportError as exc:
                # Some ArcGIS/image-service failures are reported generically.
                # Perform one smaller-block recovery attempt before surfacing the
                # original service/network error.
                next_limit = _next_smaller_large_export_limit(
                    current_limit
                )
                if (
                    generic_recovery_used
                    or next_limit is None
                ):
                    raise

                generic_recovery_used = True

                try:
                    if export_root.exists():
                        shutil.rmtree(export_root)
                except Exception as cleanup_exc:
                    _warn(
                        f"Could not remove failed export attempt folder "
                        f"{export_root}: {cleanup_exc}"
                    )

                _warn(
                    "ArcGIS imagery export failed without a specific service-size "
                    "message. ArcGUI will make one recovery attempt using smaller "
                    f"large blocks (<= {next_limit}px). Original error: {exc}"
                )
                current_limit = next_limit
                continue


        if cache_context:
            _complete_optimized_export_cache(
                cache_context=cache_context,
                large_exports=large_exports,
                blocks=blocks,
                skipped_large_blocks=skipped_large_blocks,
                actual_large_export_limit=current_limit,
            )
            _log(f"Reusable imagery cache completed: {cache_context['text_path']}")

    _log(f"Large imagery TIFFs available for inference: {len(large_exports)}")

    _log("\n===================================")
    _log("STAGE 2 - PYTHON CROPPING + MODEL INFERENCE")
    _log("===================================")
    _log("Model crop size: 1024 x 1024 px")
    _log(
        f"Model overlap: {float(overlap_percent):.2f}%"
    )
    _log(
        f"Model stride: "
        f"{overlap_percent_to_stride(overlap_percent)} px"
    )

    probability_dir = _next_available_path(
        Path(output_root) / "large_block_probabilities"
    )
    probability_dir.mkdir(parents=True, exist_ok=True)

    block_probability_rasters = []
    stats = []

    for index, item in enumerate(large_exports, start=1):
        _log(
            f"Processing large export "
            f"{index}/{len(large_exports)} "
            f"(block {item['block_id']})"
        )

        probability_path, block_stats = _infer_one_large_block(
            large_tif=item["path"],
            block_id=item["block_id"],
            session=session,
            model_profile=model_profile,
            batch_size=batch_size,
            overlap_percent=overlap_percent,
            probability_dir=probability_dir,
            mask_geometries=mask_geometries,
        )

        stats.append(block_stats)

        if probability_path is not None:
            block_probability_rasters.append(probability_path)

    if len(block_probability_rasters) != len(large_exports):
        missing_count = len(large_exports) - len(block_probability_rasters)
        raise ArcGUIImageryContentError(
            "Optimized inference did not produce a probability raster for every "
            "retained imagery block. ArcGUI is stopping instead of creating a "
            f"partial road mosaic. Missing block outputs={missing_count}. "
            "Check imagery coverage/black pixels for the selected polygon."
        )

    if not block_probability_rasters:
        skipped_blocks = sum(
            x.get("large_block_skipped_black", 0)
            for x in stats
        )
        black_crops = sum(
            x.get("model_crops_skipped_black", 0)
            for x in stats
        )
        raise ArcGUIImageryContentError(
            "No optimized probability raster was produced. "
            f"Black/empty large blocks={skipped_blocks}; "
            f"black model crops={black_crops}. "
            "Check World Imagery coverage and the selected extent."
        )

    _log("\n===================================")
    _log(
        "STAGE 3 - MOSAIC LARGE-BLOCK PROBABILITIES"
    )
    _log("===================================")

    # Large blocks are intended not to overlap. MEAN is used defensively so a
    # tiny ArcGIS alignment overlap cannot introduce an arbitrary FIRST seam.
    probability_mosaic = _mosaic_probability_rasters(
        block_probability_rasters,
        final_dir,
        "roads_probability_mosaic.tif",
        mosaic_method="MEAN",
    )

    total_crops = sum(
        x["model_crops_total"] for x in stats
    )
    intersecting_crops = sum(
        x.get(
            "model_crops_intersecting",
            x["model_crops_total"],
        )
        for x in stats
    )
    inferred_crops = sum(
        x["model_crops_inferred"] for x in stats
    )
    skipped_outside = sum(
        x.get("model_crops_skipped_outside_extent", 0)
        for x in stats
    )
    skipped_black = sum(
        x["model_crops_skipped_black"] for x in stats
    )
    skipped_blocks = sum(
        x["large_block_skipped_black"] for x in stats
    )

    if intersecting_crops > 0 and inferred_crops == 0:
        raise ArcGUIImageryContentError(
            "Model inference produced zero valid crops even though model "
            "windows intersected the requested extent."
        )

    _log(
        f"Planned 1024 model crops across large blocks: "
        f"{total_crops}"
    )
    _log(
        f"Crops intersecting requested extent: "
        f"{intersecting_crops}"
    )
    _log(
        "Crops skipped completely outside requested extent: "
        f"{skipped_outside}"
    )
    _log(f"Model crops inferred: {inferred_crops}")
    _log(
        f"Completely black crops skipped: {skipped_black}"
    )
    _log(
        "Completely black large blocks skipped after export: "
        f"{skipped_blocks}"
    )
    _log(
        "Large blocks skipped BEFORE export because they did not "
        f"intersect the requested extent: "
        f"{len(skipped_large_blocks)}"
    )

    return (
        probability_mosaic,
        len(large_exports),
        skipped_outside + skipped_black,
    )

# =========================================================
# SINGLE-EXTENT PIPELINE
# =========================================================


def run_road_tool(
    extent_layer,
    input_raster_name,
    output_root,
    model_path,
    overlap_percent,
    threshold,
    optimized_large_crop=True,
    batch_size=2,
    output_name_prefix=None,
    final_predictions_dir=None,
    add_output_to_map=True,
    reuse_matching_exports=True,
):
    """Run road segmentation for one selected ArcGIS extent."""

    if optimized_large_crop is None:
        raise ValueError("Optimized Large-Area Export option must be True or False.")
    if reuse_matching_exports is None:
        raise ValueError("Reuse Matching Exported Imagery option must be True or False.")

    if output_root is None or not str(output_root).strip():
        raise ValueError("Output Folder is required.")
    if not str(input_raster_name or "").strip():
        raise ValueError("Input Raster Layer Name is required.")

    output_root = _ensure_writable_output_directory(output_root)

    if output_name_prefix is None:
        output_name_prefix = _layer_display_name(extent_layer)
    output_name_prefix = _safe_name(output_name_prefix)

    if final_predictions_dir is not None:
        final_predictions_dir = Path(final_predictions_dir)
        final_predictions_dir.mkdir(parents=True, exist_ok=True)

    overlap_percent = float(overlap_percent)
    threshold = float(threshold)
    try:
        batch_numeric = float(batch_size)
    except Exception as exc:
        raise ValueError("ONNX batch size must be an integer.") from exc
    if not math.isfinite(batch_numeric) or not batch_numeric.is_integer():
        raise ValueError("ONNX batch size must be a finite whole number.")
    batch_size = int(batch_numeric)
    optimized_large_crop = _bool_from_tool_value(optimized_large_crop)
    reuse_matching_exports = _bool_from_tool_value(reuse_matching_exports)

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Prediction threshold must be a finite value between 0 and 1.")

    if batch_size <= 0:
        raise ValueError("ONNX batch size must be > 0.")
    if batch_size > MAX_ONNX_BATCH_SIZE:
        raise ValueError(
            f"ONNX batch size must be <= {MAX_ONNX_BATCH_SIZE}; received {batch_size}."
        )

    stride_pixels = overlap_percent_to_stride(overlap_percent)

    if not arcpy.Exists(extent_layer):
        raise ValueError(f"Extent layer not found: {extent_layer}")

    # Always clean temporary feature classes even if CRS projection, direct
    # planning, model loading, or imagery export fails before inference starts.
    temp_source_extent = None
    temp_projected_extent = None
    run_root = None
    run_status_path = None

    try:
        source_desc = _validate_polygon_extent_layer(extent_layer)
        source_spatial_ref = source_desc.spatialReference

        source_extent_fc, temp_source_extent = _materialize_extent_layer(
            extent_layer,
            output_name_prefix,
        )

        # Resolve one unambiguous imagery source before planning exports.
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        m = aprx.activeMap
        input_raster = _resolve_input_imagery_layer(m, input_raster_name)

        # MAP_SPACE export must have one explicit projected metric map CRS.
        try:
            active_map_sr = m.spatialReference
        except Exception as exc:
            raise RuntimeError(f"Could not read the active map coordinate system: {exc}") from exc
        if not _is_valid_spatial_reference(active_map_sr):
            raise RuntimeError(
                "The active ArcGIS Pro map has an Unknown/undefined coordinate system. "
                "Set the map to a projected metric CRS before running ArcGUI."
            )
        _validate_fixed_cell_export_crs(active_map_sr)

        selected_profile = detect_model_profile(model_path)
        model_slug = _model_output_slug(model_path, selected_profile)

        export_cache_config = _build_export_cache_config(
            extent_layer=extent_layer,
            input_raster=input_raster,
            input_raster_name=input_raster_name,
            active_map=m,
            overlap_percent=overlap_percent,
            optimized_large_crop=optimized_large_crop,
        )
        cache_context = _prepare_export_cache_context(
            extent_output_root=output_root,
            config=export_cache_config,
            reuse_matching_exports=reuse_matching_exports,
        )

        if cache_context.get("reuse"):
            _warn(
                "MATCHING EXPORT CACHE FOUND AND PRELOADED. "
                "World Imagery export will be skipped; the selected model will "
                "run on the existing georeferenced imagery TIFFs."
            )
            _log(f"Reused cache: {cache_context['cache_tag']}")
        else:
            _log(
                "No reusable matching imagery cache will be used. "
                "A fresh World Imagery export will be created and cached for "
                "future model runs."
            )
            _log(f"New cache: {cache_context['cache_tag']}")

        spatial_ref = _get_export_spatial_reference(
            input_raster,
            active_map=m,
            fallback_spatial_ref=source_spatial_ref,
        )
        _validate_fixed_cell_export_crs(spatial_ref)

        extent_fc, temp_projected_extent = (
            _project_feature_class_to_spatial_ref(
                source_extent_fc,
                spatial_ref,
                output_name_prefix,
            )
        )

        original_extent = _feature_extent_safe(
            extent_fc,
            label=f"projected production extent '{output_name_prefix}'",
        )

        export_extent = original_extent

        production_width_px, production_height_px = _extent_pixel_size(
            export_extent
        )

        # Read the exact requested polygon geometry BEFORE loading ONNX.
        mask_geometries = _read_geometries(
            extent_fc,
            label="requested extent",
        )

        direct_plan = None

        if not optimized_large_crop:
            # This is the authoritative direct-mode preflight. It catches
            # geometry/overlap issues and counts exact useful work before the
            # ONNX graph is loaded or any World Imagery request is made.
            direct_plan = _build_direct_tile_plan(
                export_extent=export_extent,
                spatial_ref=spatial_ref,
                mask_geometries=mask_geometries,
                stride_pixels=stride_pixels,
            )

            if not cache_context.get("reuse"):
                service_limit = _get_image_service_pixel_limit(input_raster)
                if (
                    service_limit is not None
                    and service_limit < FIXED_TILE_SIZE
                ):
                    raise ArcGUIExportError(
                        "The input image service reports a maximum request size "
                        f"smaller than the required {FIXED_TILE_SIZE}px model chip "
                        f"(reported limit={service_limit}px)."
                    )

            _log(
                f"Direct preflight: "
                f"{len(direct_plan['candidate_crops']):,} candidate tiles, "
                f"{len(direct_plan['active_crops']):,} retained, "
                f"{len(direct_plan['skipped_crops']):,} skipped before export."
            )

            if len(direct_plan["active_crops"]) > 5000:
                _warn(
                    f"Direct mode will require "
                    f"{len(direct_plan['active_crops']):,} retained model "
                    "windows. The filtering is correct, but optimized "
                    "large-area mode is normally much faster for an area "
                    "this large."
                )


        # Older/pre-r7 exports are intentionally not auto-adopted in r7.
        # An export is reused only if it was created by the current exact-polygon
        # cache schema and exactly matches the current configuration.

        # Production performs the authoritative cache validation after the
        # true export CRS and direct plan are known, but before ONNX loading.
        cache_context = _runtime_preflight_cached_imagery(
            cache_context=cache_context,
            optimized_large_crop=optimized_large_crop,
            spatial_ref=spatial_ref,
            direct_plan=direct_plan,
            export_extent=export_extent,
            mask_geometries=mask_geometries,
        )

        if cache_context.get("reuse"):
            _warn(
                "EXPORT CACHE READY: cached imagery passed production preflight; "
                "the ONNX model will run directly on existing TIFFs."
            )

        _log("\\n===================================")
        _log(f"ARCGUI ROAD SEGMENTATION - {ARCGUI_VERSION}")
        _log("===================================")
        _log(f"Extent: {extent_layer}")
        _log(f"Input raster: {input_raster.name}")
        _log(
            f"Extent source CRS: "
            f"{_spatial_reference_label(source_spatial_ref)}"
        )
        _log(
            f"Map/export CRS: "
            f"{_spatial_reference_label(spatial_ref)}"
        )
        _log(
            "All export-block coordinates are planned after projecting the "
            "extent into the selected MAP_SPACE export CRS."
        )
        _log(
            f"Working export extent: "
            f"{export_extent.XMin:.3f}, {export_extent.YMin:.3f}, "
            f"{export_extent.XMax:.3f}, {export_extent.YMax:.3f}"
        )
        _log(
            f"Working export size at {FIXED_CELL_SIZE:g} map-units/pixel: "
            f"{production_width_px} x {production_height_px} px"
        )
        _log(f"Model: {model_path}")
        _log("Fixed model crop: 1024 x 1024 px")
        _log(f"Overlap: {overlap_percent:.2f}%")
        _log(f"Derived stride: {stride_pixels} px")
        _log(f"Optimized large-area export: {optimized_large_crop}")
        _log("Extent policy: exact selected polygon")
        _log(f"Threshold: {threshold}")
        _log(f"Imagery cache: {cache_context['cache_tag']}")
        _log(f"Imagery cache reused: {bool(cache_context.get('reuse'))}")
        _log(f"Model output suffix: {model_slug}")

        # Load the network only after extent/CRS/direct-tile preflight
        # has succeeded.
        session, model_profile, effective_target = _build_onnx_session(
            model_path
        )

        if model_profile.get("static_batch_size") == 1 and batch_size != 1:
            _warn(
                f"The selected ONNX model has static batch=1. Requested batch "
                f"size {batch_size} will be reduced to 1."
            )
            batch_size = 1

        _log(f"Architecture: {model_profile['architecture']}")
        _log(f"Effective execution target: {effective_target.upper()}")
        _log(f"Effective ONNX batch size: {batch_size}")

        model_cache_root = (
            Path(output_root)
            / MODEL_OUTPUTS_DIRNAME
            / model_slug
            / cache_context["cache_tag"]
        )
        run_root = _next_available_path(model_cache_root / "run")
        run_root.mkdir(parents=True, exist_ok=True)
        run_status_path = run_root / "run_status.txt"
        run_status_path.write_text(
            f"RUNNING\nStarted UTC: {_utc_now_text()}\n",
            encoding="utf-8",
        )
        final_dir = run_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        _write_model_run_info(
            run_root=run_root,
            model_path=model_path,
            model_profile=model_profile,
            effective_target=effective_target,
            threshold=threshold,
            batch_size=batch_size,
            cache_context=cache_context,
        )

        if optimized_large_crop:
            probability_mosaic, export_count, skipped_count = (
                _run_optimized_large_crop_mode(
                    input_raster=input_raster,
                    export_extent=export_extent,
                    spatial_ref=spatial_ref,
                    mask_geometries=mask_geometries,
                    session=session,
                    model_profile=model_profile,
                    batch_size=batch_size,
                    overlap_percent=overlap_percent,
                    output_root=run_root,
                    final_dir=final_dir,
                    cache_context=cache_context,
                )
            )
        else:
            probability_mosaic, export_count, skipped_count = (
                _run_direct_export_mode(
                    input_raster=input_raster,
                    export_extent=export_extent,
                    spatial_ref=spatial_ref,
                    mask_geometries=mask_geometries,
                    session=session,
                    model_profile=model_profile,
                    batch_size=batch_size,
                    overlap_percent=overlap_percent,
                    output_root=run_root,
                    final_dir=final_dir,
                    direct_plan=direct_plan,
                    cache_context=cache_context,
                )
            )

        _log("\\n===================================")
        _log("FINALIZE - CLIP, THRESHOLD, CLEAN")
        _log("===================================")

        clean_raster, clipped_probability, binary_raster = (
            _finalize_probability_to_clean_roads(
                probability_mosaic=probability_mosaic,
                extent_fc=extent_fc,
                original_extent=original_extent,
                threshold=threshold,
                final_dir=final_dir,
                output_name_prefix=f"{output_name_prefix}_{model_slug}",
            )
        )

        copied_final_clean = None

        if final_predictions_dir is not None:
            model_final_predictions_dir = (
                Path(final_predictions_dir)
                / model_slug
                / cache_context["cache_tag"]
            )
            model_final_predictions_dir.mkdir(parents=True, exist_ok=True)
            copied_final_clean = _next_available_file_path(
                model_final_predictions_dir / clean_raster.name
            )

            arcpy.management.CopyRaster(
                str(clean_raster),
                str(copied_final_clean),
            )
            if not arcpy.Exists(str(copied_final_clean)):
                raise RuntimeError(
                    f"Final prediction copy was not created: {copied_final_clean}"
                )

            try:
                arcpy.management.DefineProjection(
                    str(copied_final_clean),
                    arcpy.Raster(str(clean_raster)).spatialReference,
                )
            except Exception:
                pass

        map_output = (
            copied_final_clean
            if copied_final_clean is not None
            else clean_raster
        )

        if add_output_to_map:
            try:
                m.addDataFromPath(str(map_output))
            except Exception as exc:
                _warn(f"Could not add final output to map: {exc}")

        _log("\\n===================================")
        _log("DONE")
        _log("===================================")
        _log(
            f"Mode: "
            f"{'optimized large-area export' if optimized_large_crop else 'direct 1024 export'}"
        )
        _log(f"Usable ArcGIS imagery exports processed: {export_count}")
        _log(
            f"Planned/model work skipped outside extent or rejected as "
            f"black duplicate: {skipped_count}"
        )
        _log(f"Probability mosaic: {probability_mosaic}")
        _log(f"Clipped probability: {clipped_probability}")
        _log(f"Binary prediction: {binary_raster}")
        _log(f"Final clean roads: {map_output}")
        _log(f"Shared imagery cache: {cache_context['cache_root']}")
        _log(f"Model-specific run folder: {run_root}")

        if run_status_path is not None:
            run_status_path.write_text(
                f"SUCCESS\nCompleted UTC: {_utc_now_text()}\nFinal output: {map_output}\n",
                encoding="utf-8",
            )

        return str(map_output)

    except Exception as exc:
        if run_status_path is not None:
            try:
                run_status_path.write_text(
                    f"FAILED\nCompleted UTC: {_utc_now_text()}\nError: {exc}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        raise

    finally:
        if (
            temp_projected_extent
            and arcpy.Exists(temp_projected_extent)
        ):
            try:
                arcpy.management.Delete(temp_projected_extent)
            except Exception:
                pass

        if temp_source_extent and arcpy.Exists(temp_source_extent):
            try:
                arcpy.management.Delete(temp_source_extent)
            except Exception:
                pass

# =========================================================
# MULTI-EXTENT PIPELINE
# =========================================================



def run_road_tool_multiple(
    extent_layers,
    input_raster_name,
    output_root,
    model_path,
    overlap_percent,
    threshold,
    optimized_large_crop=True,
    batch_size=2,
    reuse_matching_exports=True,
):
    if not extent_layers:
        raise ValueError("At least one extent layer must be selected.")
    if optimized_large_crop is None:
        raise ValueError("Optimized Large-Area Export option must be True or False.")
    if reuse_matching_exports is None:
        raise ValueError("Reuse Matching Exported Imagery option must be True or False.")
    optimized_large_crop = _bool_from_tool_value(optimized_large_crop)
    reuse_matching_exports = _bool_from_tool_value(reuse_matching_exports)
    if output_root is None or not str(output_root).strip():
        raise ValueError("Output Folder is required.")
    if not str(input_raster_name or "").strip():
        raise ValueError("Input Raster Layer Name is required.")

    normalized_extents = [str(x).strip().casefold() for x in extent_layers]
    if len(set(normalized_extents)) != len(normalized_extents):
        raise ValueError(
            "The same extent layer was supplied more than once. Remove duplicate "
            "extent entries before running ArcGUI."
        )

    output_root = _ensure_writable_output_directory(output_root)

    # Validate filename before starting a long job.
    profile = detect_model_profile(model_path)
    model_file = Path(str(model_path))
    if not model_file.exists() or not model_file.is_file():
        raise FileNotFoundError(f"ONNX model not found or not a file: {model_file}")
    if model_file.stat().st_size <= 0:
        raise ValueError(f"Selected ONNX model file is empty: {model_file}")

    # Validate numeric parameters once before checking out ArcGIS extensions.
    overlap_percent_to_stride(overlap_percent)
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("Prediction threshold must be a finite value between 0 and 1.")
    try:
        batch_numeric = float(batch_size)
    except Exception as exc:
        raise ValueError("ONNX batch size must be numeric.") from exc
    if not math.isfinite(batch_numeric) or not batch_numeric.is_integer():
        raise ValueError("ONNX batch size must be a finite whole number.")
    batch_value = int(batch_numeric)
    if batch_value <= 0 or batch_value > MAX_ONNX_BATCH_SIZE:
        raise ValueError(
            f"ONNX batch size must be between 1 and {MAX_ONNX_BATCH_SIZE}."
        )
    batch_size = batch_value

    final_predictions_dir = output_root / "final_predictions"
    final_predictions_dir.mkdir(parents=True, exist_ok=True)

    _log("\n===================================")
    _log("ARCGUI MULTI-EXTENT JOB")
    _log("===================================")
    _log(f"Model: {profile['display_name']}")
    _log(f"Extent count: {len(extent_layers)}")
    _log(
        f"Optimized large-area export: "
        f"{_bool_from_tool_value(optimized_large_crop)}"
    )
    _log(
        f"Reuse matching exported imagery: "
        f"{_bool_from_tool_value(reuse_matching_exports)}"
    )

    outputs = []
    failures = []
    used_extent_names = {}
    checked_out_extensions = []

    try:
        checked_out_extensions = _ensure_required_extensions()

        for idx, extent_layer in enumerate(
            extent_layers,
            start=1,
        ):
            base_extent_name = _layer_display_name(extent_layer)
            occurrence = (
                used_extent_names.get(base_extent_name, 0) + 1
            )
            used_extent_names[base_extent_name] = occurrence

            extent_name = (
                base_extent_name
                if occurrence == 1
                else f"{base_extent_name}_{occurrence}"
            )

            extent_output_root = (
                output_root / extent_name
            )

            _log("\n===================================")
            _log(
                f"PROCESSING EXTENT {idx}/{len(extent_layers)}: "
                f"{extent_name}"
            )
            _log("===================================")

            try:
                output = run_road_tool(
                    extent_layer=extent_layer,
                    input_raster_name=input_raster_name,
                    output_root=extent_output_root,
                    model_path=model_path,
                    overlap_percent=overlap_percent,
                    threshold=threshold,
                    optimized_large_crop=optimized_large_crop,
                    batch_size=batch_size,
                    reuse_matching_exports=reuse_matching_exports,
                    output_name_prefix=extent_name,
                    final_predictions_dir=final_predictions_dir,
                    add_output_to_map=True,
                )
                outputs.append(output)

            except Exception as exc:
                _warn(
                    f"FAILED EXTENT: {extent_layer} -> {exc}"
                )
                failures.append(
                    (extent_layer, str(exc))
                )

    finally:
        _checkin_extensions(checked_out_extensions)

    _log("\n===================================")
    _log("MULTI-EXTENT SUMMARY")
    _log("===================================")
    _log(f"Successful: {len(outputs)}")
    _log(f"Failed: {len(failures)}")
    _log(
        f"Final predictions folder: "
        f"{final_predictions_dir}"
    )

    if failures:
        _log("Failed extents:")
        for layer, message in failures:
            _log(f" - {layer}: {message}")

    if not outputs:
        raise RuntimeError(
            "All requested extents failed. No road-segmentation output was "
            "created. Review the ArcGIS geoprocessing messages above."
        )

    return outputs, failures

# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":
    run_road_tool(
        extent_layer="extent_file",
        input_raster_name="World Imagery",
        output_root=r"C:\temp\arcgui_roads",
        model_path=r"E:\Models\segformer_cuda_fp16.onnx",
        overlap_percent=50,
        threshold=0.5,
        optimized_large_crop=True,
        batch_size=2,
        reuse_matching_exports=True,
    )
