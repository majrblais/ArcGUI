# =========================================================
# SINGLE-CELL ARCGIS PRO NOTEBOOK PIPELINE
# Export one test extent -> ONNX prediction -> mosaic -> clip/clean
# No .pyt / .py import required
# =========================================================

from pathlib import Path
import os, sys, glob
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# USER VARIABLES
# ---------------------------------------------------------
EXTENT_LAYER = "test"          # polygon/extent layer in current ArcGIS Pro map
INPUT_RASTER_NAME = "World Imagery"

MODEL_PATH = r"E:\Desktop\ccnb\best_unet_resnet101_norm_1024.onnx"
OUTPUT_ROOT = r"C:\temp\arcgis_roads_notebook_test"

STRIDE_X = 1024
STRIDE_Y = 1024
THRESHOLD = 0.5
BATCH_SIZE = 4

USE_EXTENT_BUFFER = True
BUFFER_PIXELS = 100

# ---------------------------------------------------------
# FIXED PARAMETERS
# ---------------------------------------------------------
TILE_SIZE = 1024
CELL_SIZE = 2.0
MODEL_INPUT_SIZE = 1024

# ---------------------------------------------------------
# CUDA / ONNX DLL FIX
# ---------------------------------------------------------
_ENV = sys.prefix
_NVIDIA_ROOT = Path(_ENV) / "Lib" / "site-packages" / "nvidia"

for dll_dir in glob.glob(str(_NVIDIA_ROOT / "*" / "bin")):
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

_LIBRARY_BIN = Path(_ENV) / "Library" / "bin"
if _LIBRARY_BIN.exists():
    os.add_dll_directory(str(_LIBRARY_BIN))
    os.environ["PATH"] = str(_LIBRARY_BIN) + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------
# IMPORT ARCGIS / ONNX
# ---------------------------------------------------------
import arcpy
import onnxruntime as ort
from arcpy.ia import ExportTrainingDataForDeepLearning
from arcpy.sa import SetNull

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("ImageAnalyst")
arcpy.CheckOutExtension("Spatial")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def prepare_input_rgb_imagenet(image_rgb):
    img = cv2.resize(
        image_rgb,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1).astype(np.float32)
    return np.expand_dims(img, axis=0)

def output_to_mask(raw_output, threshold):
    pred = np.asarray(raw_output)

    if pred.ndim == 4:
        if pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        elif pred.shape[-1] == 1:
            pred = pred[:, :, :, 0]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {raw_output.shape}")
    elif pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)

    pred = sigmoid(pred)
    return (pred > threshold).astype(np.uint8)

def chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# ---------------------------------------------------------
# OUTPUT FOLDERS
# ---------------------------------------------------------
OUTPUT_ROOT = Path(OUTPUT_ROOT)
EXPORT_ROOT = OUTPUT_ROOT / "export_training_data"
EXPORT_IMG_DIR = EXPORT_ROOT / "images"
PRED_RASTER_DIR = OUTPUT_ROOT / "prediction_rasters"
FINAL_DIR = OUTPUT_ROOT / "final"

for folder in [EXPORT_ROOT, PRED_RASTER_DIR, FINAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FINAL_MOSAIC = FINAL_DIR / "roads_prediction_mosaic.tif"
FINAL_CLIPPED = FINAL_DIR / "roads_prediction_clipped.tif"
FINAL_CLEAN = FINAL_DIR / "roads_prediction_clean.tif"

print("Output root:", OUTPUT_ROOT)

# ---------------------------------------------------------
# GET EXTENT LAYER AND INPUT RASTER FROM CURRENT MAP
# ---------------------------------------------------------
if not arcpy.Exists(EXTENT_LAYER):
    raise RuntimeError(f"Extent layer not found: {EXTENT_LAYER}")

extent_desc = arcpy.Describe(EXTENT_LAYER)
extent_fc = extent_desc.catalogPath
spatial_ref = extent_desc.spatialReference
original_extent = extent_desc.extent

aprx = arcpy.mp.ArcGISProject("CURRENT")
m = aprx.activeMap

layers = m.listLayers(INPUT_RASTER_NAME)
if not layers:
    raise RuntimeError(f"Input raster layer not found in active map: {INPUT_RASTER_NAME}")

input_raster = layers[0]

print("Extent layer:", EXTENT_LAYER)
print("Input raster:", input_raster.name)

# ---------------------------------------------------------
# LOAD ONNX MODEL
# ---------------------------------------------------------
available = ort.get_available_providers()
providers = []

if "CUDAExecutionProvider" in available:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Active session providers:", session.get_providers())
print("ONNX input:", input_name)
print("ONNX output:", output_name)

# ---------------------------------------------------------
# CREATE EXPORT EXTENT
# ---------------------------------------------------------
if USE_EXTENT_BUFFER:
    buffer_map_units = BUFFER_PIXELS * CELL_SIZE
    export_extent = arcpy.Extent(
        original_extent.XMin - buffer_map_units,
        original_extent.YMin - buffer_map_units,
        original_extent.XMax + buffer_map_units,
        original_extent.YMax + buffer_map_units
    )
else:
    export_extent = original_extent

export_extent_str = (
    f"{export_extent.XMin} {export_extent.YMin} "
    f"{export_extent.XMax} {export_extent.YMax}"
)

print("Export extent:", export_extent_str)

# ---------------------------------------------------------
# STAGE 1 - EXPORT TRAINING DATA
# ---------------------------------------------------------
print("\nSTAGE 1 - ExportTrainingDataForDeepLearning")

with arcpy.EnvManager(
    extent=export_extent_str,
    cellSize=CELL_SIZE,
    snapRaster=None,
    mask=None,
    outputCoordinateSystem=spatial_ref
):
    ExportTrainingDataForDeepLearning(
        in_raster=input_raster,
        out_folder=str(EXPORT_ROOT),
        image_chip_format="TIFF",
        tile_size_x=TILE_SIZE,
        tile_size_y=TILE_SIZE,
        stride_x=STRIDE_X,
        stride_y=STRIDE_Y,
        output_nofeature_tiles="ALL_TILES",
        metadata_format="Export_Tiles",
        start_index=0
    )

tifs = sorted(EXPORT_IMG_DIR.glob("*.tif"))
print("Exported tiles:", len(tifs))

if not tifs:
    raise RuntimeError("No tiles were exported.")

# ---------------------------------------------------------
# READ TILE GEOREFERENCE INFO
# ---------------------------------------------------------
tile_info = []

for tif in tifs:
    r = arcpy.Raster(str(tif))
    tile_info.append({
        "path": tif,
        "uid": tif.stem,
        "extent": r.extent,
        "cell_w": r.meanCellWidth,
        "cell_h": r.meanCellHeight,
        "spatial_ref": r.spatialReference
    })

# ---------------------------------------------------------
# STAGE 2 - ONNX PREDICTION
# ---------------------------------------------------------
print("\nSTAGE 2 - ONNX prediction")

pred_rasters = []

for batch_idx, batch in enumerate(chunks(tile_info, BATCH_SIZE), start=1):
    print(f"Batch {batch_idx}")

    tensors = []
    prepared = []

    for item in batch:
        img = np.array(Image.open(item["path"]).convert("RGB"))
        h, w = img.shape[:2]

        tensors.append(prepare_input_rgb_imagenet(img))
        prepared.append((item, h, w))

    batch_input = np.concatenate(tensors, axis=0).astype(np.float32)
    raw = session.run([output_name], {input_name: batch_input})[0]
    masks = output_to_mask(raw, THRESHOLD)

    for mask, (item, h, w) in zip(masks, prepared):
        ext = item["extent"]

        mask_big = cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        raster = arcpy.NumPyArrayToRaster(
            mask_big.astype(np.uint8) * 255,
            arcpy.Point(ext.XMin, ext.YMin),
            item["cell_w"],
            item["cell_h"],
            value_to_nodata=0
        )

        out_raster = PRED_RASTER_DIR / f"{item['uid']}.tif"
        raster.save(str(out_raster))
        arcpy.management.DefineProjection(str(out_raster), item["spatial_ref"])

        pred_rasters.append(str(out_raster))

print("Prediction rasters:", len(pred_rasters))

if not pred_rasters:
    raise RuntimeError("No prediction rasters were created.")

# ---------------------------------------------------------
# STAGE 3 - MOSAIC PREDICTIONS
# ---------------------------------------------------------
print("\nSTAGE 3 - Mosaic")

for p in [FINAL_MOSAIC, FINAL_CLIPPED, FINAL_CLEAN]:
    if arcpy.Exists(str(p)):
        arcpy.management.Delete(str(p))

first_pred = arcpy.Raster(pred_rasters[0])

arcpy.management.MosaicToNewRaster(
    input_rasters=";".join(pred_rasters),
    output_location=str(FINAL_DIR),
    raster_dataset_name_with_extension=FINAL_MOSAIC.name,
    coordinate_system_for_the_raster=first_pred.spatialReference,
    pixel_type="8_BIT_UNSIGNED",
    cellsize=first_pred.meanCellWidth,
    number_of_bands=1,
    mosaic_method="MAXIMUM",
    mosaic_colormap_mode="FIRST"
)

print("Mosaic:", FINAL_MOSAIC)

# ---------------------------------------------------------
# STAGE 4 - CLIP BACK TO ORIGINAL EXTENT GEOMETRY
# ---------------------------------------------------------
print("\nSTAGE 4 - Clip to original extent")

original_extent_str = (
    f"{original_extent.XMin} {original_extent.YMin} "
    f"{original_extent.XMax} {original_extent.YMax}"
)

arcpy.management.Clip(
    in_raster=str(FINAL_MOSAIC),
    rectangle=original_extent_str,
    out_raster=str(FINAL_CLIPPED),
    in_template_dataset=extent_fc,
    nodata_value="0",
    clipping_geometry="ClippingGeometry",
    maintain_clipping_extent="MAINTAIN_EXTENT"
)

arcpy.management.DefineProjection(str(FINAL_CLIPPED), first_pred.spatialReference)

print("Clipped:", FINAL_CLIPPED)

# ---------------------------------------------------------
# STAGE 5 - CLEAN BACKGROUND TO NODATA
# ---------------------------------------------------------
print("\nSTAGE 5 - Clean output")

SetNull(str(FINAL_CLIPPED), str(FINAL_CLIPPED), "VALUE = 0").save(str(FINAL_CLEAN))
arcpy.management.DefineProjection(str(FINAL_CLEAN), first_pred.spatialReference)

try:
    arcpy.management.CalculateStatistics(str(FINAL_CLEAN))
    arcpy.management.BuildPyramids(str(FINAL_CLEAN))
except Exception as e:
    print("Warning while building stats/pyramids:", e)

# ---------------------------------------------------------
# ADD FINAL OUTPUT TO MAP
# ---------------------------------------------------------
try:
    m.addDataFromPath(str(FINAL_CLEAN))
except Exception as e:
    print("Could not add output to map:", e)

print("\nDONE")
print("Final clean raster:")
print(FINAL_CLEAN)