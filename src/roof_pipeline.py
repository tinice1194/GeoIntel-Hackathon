                       

from pathlib import Path
import gc
import warnings
from collections import defaultdict

import cv2
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from scipy import ndimage
import torch
from tqdm import tqdm
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, box

from unet_model import UNet

warnings.filterwarnings("ignore")

PROJECT_ROOT        = Path(r"G:\GIS_AI_PROJECT")
GEOTIFF_DIR         = PROJECT_ROOT / "data" / "intermediate" / "geotiff"
PREDICTIONS_DIR     = PROJECT_ROOT / "outputs" / "predictions"
ROOF_DIR            = PROJECT_ROOT / "outputs" / "roof_results"
CNN2_PKL            = PROJECT_ROOT / "models" / "roof_classified.pkl"
UNET_CKPT           = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_best.pth"

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE          = 128
CROP_SIZE           = 128
BUILT_UP_CLASS_ID   = 1
MIN_BUILDING_PIXELS = 10
PAD                 = 8
SAVE_PRED_MASK      = True
SKIP_DONE           = True

BUILDING_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "building_id": "int",
        "roof_class":  "str",
        "area_pixels": "int",
        "area_m2":     "float",
    },
}


def load_models():
    print("Loading CNN2...")
    cnn2 = joblib.load(CNN2_PKL)
    print(f"  CNN2 classes: {cnn2.classes_}")

    print("Loading U-Net...")
    model = UNet(in_channels=3, num_classes=11, base_ch=32).to(DEVICE)
    model.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE))
    model.eval()
    print(f"  Device: {DEVICE}")
    return cnn2, model


def infer_unet_tile(model, tile_rgb):
    tile = torch.from_numpy(tile_rgb).unsqueeze(0).float().to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(tile)
        else:
            logits = model(tile)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    del tile, logits
    return pred


def predict_full_raster(model, raster_path):
    with rasterio.open(raster_path) as src:
        height, width  = src.height, src.width
        transform, crs = src.transform, src.crs
        pred_mask = np.zeros((height, width), dtype=np.uint8)

        for y in tqdm(range(0, height, PATCH_SIZE), desc="  U-Net tiles", leave=False):
            for x in range(0, width, PATCH_SIZE):
                win_h = min(PATCH_SIZE, height - y)
                win_w = min(PATCH_SIZE, width - x)
                window = Window(x, y, win_w, win_h)

                patch = src.read([1, 2, 3], window=window).astype(np.float32)
                ph, pw = patch.shape[1], patch.shape[2]

                if ph < PATCH_SIZE or pw < PATCH_SIZE:
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, PATCH_SIZE - ph), (0, PATCH_SIZE - pw)),
                        mode="constant"
                    )

                patch /= 255.0
                pred_tile = infer_unet_tile(model, patch)
                pred_mask[y:y + win_h, x:x + win_w] = pred_tile[:win_h, :win_w]

                del patch, pred_tile

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    return pred_mask, transform, crs


def extract_features(rgb_crop):
    img_bgr = cv2.cvtColor(rgb_crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask = np.any(img_bgr > 15, axis=2).astype(np.uint8)

    if mask.sum() < 100:
        return None

    pixels_bgr = img_bgr[mask == 1]
    pixels_rgb = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3),
        cv2.COLOR_BGR2RGB
    ).reshape(-1, 3)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask == 1]

    mean_rgb = pixels_rgb.mean(axis=0)
    std_rgb = pixels_rgb.std(axis=0)
    mean_hsv = hsv_pixels.mean(axis=0)
    std_hsv = hsv_pixels.std(axis=0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = gray[mask == 1]

    edges = cv2.Canny(gray, 60, 140)
    edge_density = edges[mask == 1].mean() / 255.0

    texture_mean = gray_masked.mean()
    texture_std = gray_masked.std()

    h_hist = cv2.calcHist([hsv], [0], mask, [12], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [12], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [12], [0, 256]).flatten()

    h_hist /= (h_hist.sum() + 1e-8)
    s_hist /= (s_hist.sum() + 1e-8)
    v_hist /= (v_hist.sum() + 1e-8)

    feats = np.concatenate([
        mean_rgb, std_rgb,
        mean_hsv, std_hsv,
        [texture_mean, texture_std, edge_density],
        h_hist, s_hist, v_hist
    ]).astype(np.float32)

    return feats.reshape(1, -1)


def fast_extract_components(building_mask, min_pixels=MIN_BUILDING_PIXELS):
    print("  Labeling components...", flush=True)
    labeled, num = ndimage.label(building_mask)
    print(f"  Raw components: {num:,}", flush=True)

    sizes = ndimage.sum(building_mask, labeled, range(1, num + 1))
    objects = ndimage.find_objects(labeled)
    print("  Got component slices.", flush=True)

    components = []
    for label_id, (slc, area) in enumerate(zip(objects, sizes), start=1):
        if slc is None:
            continue
        area = int(area)
        if area < min_pixels:
            continue

        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop

        components.append({
            "label": label_id,
            "area": area,
            "bbox": (y0, x0, y1, x1),
        })

    del labeled, sizes, objects
    gc.collect()

    print(f"  After size filter (>={min_pixels}px): {len(components):,} components", flush=True)
    return components


def crop_rgb_from_bbox(src_img, bbox, pad=PAD, target_size=CROP_SIZE):
    minr, minc, maxr, maxc = bbox

    minr = max(0, minr - pad)
    minc = max(0, minc - pad)
    maxr = min(src_img.height, maxr + pad)
    maxc = min(src_img.width, maxc + pad)

    window = Window(minc, minr, maxc - minc, maxr - minr)
    crop = src_img.read([1, 2, 3], window=window)
    crop = np.transpose(crop, (1, 2, 0)).astype(np.uint8)
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop


def bbox_to_polygon(transform, bbox):
    minr, minc, maxr, maxc = bbox

    x_left, y_top = xy(transform, minr, minc, offset="ul")
    x_right, y_bottom = xy(transform, maxr, maxc, offset="lr")

    return box(x_left, y_bottom, x_right, y_top)


def write_gpkg(components_by_class, crs, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    try:
        epsg = crs.to_epsg()
        fiona_crs = from_epsg(epsg) if epsg else {"wkt": crs.to_wkt()}
    except Exception:
        fiona_crs = {"wkt": crs.to_wkt()}

    total = 0
    for roof_class, rows in sorted(components_by_class.items()):
        if not rows:
            continue

        layer_name = f"roof_{roof_class}"
        with fiona.open(
            str(out_path),
            "w",
            driver="GPKG",
            crs=fiona_crs,
            schema=BUILDING_SCHEMA,
            layer=layer_name,
        ) as sink:
            for row in rows:
                sink.write({
                    "geometry": mapping(row["polygon"]),
                    "properties": {
                        "building_id": row["building_id"],
                        "roof_class": row["roof_class"],
                        "area_pixels": row["area_pixels"],
                        "area_m2": row["area_m2"],
                    },
                })

        total += len(rows)
        print(f"    Layer 'roof_{roof_class}': {len(rows):,} buildings")

    print(f"  GeoPackage saved -> {out_path} ({total} total)")


def save_prediction_mask(pred_mask, transform, crs, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=pred_mask.shape[0],
        width=pred_mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(pred_mask, 1)


def process_single_raster(raster_path, cnn2, unet_model):
    stem = raster_path.stem
    gpkg_path = ROOF_DIR / stem / f"{stem}_roofs.gpkg"

    if SKIP_DONE and gpkg_path.exists():
        print(f"  Skipping {stem} (already done)")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"  Raster : {stem}")
    print(f"{'='*60}", flush=True)

    pred_mask, transform, crs = predict_full_raster(unet_model, raster_path)

    if SAVE_PRED_MASK:
        save_prediction_mask(pred_mask, transform, crs, PREDICTIONS_DIR / f"{stem}_pred_mask.tif")
        print("  Prediction mask saved.", flush=True)

    building_mask = (pred_mask == BUILT_UP_CLASS_ID).astype(np.uint8)
    del pred_mask
    gc.collect()

    pixel_area_m2 = abs(transform.a * transform.e)
    components = fast_extract_components(building_mask)
    del building_mask
    gc.collect()

    if not components:
        print("  No buildings found above size threshold.")
        return 0, 0

    components_by_class = defaultdict(list)
    detailed_rows = []
    bid = 0

    with rasterio.open(raster_path) as src_img:
        for comp in tqdm(components, desc="  Classifying", leave=False, unit="bldg"):
            crop = crop_rgb_from_bbox(src_img, comp["bbox"])
            feats = extract_features(crop)
            del crop

            roof_class = "unknown" if feats is None else str(cnn2.predict(feats)[0])
            del feats

            bid += 1
            area_m2 = round(comp["area"] * pixel_area_m2, 2)
            polygon = bbox_to_polygon(transform, comp["bbox"])

            row = {
                "building_id": bid,
                "roof_class": roof_class,
                "area_pixels": comp["area"],
                "area_m2": area_m2,
                "polygon": polygon,
                "bbox_min_row": comp["bbox"][0],
                "bbox_min_col": comp["bbox"][1],
                "bbox_max_row": comp["bbox"][2],
                "bbox_max_col": comp["bbox"][3],
            }
            components_by_class[roof_class].append(row)
            detailed_rows.append({k: v for k, v in row.items() if k != "polygon"})

            if bid % 100 == 0:
                gc.collect()

    print("  Writing GeoPackage...", flush=True)
    write_gpkg(components_by_class, crs, gpkg_path)

    df = pd.DataFrame(detailed_rows)
    df.to_csv(ROOF_DIR / stem / "buildings_detailed.csv", index=False)

    summary = (df.groupby("roof_class")
                 .agg(num_buildings=("building_id", "count"),
                      total_area_m2=("area_m2", "sum"),
                      avg_area_m2=("area_m2", "mean"))
                 .reset_index())
    summary.to_csv(ROOF_DIR / stem / "roof_summary.csv", index=False)
    print(summary.to_string(index=False))

    del components, components_by_class, detailed_rows, df
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"  DONE: {bid} buildings -> {gpkg_path}", flush=True)
    return bid, len(summary)

def main():
    print("=== FAST CNN1 -> CNN2 -> GeoPackage Pipeline (Single Raster Test) ===")
    cnn2, unet_model = load_models()

    test_raster = GEOTIFF_DIR / "37458_fattu_bhila_ortho_3857.tif"

    if not test_raster.exists():
        print(f"Raster not found: {test_raster}")
        return

    total_buildings, total_rasters = 0, 0

    try:
        n_b, _ = process_single_raster(test_raster, cnn2, unet_model)
        total_buildings += n_b
        total_rasters += 1
    except Exception as e:
        import traceback
        print(f"\n  FAILED {test_raster.name}: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  SINGLE RASTER TEST COMPLETE")
    print(f"  Raster processed : {total_rasters}/1")
    print(f"  Total buildings  : {total_buildings}")
    print(f"  Results in       : {ROOF_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
