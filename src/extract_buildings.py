from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import mapping, box
from shapely.validation import make_valid
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CITY_TIF = PROJECT_ROOT / "data" / "intermediate" / "geotiff" / "FILENAME.tif"
BUILDINGS_GPKG = PROJECT_ROOT / "data" / "intermediate" / "Buildings" / "FILENAME.gpkg"

OUTPUT_ROOT = PROJECT_ROOT / "output"
TRAIN_GEOTIFF_DIR = OUTPUT_ROOT / "train" / "geotiff"
TRAIN_PREVIEW_DIR = OUTPUT_ROOT / "train" / "preview"
VAL_GEOTIFF_DIR = OUTPUT_ROOT / "val" / "geotiff"
VAL_PREVIEW_DIR = OUTPUT_ROOT / "val" / "preview"
MANIFEST_CSV = OUTPUT_ROOT / "manifest.csv"
LOG_FILE = OUTPUT_ROOT / "extraction.log"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
PREVIEW_EXT = ".png"
PREVIEW_MAX_SIZE = 512
PAD_FOR_CROP = True
PAD_WIDTH = 0.5
ALL_TOUCHED = False

BUILDING_ID_START = 0

def setup_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_output_dirs() -> None:
    for path in [TRAIN_GEOTIFF_DIR, TRAIN_PREVIEW_DIR, VAL_GEOTIFF_DIR, VAL_PREVIEW_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        try:
            geom = make_valid(geom)
        except Exception:
            return None
    if geom is None or geom.is_empty:
        return None
    return geom


def explode_if_needed(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    before = len(gdf)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    after = len(gdf)
    logging.info("After explode(): %s feature(s) -> %s polygon(s)", before, after)
    return gdf


def assign_building_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    candidate_cols = [
        "building_id", "BUILDING_ID", "Uniq_Id", "uniq_id", "id", "ID",
        "fid", "FID", "objectid", "OBJECTID", "name", "Name"
    ]
    id_col = next((c for c in candidate_cols if c in gdf.columns), None)

    ids = []
    seen = set()

    if id_col is not None:
        logging.info("Using column '%s' as base building ID where possible", id_col)
        for i, value in enumerate(gdf[id_col].tolist(), start=BUILDING_ID_START):
            text = str(value).strip() if value is not None else ""
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_")
            if not safe or safe in seen:
                safe = f"building_{i:06d}"
            seen.add(safe)
            ids.append(safe)
    else:
        logging.info("No suitable ID column found; using sequential IDs starting from %d", BUILDING_ID_START)
        ids = [f"building_{i:06d}" for i in range(BUILDING_ID_START, BUILDING_ID_START + len(gdf))]

    gdf = gdf.copy()
    gdf["building_id"] = ids
    return gdf


def assign_splits(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(gdf))
    rng.shuffle(idx)

    train_count = int(round(len(gdf) * TRAIN_RATIO))
    split = np.array(["val"] * len(gdf), dtype=object)
    split[idx[:train_count]] = "train"

    gdf = gdf.copy()
    gdf["split"] = split
    return gdf


def to_preview_image(data: np.ndarray, nodata: Optional[float]) -> Image.Image:
    arr = data.astype(np.float32)

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError("Unsupported raster band count for preview")

    out = np.zeros_like(arr, dtype=np.uint8)

    for b in range(arr.shape[0]):
        band = arr[b]
        valid = np.isfinite(band)
        if not valid.any():
            continue

        vals = band[valid]
        lo, hi = np.percentile(vals, [2, 98])

        if np.isclose(lo, hi):
            scaled = np.zeros_like(band, dtype=np.uint8)
            scaled[valid] = 255 if hi > 0 else 0
        else:
            scaled = np.clip((band - lo) / (hi - lo), 0, 1)
            scaled = (scaled * 255).astype(np.uint8)

        out[b] = scaled

    rgb = np.transpose(out, (1, 2, 0))
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    setup_logging()
    ensure_output_dirs()

    if not CITY_TIF.exists():
        raise FileNotFoundError(f"Missing city raster: {CITY_TIF}")
    if not BUILDINGS_GPKG.exists():
        raise FileNotFoundError(f"Missing building GeoPackage: {BUILDINGS_GPKG}")

    logging.info("Reading GeoPackage: %s", BUILDINGS_GPKG)
    gdf = gpd.read_file(BUILDINGS_GPKG)

    if gdf.empty:
        raise ValueError("GeoPackage contains no features")
    if gdf.crs is None:
        raise ValueError("GeoPackage CRS is missing")

    logging.info("Original geometry types:\n%s", gdf.geometry.geom_type.value_counts())
    gdf = explode_if_needed(gdf)
    logging.info("Geometry types after explode:\n%s", gdf.geometry.geom_type.value_counts())

    gdf["geometry"] = gdf.geometry.apply(fix_geometry)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No valid polygons remain after cleanup")

    with rasterio.open(CITY_TIF) as src:
        if src.crs is None:
            raise ValueError("Raster CRS is missing")

        if gdf.crs != src.crs:
            logging.info("Reprojecting building polygons from %s to %s", gdf.crs, src.crs)
            gdf = gdf.to_crs(src.crs)

        gdf = assign_building_ids(gdf)
        gdf = assign_splits(gdf)

                                                        
        left, bottom, right, top = src.bounds
        raster_bounds_geom = gpd.GeoSeries([box(left, bottom, right, top)], crs=src.crs).iloc[0]

        nodata = src.nodata if src.nodata is not None else 0

        records: List[Dict[str, Any]] = []

        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extracting buildings"):
            building_id = row["building_id"]
            split = row["split"]
            geom = row.geometry

            if not geom.intersects(raster_bounds_geom):
                logging.warning("Skipping %s: polygon outside raster extent", building_id)
                continue

            try:
                out_image, out_transform = mask(
                    src,
                    [mapping(geom)],
                    crop=True,
                    pad=PAD_FOR_CROP,
                    pad_width=PAD_WIDTH,
                    nodata=nodata,
                    all_touched=ALL_TOUCHED,
                    filled=True,
                )
            except Exception as exc:
                logging.warning("Skipping %s: mask failed: %s", building_id, exc)
                continue

            if out_image.size == 0 or out_image.shape[1] == 0 or out_image.shape[2] == 0:
                logging.warning("Skipping %s: empty crop", building_id)
                continue

            if np.all(out_image == nodata):
                logging.warning("Skipping %s: crop is only nodata", building_id)
                continue

            geotiff_dir = TRAIN_GEOTIFF_DIR if split == "train" else VAL_GEOTIFF_DIR
            preview_dir = TRAIN_PREVIEW_DIR if split == "train" else VAL_PREVIEW_DIR

            tif_name = f"{building_id}.tif"
            preview_name = f"{building_id}{PREVIEW_EXT}"
            tif_path = geotiff_dir / tif_name
            preview_path = preview_dir / preview_name

            meta = src.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": nodata,
                }
            )

            try:
                with rasterio.open(tif_path, "w", **meta) as dst:
                    dst.write(out_image)
            except Exception as exc:
                logging.warning("Failed writing %s: %s", tif_path, exc)
                continue

            try:
                preview = to_preview_image(out_image, nodata)
                preview.save(preview_path)
            except Exception as exc:
                logging.warning("Preview generation failed for %s: %s", building_id, exc)

            xmin, ymin, xmax, ymax = geom.bounds
            records.append(
                {
                    "filename": tif_name,
                    "split": split,
                    "building_id": building_id,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                }
            )

        manifest = pd.DataFrame(records)
        manifest.to_csv(MANIFEST_CSV, index=False)
        logging.info("Saved manifest to %s", MANIFEST_CSV)
        logging.info("Extraction complete. Total saved: %d", len(manifest))

        if not manifest.empty:
            last_building_id = manifest["building_id"].iloc[-1]
            logging.info("Last building ID saved: %s", last_building_id)
        else:
            logging.warning("No buildings were saved; no last building ID to log.")


if __name__ == "__main__":
    main()
