from pathlib import Path
import os

import numpy as np
import rasterio
from rasterio import features
from rasterio.windows import Window
import geopandas as gpd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
INTERMEDIATE_GEOTIFF = INTERMEDIATE_DIR / "geotiff"
MASKS_FULL = INTERMEDIATE_DIR / "masks-full"
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"

SHAPE_DIR_CG2 = RAW_DATA_DIR / "CG" / "CG_Training_dataSet_2" / "shp-file"
SHAPE_DIR_CG3 = RAW_DATA_DIR / "CG" / "CG_Training_dataSet_3" / "shp-file"
SHAPE_DIR_PB  = RAW_DATA_DIR / "PB" / "PB_training_dataSet_shp_file" / "shp-file"

PATCH_SIZE = 256
MIN_LABEL_PIXELS = 50

CLASS_MAP = {
    "Built_Up_Area_type": 1,
    "Road": 2,
    "Road_Centre_Line": 3,
    "Railway": 4,
    "Bridge": 5,
    "Water_Body": 6,
    "Water_Body_Line": 7,
    "Waterbody_Point": 8,
    "Utility": 9,
    "Utility_Poly": 10,
}

PB_CLIPS = {
    "FILENAMES",
}

CG2_CLIPS = {
    "FILENAMES",
}

CG3_CLIPS = {
    "FILENAMES",
}


def rasterize_labels(raster_path: Path, shp_folder: Path, out_mask_path: Path):
    print(f"  Rasterizing labels for {raster_path.name}")
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        mask = np.zeros((height, width), dtype=np.uint8)

        for shp_name, class_id in CLASS_MAP.items():
            shp_file = shp_folder / f"{shp_name}.shp"
            if not shp_file.exists():
                continue

            gdf = gpd.read_file(shp_file)
            if gdf.empty:
                continue

            shapes = ((geom, class_id) for geom in gdf.geometry if geom is not None)
            burned = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            mask = np.maximum(mask, burned)

        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": rasterio.uint8})

        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_mask_path, "w", **meta) as dst:
            dst.write(mask, 1)


def create_patches(img_path: Path, mask_path: Path, split_ratios=(0.7, 0.15, 0.15)):
    print(f"  Creating patches for {img_path.name}")
    with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
        assert src_img.width == src_mask.width and src_img.height == src_mask.height

        height, width = src_img.height, src_img.width
        img_name = img_path.stem

        patch_coords = []
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                if x + PATCH_SIZE > width or y + PATCH_SIZE > height:
                    continue
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                mask_patch = src_mask.read(1, window=window)
                if np.count_nonzero(mask_patch) < MIN_LABEL_PIXELS:
                    continue
                patch_coords.append((x, y))

        print(f"    Kept {len(patch_coords)} patches with labels")
        np.random.shuffle(patch_coords)
        n = len(patch_coords)
        n_train = int(split_ratios[0] * n)
        n_val = int(split_ratios[1] * n)

        splits = {
            "train": patch_coords[:n_train],
            "val": patch_coords[n_train:n_train + n_val],
            "test": patch_coords[n_train + n_val:],
        }

        for split, coords in splits.items():
            img_out_dir = PATCHES_DIR / split / "images"
            mask_out_dir = PATCHES_DIR / split / "masks"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            mask_out_dir.mkdir(parents=True, exist_ok=True)

            for i, (x, y) in enumerate(tqdm(coords, desc=f"    {split}", leave=False)):
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)

                img_patch = src_img.read([1, 2, 3], window=window)
                img_patch = np.transpose(img_patch, (1, 2, 0))
                mask_patch = src_mask.read(1, window=window)

                img_file = img_out_dir / f"{img_name}_{split}_{i}.npy"
                mask_file = mask_out_dir / f"{img_name}_{split}_{i}.npy"

                np.save(img_file, img_patch.astype(np.float32))
                np.save(mask_file, mask_patch.astype(np.uint8))


def choose_shapefile_folder(tif: Path) -> Path:
    base = tif.stem.upper()

    if base in PB_CLIPS:
        print(f"  → Using PB shapefiles")
        return SHAPE_DIR_PB
    if base in CG2_CLIPS:
        print(f"  → Using CG2 shapefiles")
        return SHAPE_DIR_CG2
    if base in CG3_CLIPS:
        print(f"  → Using CG3 shapefiles")
        return SHAPE_DIR_CG3

    print(f"  [WARN] Unknown raster name, defaulting to PB: {tif.name}")
    return SHAPE_DIR_PB


def main():
    geotiffs = [
        t for t in INTERMEDIATE_GEOTIFF.rglob("*.tif")
        if not t.name.upper().endswith("_MASK.TIF")
    ]

    if not geotiffs:
        print(f"No .tif rasters found under {INTERMEDIATE_GEOTIFF}")
        return

    print("Found the following training rasters:")
    for t in geotiffs:
        print(" ", t)

    for tif in geotiffs:
        shp_folder = choose_shapefile_folder(tif)
        mask_path = MASKS_FULL / f"{tif.stem}_mask.tif"
        rasterize_labels(tif, shp_folder, mask_path)
        create_patches(tif, mask_path)

    print("Done. Patches are in:", PATCHES_DIR)


if __name__ == "__main__":
    main()
