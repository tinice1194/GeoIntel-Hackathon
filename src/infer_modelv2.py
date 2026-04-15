from pathlib import Path
import gc

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


PROJECT_ROOT = Path(r"G:\GIS_AI_PROJECT")

INPUT_DIR = PROJECT_ROOT / "test" / "test_tiles" 
OUTPUT_DIR = PROJECT_ROOT / "test" / "outputs" / "final_predictions"
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "deeplab_best.pth"


PATCH_SIZE = 256
NUM_CLASSES = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = deeplabv3_resnet50(weights=None, aux_loss=True) 

    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    model = model.to(DEVICE)

    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)   

    model.eval()
    return model


def pad_patch_to_size(img, patch_size):
    bands, h, w = img.shape

    if h == patch_size and w == patch_size:
        return img, h, w

    padded = np.zeros((bands, patch_size, patch_size), dtype=img.dtype)
    padded[:, :h, :w] = img
    return padded, h, w


def predict_single_raster(model, input_raster, output_raster):
    output_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_raster) as src:
        height, width = src.height, src.width
        pred_mask = np.zeros((height, width), dtype=np.uint8)

        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):

                win_w = min(PATCH_SIZE, width - x)
                win_h = min(PATCH_SIZE, height - y)

                window = Window(x, y, win_w, win_h)

                img = src.read([1, 2, 3], window=window).astype(np.float32)
                img, orig_h, orig_w = pad_patch_to_size(img, PATCH_SIZE)

                
                img = img / 255.0

                img = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE)

                with torch.no_grad():
                    logits = model(img)["out"]   
                    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                pred = pred[:orig_h, :orig_w]
                pred_mask[y:y + orig_h, x:x + orig_w] = pred

                del img, logits, pred

        meta = src.meta.copy()
        meta.update(
            count=1,
            dtype=rasterio.uint8,
            compress="lzw"
        )

        with rasterio.open(output_raster, "w", **meta) as dst:
            dst.write(pred_mask, 1)

    del pred_mask
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved prediction mask to: {output_raster}")


def predict_all_rasters():
    tif_files = sorted(INPUT_DIR.glob("*.tif"))

    if not tif_files:
        print(f"No .tif files found in: {INPUT_DIR}")
        return

    model = load_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for input_raster in tif_files:
        output_raster = OUTPUT_DIR / f"{input_raster.stem}_pred_mask_final.tif"
        print(f"Processing: {input_raster.name}")
        predict_single_raster(model, input_raster, output_raster)

    print("All rasters processed successfully.")


if __name__ == "__main__":
    predict_all_rasters()
