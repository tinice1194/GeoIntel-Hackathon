import shutil
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

LABEL_CSV = Path(r"G:/GIS_AI_PROJECT/output/roof_labels_1000.csv")
TRAIN_PREVIEW_DIR = Path(r"G:/GIS_AI_PROJECT/output/train/preview")
VAL_PREVIEW_DIR = Path(r"G:/GIS_AI_PROJECT/output/val/preview")

TRAIN_GEOTIFF_DIRS = [
    Path(r"G:/GIS_AI_PROJECT/output/train/geotiff"),
]
VAL_GEOTIFF_DIRS = [
    Path(r"G:/GIS_AI_PROJECT/output/val/geotiff"),
]

OUTPUT_DIR = Path(r"G:/GIS_AI_PROJECT/output/roof_sorted")
MODEL_PATH = OUTPUT_DIR / "roof_type_model.pkl"
TRAIN_CSV_OUT = OUTPUT_DIR / "train_predictions.csv"
VAL_CSV_OUT = OUTPUT_DIR / "val_predictions.csv"

CLASSES = ["rcc", "tin", "tiled", "granite_marble", "unknown"]
GEOTIFF_EXTS = {".tif", ".tiff", ".img", ".jp2", ".vrt"}
MIN_SAMPLES_FOR_TEST = 2
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.55
OVERSAMPLE_TO_MAX = True
MAX_OVERSAMPLE_MULTIPLIER = 4

def extract_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.any(img > 15, axis=2).astype(np.uint8)
    if mask.sum() < 100:
        return None

    pixels = img[mask == 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[mask == 1]

    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)
    mean_hsv = hsv_pixels.mean(axis=0)
    std_hsv = hsv_pixels.std(axis=0)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

    return feats

def build_geotiff_index(source_dirs):
    idx = {}
    for folder in source_dirs:
        if not folder.exists():
            continue
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix.lower() in GEOTIFF_EXTS:
                idx[f.stem] = f
    return idx

def load_labeled_feature_table(label_csv, preview_dir):
    df = pd.read_csv(label_csv)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(CLASSES)].copy()

    rows = []
    for _, row in df.iterrows():
        img_path = preview_dir / row["png_file"]
        if not img_path.exists():
            continue
        feats = extract_features(img_path)
        if feats is None:
            continue
        rows.append({
            "png_file": row["png_file"],
            "label": row["label"],
            "features": feats,
        })

    feat_df = pd.DataFrame(rows)
    if len(feat_df) < 10:
        raise ValueError("Not enough valid labeled PNG previews found to train.")
    return feat_df


def oversample_feature_table(feat_df):
    counts = feat_df["label"].value_counts()
    print("Original class counts:")
    print(counts.sort_index())

    if OVERSAMPLE_TO_MAX:
        target_count = counts.max()
    else:
        target_count = int(np.median(counts))

    balanced_parts = []
    for cls in counts.index:
        cls_df = feat_df[feat_df["label"] == cls]
        n_current = len(cls_df)
        n_target = min(target_count, n_current * MAX_OVERSAMPLE_MULTIPLIER)
        n_target = max(n_target, n_current)

        if n_current < n_target:
            cls_df_resampled = resample(
                cls_df,
                replace=True,
                n_samples=n_target,
                random_state=RANDOM_STATE,
            )
            balanced_parts.append(cls_df_resampled)
        else:
            balanced_parts.append(cls_df)

    out_df = pd.concat(balanced_parts, ignore_index=True)
    out_counts = out_df["label"].value_counts()
    print("Oversampled class counts:")
    print(out_counts.sort_index())
    return out_df

def train_model(label_csv, preview_dir):
    feat_df = load_labeled_feature_table(label_csv, preview_dir)
    class_counts = feat_df["label"].value_counts()

    use_stratify = class_counts.min() >= MIN_SAMPLES_FOR_TEST and len(feat_df) >= max(20, len(class_counts) * 2)

    if use_stratify:
        train_df, test_df = train_test_split(
            feat_df,
            test_size=max(0.2, len(class_counts) / len(feat_df)),
            random_state=RANDOM_STATE,
            stratify=feat_df["label"],
        )
    else:
        train_df, test_df = feat_df.copy(), None
        print("Skipping holdout split because one or more classes are too small.")

    train_df_balanced = oversample_feature_table(train_df)

    X_train = np.stack(train_df_balanced["features"].values)
    y_train = train_df_balanced["label"].values

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    if test_df is not None and len(test_df) > 0:
        X_test = np.stack(test_df["features"].values)
        y_test = test_df["label"].values
        preds = []
        for feats in X_test:
            proba = model.predict_proba([feats])[0]
            best_idx = int(np.argmax(proba))
            best_label = model.classes_[best_idx]
            best_conf = float(np.max(proba))
            if best_conf < CONFIDENCE_THRESHOLD:
                preds.append("unknown")
            else:
                preds.append(best_label)

        print("\nValidation report with low-confidence -> unknown:\n")
        print(classification_report(y_test, preds, zero_division=0))
    else:
        print("Model trained on all labeled samples. No holdout report generated.")

    return model

def predict_and_arrange(split_name, preview_dir, geotiff_dirs, model, out_csv):
    rows = []
    geotiff_index = build_geotiff_index(geotiff_dirs)

    png_root_out = OUTPUT_DIR / split_name / "preview_by_class"
    tif_root_out = OUTPUT_DIR / split_name / "geotiff_by_class"

    for cls in CLASSES:
        (png_root_out / cls).mkdir(parents=True, exist_ok=True)
        (tif_root_out / cls).mkdir(parents=True, exist_ok=True)

    for png_file in sorted(preview_dir.glob("*.png")):
        feats = extract_features(png_file)
        if feats is None:
            pred = "unknown"
            conf = 0.0
        else:
            proba = model.predict_proba([feats])[0]
            best_idx = int(np.argmax(proba))
            best_label = model.classes_[best_idx]
            best_conf = float(np.max(proba))
            if best_conf < CONFIDENCE_THRESHOLD:
                pred = "unknown"
                conf = best_conf
            else:
                pred = best_label
                conf = best_conf

        shutil.copy2(png_file, png_root_out / pred / png_file.name)

        matched_tif = geotiff_index.get(png_file.stem)
        matched_tif_name = ""
        if matched_tif is not None:
            shutil.copy2(matched_tif, tif_root_out / pred / matched_tif.name)
            matched_tif_name = matched_tif.name

        rows.append({
            "split": split_name,
            "png_file": png_file.name,
            "predicted_label": pred,
            "confidence": round(conf, 4),
            "matched_geotiff": matched_tif_name,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions: {out_csv}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_model(LABEL_CSV, TRAIN_PREVIEW_DIR)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Low-confidence threshold: {CONFIDENCE_THRESHOLD}")

    predict_and_arrange("train", TRAIN_PREVIEW_DIR, TRAIN_GEOTIFF_DIRS, model, TRAIN_CSV_OUT)
    predict_and_arrange("val", VAL_PREVIEW_DIR, VAL_GEOTIFF_DIRS, model, VAL_CSV_OUT)

    print("Done")

if __name__ == "__main__":
    main()
