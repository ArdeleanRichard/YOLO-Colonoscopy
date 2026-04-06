"""
Mask-to-YOLO conversion pipeline
────────────────────────────────────────────────────────────────────────────
Expected input layout
    root/
        images/   *.png | *.tif | *.jpg | ...
        masks/    *.png | *.tif | ...          (binary: 0 or 255)

Output layout produced by this script
    root/
        images/
            train/   val/   test/
        masks/
            train/   val/   test/
        labels/
            train/   val/   test/

Three public functions
    1. convert_masks_to_yolo  – masks/ → labels/ flat (one .txt per mask)
    2. split_dataset          – moves/copies everything into train/val/test
    3. visualise_sample       – 3-subplot figure: image | mask | image+bbox
────────────────────────────────────────────────────────────────────────────
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTS  = (".png", ".tif", ".tiff", ".bmp")


def _find_file(directory: Path, stem: str, exts: tuple):
    """Return the first existing file that matches stem + any extension."""
    for ext in exts:
        for candidate in (directory / f"{stem}{ext}",
                          directory / f"{stem}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def _collect_stems(directory: Path, exts: tuple):
    """Return sorted unique stems of all files with given extensions."""
    stems = []
    for ext in exts:
        for p in list(directory.glob(f"*{ext}")) + list(directory.glob(f"*{ext.upper()}")):
            stems.append(p.stem)
    return sorted(set(stems))


def _mask_to_yolo_lines(mask_path: Path, img_w: int, img_h: int,
                         class_id: int, min_area_px: int):
    """
    Read a binary mask and return YOLO-format strings, one per connected blob.
    Returns an empty list if no valid blobs are found.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    lines = []
    for lbl in range(1, num_labels):          # 0 is background
        x    = stats[lbl, cv2.CC_STAT_LEFT]
        y    = stats[lbl, cv2.CC_STAT_TOP]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        h    = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        xc = (x + w / 2) / img_w
        yc = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 1. Convert masks → flat labels/
# ─────────────────────────────────────────────────────────────────────────────

def convert_masks_to_yolo(
    root_dir    = ".",
    images_dir  = "images",
    masks_dir   = "masks",
    labels_dir  = "labels",
    class_id    = 0,
    min_area_px = 10,
):
    """
    Convert every mask in <root_dir>/<masks_dir>/ to a YOLO .txt label file
    written to <root_dir>/<labels_dir>/.

    The images are read only to obtain their true (W, H); the mask filename
    must match the image filename (same stem, any supported extension).

    YOLO format per line:
        <class_id> <x_center> <y_center> <width> <height>   (all normalised)
    """

    root     = Path(root_dir)
    images_p = root / images_dir
    masks_p  = root / masks_dir
    labels_p = root / labels_dir
    labels_p.mkdir(parents=True, exist_ok=True)

    mask_stems = _collect_stems(masks_p, MASK_EXTS)
    if not mask_stems:
        print(f"[WARNING] No masks found in '{masks_p}'")
        return

    ok = skipped = 0

    for stem in mask_stems:
        mask_path = _find_file(masks_p, stem, MASK_EXTS)
        img_path  = _find_file(images_p, stem, IMAGE_EXTS)

        if img_path is None:
            print(f"[SKIP] No matching image for mask '{stem}'")
            skipped += 1
            continue

        probe = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if probe is None:
            print(f"[SKIP] Cannot read image '{img_path}'")
            skipped += 1
            continue
        img_h, img_w = probe.shape[:2]

        lines = _mask_to_yolo_lines(mask_path, img_w, img_h, class_id, min_area_px)

        label_path = labels_p / f"{stem}.txt"
        label_path.write_text("\n".join(lines))

        print(f"[OK] {stem}: {len(lines)} box(es)  ->  {label_path}")
        ok += 1

    print(f"\nConversion done.  OK={ok}  Skipped={skipped}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Split into images/train|val|test  labels/train|val|test  masks/train|val|test
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    root_dir   = ".",
    images_dir = "images",
    masks_dir  = "masks",
    labels_dir = "labels",
    train_pct  = 0.70,
    val_pct    = 0.15,
    test_pct   = 0.15,
    seed       = 42,
    copy       = True,    # True -> copy files;  False -> move files
):
    """
    Randomly split paired (image, mask, label) samples into train/val/test.

    Final structure under root_dir:
        images/
            train/  val/  test/
        masks/
            train/  val/  test/
        labels/
            train/  val/  test/

    Only stems that have ALL THREE of (image, mask, label) are included.
    Set copy=False to move instead of copy (saves disk space).
    """

    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, \
        "train_pct + val_pct + test_pct must sum to exactly 1.0"

    root     = Path(root_dir)
    images_p = root / images_dir
    masks_p  = root / masks_dir
    labels_p = root / labels_dir

    img_stems   = set(_collect_stems(images_p, IMAGE_EXTS))
    mask_stems  = set(_collect_stems(masks_p,  MASK_EXTS))
    label_stems = {p.stem for p in labels_p.glob("*.txt")}

    stems = sorted(img_stems & mask_stems & label_stems)

    if not stems:
        print("[ERROR] No samples found with all three of: image, mask, label.")
        return

    random.seed(seed)
    random.shuffle(stems)

    n       = len(stems)
    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)
    n_test  = n - n_train - n_val   # remainder avoids rounding gaps

    splits = {
        "train": stems[:n_train],
        "val":   stems[n_train : n_train + n_val],
        "test":  stems[n_train + n_val :],
    }

    print(f"Total samples: {n}  ->  train={n_train}  val={n_val}  test={n_test}")

    transfer = shutil.copy2 if copy else shutil.move

    for split_name, split_stems in splits.items():

        (images_p / split_name).mkdir(parents=True, exist_ok=True)
        (masks_p  / split_name).mkdir(parents=True, exist_ok=True)
        (labels_p / split_name).mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # image
            src = _find_file(images_p, stem, IMAGE_EXTS)
            if src:
                transfer(src, images_p / split_name / src.name)

            # mask
            src = _find_file(masks_p, stem, MASK_EXTS)
            if src:
                transfer(src, masks_p / split_name / src.name)

            # label
            src = labels_p / f"{stem}.txt"
            if src.exists():
                transfer(src, labels_p / split_name / src.name)

        print(f"  [{split_name:5s}] {len(split_stems)} samples")

    print("\nSplit complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualise: image | mask | image + YOLO bbox
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sample(
    stem       = None,          # None -> pick randomly from images_dir
    images_dir = "images",      # point to split sub-folder if needed, e.g. "images/train"
    masks_dir  = "masks",
    labels_dir = "labels",
    bbox_color = "red",
    bbox_lw    = 2.0,
    figsize    = (15, 5),
    save_path  = None,          # None -> plt.show(); string path -> save figure
):
    """
    Plot three subplots for a single sample:
        1. Original image
        2. Binary mask
        3. Image with YOLO bounding boxes drawn on top

    images_dir / masks_dir / labels_dir can each point to a split sub-folder
    (e.g. "images/train") or the flat parent folder.
    """

    images_p = Path(images_dir)
    masks_p  = Path(masks_dir)
    labels_p = Path(labels_dir)

    # ── Pick a stem ──────────────────────────────────────────────────────────
    if stem is None:
        candidates = []
        for ext in IMAGE_EXTS:
            candidates += list(images_p.glob(f"*{ext}"))
            candidates += list(images_p.glob(f"*{ext.upper()}"))
        candidates = sorted(set(candidates))
        if not candidates:
            print(f"[ERROR] No images found in '{images_p}'.")
            return
        chosen = random.choice(candidates)
        stem   = chosen.stem
        print(f"[INFO] Random sample: '{stem}'")

    # ── Load image ───────────────────────────────────────────────────────────
    img_path = _find_file(images_p, stem, IMAGE_EXTS)
    if img_path is None:
        print(f"[ERROR] Image not found for stem '{stem}' in '{images_p}'.")
        return

    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"[ERROR] Cannot read '{img_path}'.")
        return

    if raw.ndim == 2:
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    # ── Load mask ────────────────────────────────────────────────────────────
    mask_path = _find_file(masks_p, stem, MASK_EXTS)
    if mask_path is None:
        print(f"[WARNING] Mask not found for '{stem}'. Showing blank.")
        mask_disp = np.zeros((H, W), dtype=np.uint8)
    else:
        mask_disp = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_disp is None:
            mask_disp = np.zeros((H, W), dtype=np.uint8)

    # ── Load YOLO label ──────────────────────────────────────────────────────
    label_path = labels_p / f"{stem}.txt"
    bboxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid, xc, yc, bw, bh = parts
                bboxes.append((int(cid), float(xc), float(yc),
                               float(bw), float(bh)))
    else:
        print(f"[WARNING] No label file at '{label_path}'.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"Sample: {stem}", fontsize=13, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_disp, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(img_rgb)
    axes[2].set_title(f"Image + YOLO BBoxes  ({len(bboxes)} box(es))")
    axes[2].axis("off")

    for cid, xc, yc, bw, bh in bboxes:
        x_px = (xc - bw / 2) * W
        y_px = (yc - bh / 2) * H
        w_px = bw * W
        h_px = bh * H

        rect = patches.Rectangle(
            (x_px, y_px), w_px, h_px,
            linewidth=bbox_lw, edgecolor=bbox_color, facecolor="none"
        )
        axes[2].add_patch(rect)
        # axes[2].text(x_px, max(y_px - 4, 0), f"cls {cid}", color=bbox_color, fontsize=8, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved -> '{save_path}'")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point – edit variables here, no argparse
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Shared paths ─────────────────────────────────────────────────────────
    ROOT_DIR   = "."        # root folder containing images/ and masks/
    IMAGES_DIR = "images"
    MASKS_DIR  = "masks"
    LABELS_DIR = "labels"

    # ── Step 1 config: mask -> YOLO label conversion ──────────────────────────
    CLASS_ID    = 0     # YOLO class index assigned to every bounding box
    MIN_AREA_PX = 10    # blobs smaller than this (px²) are discarded as noise

    # ── Step 2 config: train / val / test split ───────────────────────────────
    TRAIN_PCT  = 0.70
    VAL_PCT    = 0.15
    TEST_PCT   = 0.15
    SEED       = 42
    COPY_FILES = True   # False -> move files instead of copying them

    # ── Step 3 config: visualisation ─────────────────────────────────────────
    # After splitting, point these to the desired sub-folder, e.g.:
    #     VIS_IMAGES_DIR = "images/train"
    #     VIS_MASKS_DIR  = "masks/train"
    #     VIS_LABELS_DIR = "labels/train"
    VIS_STEM       = None           # None -> random sample
    VIS_IMAGES_DIR = "images/train"
    VIS_MASKS_DIR  = "masks/train"
    VIS_LABELS_DIR = "labels/train"
    BBOX_COLOR     = "red"
    BBOX_LW        = 2.0
    FIGSIZE        = (15, 5)
    SAVE_PATH      = "./samples_visualization.png"           # e.g. "preview.png";  None -> interactive


    # ── Run ───────────────────────────────────────────────────────────────────

    # # 1. Convert masks -> flat labels/
    # convert_masks_to_yolo(
    #     root_dir    = ROOT_DIR,
    #     images_dir  = IMAGES_DIR,
    #     masks_dir   = MASKS_DIR,
    #     labels_dir  = LABELS_DIR,
    #     class_id    = CLASS_ID,
    #     min_area_px = MIN_AREA_PX,
    # )
    #
    # # 2. Split -> images/train|val|test   masks/train|val|test   labels/train|val|test
    # split_dataset(
    #     root_dir   = ROOT_DIR,
    #     images_dir = IMAGES_DIR,
    #     masks_dir  = MASKS_DIR,
    #     labels_dir = LABELS_DIR,
    #     train_pct  = TRAIN_PCT,
    #     val_pct    = VAL_PCT,
    #     test_pct   = TEST_PCT,
    #     seed       = SEED,
    #     copy       = COPY_FILES,
    # )

    # 3. Visualise one sample from the split
    visualise_sample(
        stem       = VIS_STEM,
        images_dir = VIS_IMAGES_DIR,
        masks_dir  = VIS_MASKS_DIR,
        labels_dir = VIS_LABELS_DIR,
        bbox_color = BBOX_COLOR,
        bbox_lw    = BBOX_LW,
        figsize    = FIGSIZE,
        save_path  = SAVE_PATH,
    )