"""Pre-label skull data on this computer and produce Roboflow-ready zips.

Usage: python prelabel_local.py INPUT [INPUT ...]
  INPUT can be a video file, a folder of images, or a .zip of images.
Each input produces <name>_prelabel.zip next to it (YOLO segmentation + data.yaml),
ready to send back to the doctor / upload to Roboflow.
"""
import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2

HERE = Path(__file__).parent
MODEL_PATH = HERE / "Skull_31_10_69.pt"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def label_lines(result, conf_min):
    if result.masks is None or len(result.masks) == 0:
        return []
    out = []
    for box, poly in zip(result.boxes, result.masks.xyn):
        if float(box.conf[0]) < conf_min or len(poly) < 3:
            continue
        out.append(f"{int(box.cls[0])} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
    return out


def iter_video(path, fps):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_target = -(-total // step) if total else 0
    idx = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield f"frame_{n:05d}", frame, n_target
            n += 1
        idx += 1
    cap.release()


def iter_images(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    for p in files:
        img = cv2.imread(str(p))
        if img is not None:
            yield p.stem, img, len(files)


def process(src: Path, model, args):
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in src.stem)[:50]
    out_zip = src.parent / f"{stem}_prelabel.zip"
    tmp = None
    if src.is_dir():
        frames = iter_images(src)
    elif src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        zipfile.ZipFile(src).extractall(tmp.name)
        frames = iter_images(tmp.name)
    elif src.suffix.lower() in VIDEO_EXT:
        frames = iter_video(src, args.fps)
    else:
        print(f"  skip (unsupported type): {src}")
        return

    print(f"\n=== {src.name} -> {out_zip.name}")
    t0 = time.time()
    n = n_lab = 0
    used = set()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img, total in frames:
            name = f"{stem}_{name}"
            while name in used:  # same file name in two sub-folders
                name += "_"
            used.add(name)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            zf.writestr(f"{name}.jpg", jpg.tobytes())
            res = model.predict(img, conf=0.1, imgsz=args.imgsz, retina_masks=True, verbose=False)[0]
            lines = label_lines(res, args.conf)
            if lines:
                zf.writestr(f"{name}.txt", "\n".join(lines))
                n_lab += 1
            n += 1
            el = time.time() - t0
            eta = f" | ~{el / n * (total - n) / 60:.1f} min left" if total else ""
            print(f"\r  {n}/{total or '?'} frames{eta}      ", end="", flush=True)
        names = model.names
        yaml = ["names:"] + [f"  {i}: {json.dumps(names[i])}" for i in sorted(names)]
        zf.writestr("data.yaml", "\n".join(yaml + [f"nc: {len(names)}"]) + "\n")
    if tmp:
        tmp.cleanup()
    print(f"\n  done: {n} frames, {n_lab} with masks, {(time.time() - t0) / 60:.1f} min")
    print(f"  output: {out_zip}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second taken from videos (default 1)")
    ap.add_argument("--conf", type=float, default=0.3, help="min confidence for a mask (default 0.3)")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (default 1280 = training size)")
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"Loading model {MODEL_PATH.name} ...")
    model = YOLO(str(MODEL_PATH))
    for src in args.inputs:
        if not src.exists():
            print(f"  not found: {src}")
            continue
        try:
            process(src, model, args)
        except Exception as e:
            print(f"\n  ERROR on {src}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
