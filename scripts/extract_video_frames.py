#!/usr/bin/env python3
"""Costruisce un test set di sole immagini estraendo frame a frequenza fissa
(default 2 al secondo, VIDEO_TESTSET_FPS) dai video di una cartella.

I video sono materiale mai visto in training, quindi il test set è
indipendente dai dataset Roboflow: le immagini sono salvate senza label in
<out-dir>/images/, con nome <video>_f<indice-frame>_t<secondi>.jpg.

    python3 scripts/extract_video_frames.py /mnt/x/media

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.
"""
import argparse
import shutil
from pathlib import Path

import cv2

import config

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
JPEG_QUALITY = 95


def extract_frames(video: Path, out_images: Path, fps_out: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"impossibile aprire {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise SystemExit(f"{video}: FPS non valido ({fps})")
    step = fps / fps_out  # in frame sorgente tra due estrazioni (non intero in generale)
    saved = 0
    idx = 0
    next_frame = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= round(next_frame):
            name = f"{video.stem}_f{idx:06d}_t{idx / fps:07.2f}.jpg"
            cv2.imwrite(str(out_images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1
            next_frame = saved * step
        idx += 1
    cap.release()
    print(f"{video.name}: {idx}/{total} frame letti a {fps:.2f} fps, {saved} estratti")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path, help="cartella con i video")
    ap.add_argument("--out-dir", type=Path, default=config.VIDEO_TESTSET_DIR)
    ap.add_argument("--fps", type=float, default=config.VIDEO_TESTSET_FPS, help="frame estratti per secondo")
    args = ap.parse_args()

    videos = sorted(p for p in args.video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"nessun video in {args.video_dir}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    out_images = args.out_dir / "images"
    out_images.mkdir(parents=True)

    total = sum(extract_frames(v, out_images, args.fps) for v in videos)
    print(f"totale: {total} immagini in {out_images}")


if __name__ == "__main__":
    main()
