"""
main_with_static_trigger.py
-------
Pipeline orchestrator — Phase 1 & 1.5.
Termasuk HUD Overlay dan Logika Static Trigger.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

import config
from detector import VehicleDetector
from visualizer import draw_detections, draw_fps

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart City Road Safety — Phase 1 & 1.5")
    parser.add_argument("--video", type=str, default=config.DEFAULT_VIDEO_PATH,
                        help=f"Path to input video. Default: {config.DEFAULT_VIDEO_PATH}")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single input image.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without opening a display window.")
    parser.add_argument("--save", action="store_true",
                        help="Save the annotated video/image to output path.")
    parser.add_argument("--calibrated", action="store_true",
                        help="Enable future calibrated mode. Currently falls back to basic trigger.")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Input I/O helpers
# ---------------------------------------------------------------------------
def _is_image_path(source: str) -> bool:
    return Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def _open_video(source: str) -> cv2.VideoCapture:
    try:
        index = int(source)
        cap = cv2.VideoCapture(index)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)
    return cap

def _open_image(source: str) -> tuple[np.ndarray, str]:
    image = cv2.imread(source)
    if image is None:
        print(f"[ERROR] Could not open image source: {source}")
        sys.exit(1)
    return image, source

def _create_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer if writer.isOpened() else None

# ---------------------------------------------------------------------------
# MODUL HUD & TRIGGER LOGIC (Membuat kode lebih rapi)
# ---------------------------------------------------------------------------
def apply_hud_and_trigger(frame, detections, args, width, height):
    """Menambahkan garis trigger dan HUD Overlay ke dalam frame."""
    if args.calibrated:
        return frame # Placeholder untuk masa depan saat menggunakan model matematika
        
    TRIGGER_Y = int(height * 0.7)
    warning_active = False
    kendaraan_saat_ini = {}

    # 1. Ekstraksi Data
    for box in detections:
        try:
            if hasattr(box, 'bbox'):
                y_max = int(box.bbox[3])
                jenis = getattr(box, 'class_name', 'Kendaraan').capitalize()
            else:
                y_max = int(box[3])
                jenis = "Kendaraan"

            kendaraan_saat_ini[jenis] = kendaraan_saat_ini.get(jenis, 0) + 1
            if y_max >= TRIGGER_Y:
                warning_active = True
        except Exception:
            pass

    # 2. Visualisasi Garis Trigger
    garis_warna = (0, 0, 255) if warning_active else (0, 255, 0)
    cv2.line(frame, (0, TRIGGER_Y), (width, TRIGGER_Y), garis_warna, 3)
    cv2.putText(frame, "TRIGGER LINE", (10, TRIGGER_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, garis_warna, 2)

    # 3. Visualisasi HUD (Pojok Kiri Atas)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status_teks = "SINYAL: MENYALA (BAHAYA)" if warning_active else "SINYAL: MATI (AMAN)"
    status_warna = (0, 0, 255) if warning_active else (0, 255, 0)
    cv2.putText(frame, status_teks, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_warna, 2, cv2.LINE_AA)

    cv2.putText(frame, "Kendaraan Terdeteksi:", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    y_teks = 105
    if not kendaraan_saat_ini:
        cv2.putText(frame, "- Tidak ada", (20, y_teks), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        for jenis, jumlah in kendaraan_saat_ini.items():
            teks_counter = f"- {jenis} : {jumlah}"
            cv2.putText(frame, teks_counter, (20, y_teks), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y_teks += 25

    return frame

# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    detector = VehicleDetector()

    image_path = args.image
    if image_path is None and _is_image_path(args.video):
        image_path = args.video

    # ==========================================
    # MODE 1: PROSES GAMBAR TUNGGAL (IMAGE)
    # ==========================================
    if image_path is not None:
        frame, source_label = _open_image(image_path)
        height, width = frame.shape[:2]
        
        print(f"[Main] Source : {source_label}")
        print(f"[Main] Resolution : {width}×{height}  |  Mode: image")

        detections = detector.detect(frame)
        
        # Panggil fungsi HUD yang baru
        frame = apply_hud_and_trigger(frame, detections, args, width, height)
        draw_detections(frame, detections)

        if not args.no_display:
            cv2.imshow("Smart City Road Safety — HUD", frame)
            cv2.waitKey(0)

        if args.save and config.OUTPUT_VIDEO_PATH:
            output_path = Path(config.OUTPUT_VIDEO_PATH)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path.with_suffix(".png")), frame)
            print(f"[Main] Saved image to: {output_path.with_suffix('.png')}")

        cv2.destroyAllWindows()
        print("[Main] Done.")
        return

    # ==========================================
    # MODE 2: PROSES VIDEO
    # ==========================================
    cap = _open_video(args.video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[Main] Source : {args.video}")
    print(f"[Main] Resolution : {width}×{height}  |  FPS: {src_fps:.1f}")

    writer = _create_writer(cap, config.OUTPUT_VIDEO_PATH) if (args.save and config.OUTPUT_VIDEO_PATH) else None
    
    fps_display = 0.0
    frame_count = 0
    t_start = time.perf_counter()

    print("[Main] Starting pipeline. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video stream.")
            break

        frame_count += 1
        detections = detector.detect(frame)

        # Panggil fungsi HUD yang baru
        frame = apply_hud_and_trigger(frame, detections, args, width, height)
        
        draw_detections(frame, detections)
        draw_fps(frame, fps_display)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Smart City Road Safety — HUD", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("[Main] User requested exit.")
                break

        elapsed = time.perf_counter() - t_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t_start = time.perf_counter()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")

if __name__ == "__main__":
    run(_parse_args())