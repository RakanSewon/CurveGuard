"""
main_calibration.py
-------
Pipeline orchestrator — Phase 1 (Calibration Mode).

Modifikasi:
1. Menampilkan nilai koordinat Y bagian bawah (Y-Bottom) pada tiap bounding box.
2. Menambahkan fitur "Pause" (tekan 'p') agar video berhenti sejenak untuk mempermudah pencatatan.
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
    parser = argparse.ArgumentParser(
        description="Smart City Road Safety — Phase 1: Calibration Mode"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=config.DEFAULT_VIDEO_PATH,
        help="Path to the input video file, or an integer device index for webcam.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single input image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening a display window.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated video.",
    )
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
        label = f"webcam {index}"
    except ValueError:
        cap = cv2.VideoCapture(source)
        label = source

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {label}")
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
    if not writer.isOpened():
        return None
    return writer


# ---------------------------------------------------------------------------
# FUNGSI TAMBAHAN UNTUK KALIBRASI
# ---------------------------------------------------------------------------
def draw_calibration_data(frame, detections):
    """Mengekstrak nilai Y-Max dari objek kelas kustom 'Detection'."""
    for box in detections:
        try:
            # Mengakses atribut .bbox dari objek kelas Detection
            if hasattr(box, 'bbox'):
                x1 = int(box.bbox[0])
                y_max = int(box.bbox[3]) # Indeks ke-3 adalah koordinat y2
            else:
                # Jika kebetulan formatnya berupa array standar
                x1 = int(box[0])
                y_max = int(box[3])
            
            # Cetak teks warna kuning persis di atas garis bawah bounding box
            teks = f"Y: {y_max}"
            cv2.putText(frame, teks, (x1, y_max - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Tambahkan titik merah sebagai penanda tepat di posisi ban
            cv2.circle(frame, (x1, y_max), 5, (0, 0, 255), -1)
            
        except Exception as e:
            print(f"[DEBUG] Gagal mengekstrak koordinat. Error: {e}")


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    detector = VehicleDetector()

    image_path = args.image
    if image_path is None and _is_image_path(args.video):
        image_path = args.video

    # --- Mode Gambar Tunggal ---
    if image_path is not None:
        frame, source_label = _open_image(image_path)
        print(f"[Main] Source : {source_label}")
        
        detections = detector.detect(frame)
        
        # Gambar kotak bawaan
        draw_detections(frame, detections)
        # Tambahkan teks Y-max untuk kalibrasi
        draw_calibration_data(frame, detections)

        if not args.no_display:
            cv2.imshow("Smart City Road Safety — Calibration", frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Mode Video ---
    cap = _open_video(args.video)
    fps_display  = 0.0
    frame_count  = 0
    t_start      = time.perf_counter()

    writer = _create_writer(cap, config.OUTPUT_VIDEO_PATH) if args.save and config.OUTPUT_VIDEO_PATH else None

    print("[Main] Starting Calibration Pipeline.")
    print("       TEKAN 'p' UNTUK PAUSE (agar mudah mencatat angka).")
    print("       TEKAN 'q' UNTUK KELUAR.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        detections = detector.detect(frame)
        
        draw_detections(frame, detections)
        draw_fps(frame, fps_display)
        
        # Tambahkan teks Y-max untuk kalibrasi
        draw_calibration_data(frame, detections)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Smart City Road Safety — Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Logika Pause
            if key == ord('p'):
                print(">>> VIDEO PAUSED. Catat angkanya. Tekan tombol apapun untuk lanjut...")
                cv2.waitKey(0) # Program berhenti di sini sampai ada tombol ditekan
                
            elif key in (ord("q"), 27):
                print("[Main] User requested exit.")
                break

        elapsed = time.perf_counter() - t_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t_start     = time.perf_counter()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")

if __name__ == "__main__":
    run(_parse_args())