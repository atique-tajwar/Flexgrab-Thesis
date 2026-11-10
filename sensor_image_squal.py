"""
PAW3395 raw frame viewer with SQUAL overlay
===========================================
Shows the 36x36 image the sensor sees, with SQUAL, shutter and image
contrast in the corner, so you can see focus and SQUAL change together
as you move the sensor through the height rig.

Needs the paw3395_frame_squal firmware (921600 baud, 'F' command that
also returns SQUAL and shutter).

    pip install pyserial numpy opencv-python
    python sensor_image_squal.py --port COM6 --height 2.95 --surface paper

Keys in the image window:
    q        quit
    s        save frame (PNG with overlay, PNG without, raw CSV) to frames/
    n        toggle contrast stretching
    + / -    change the height label by 0.05 mm (label only; set the rig yourself)
Keep the sensor still while capturing.
"""
import argparse
import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import serial

SYNC = b"\xff\xff\xaa\x55"
N = 36


def overlay(big, lines):
    out = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    w = max(cv2.getTextSize(t, font, fs, th)[0][0] for t in lines) + 12
    h = 20 * len(lines) + 8
    box = out[0:h, 0:w].copy()
    cv2.rectangle(box, (0, 0), (w, h), (0, 0, 0), -1)
    out[0:h, 0:w] = cv2.addWeighted(box, 0.65, out[0:h, 0:w], 0.35, 0)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (6, 20 + 20 * i), font, fs, (0, 255, 255), th, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--scale", type=int, default=12)
    ap.add_argument("--height", type=float, default=None, help="current rig height in mm (label)")
    ap.add_argument("--surface", default="", help="surface name for labels and filenames")
    ap.add_argument("--outdir", default="frames")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)                       # ESP32 resets when the port opens
    ser.reset_input_buffer()
    ser.write(b"x\n")
    time.sleep(0.2)
    ser.reset_input_buffer()
    print("Connected. Keys: q quit, s save, n contrast, +/- height label")

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "frames_log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["file", "surface", "height_mm", "squal", "squal_avg10",
                      "shutter", "img_mean", "img_sd"])

    height = args.height
    stretch = True
    sq_hist = deque(maxlen=10)
    try:
        while True:
            ser.write(b"F\n")
            ser.read_until(SYNC)
            head = ser.read(3)
            raw = ser.read(N * N)
            if len(head) != 3 or len(raw) != N * N:
                continue
            squal = head[0]
            shutter = head[1] | (head[2] << 8)
            sq_hist.append(squal)
            sq_avg = float(np.mean(sq_hist))

            img = np.frombuffer(raw, dtype=np.uint8).reshape(N, N)
            show = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) if stretch else img
            big = cv2.resize(show, (N * args.scale, N * args.scale),
                             interpolation=cv2.INTER_NEAREST)
            lines = [f"SQUAL {squal:3d}  (avg10 {sq_avg:5.1f})",
                     f"shutter {shutter}",
                     f"image sd {img.std():5.1f}  mean {img.mean():5.1f}"]
            if height is not None or args.surface:
                z = f"z = {height:.2f} mm" if height is not None else ""
                lines.append(f"{z}  {args.surface}".strip())
            lines.append("contrast stretch ON" if stretch else "raw levels")
            annotated = overlay(big, lines)
            cv2.imshow("PAW3395 surface view", annotated)

            k = cv2.waitKey(10) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("n"):
                stretch = not stretch
            elif k in (ord("+"), ord("=")) and height is not None:
                height = round(height + 0.05, 2)
            elif k == ord("-") and height is not None:
                height = round(height - 0.05, 2)
            elif k == ord("s"):
                tag = []
                if args.surface:
                    tag.append(args.surface)
                if height is not None:
                    tag.append(f"z{height:.2f}mm")
                tag.append(time.strftime("%H%M%S"))
                stem = os.path.join(args.outdir, "frame_" + "_".join(tag))
                cv2.imwrite(stem + "_overlay.png", annotated)
                cv2.imwrite(stem + ".png", cv2.resize(img, (N * args.scale, N * args.scale),
                                                      interpolation=cv2.INTER_NEAREST))
                np.savetxt(stem + ".csv", img, fmt="%d", delimiter=",")
                log.writerow([os.path.basename(stem), args.surface,
                              "" if height is None else height, squal, round(sq_avg, 1),
                              shutter, round(float(img.mean()), 1), round(float(img.std()), 1)])
                logf.flush()
                print(f"saved {stem}  (SQUAL {squal}, avg {sq_avg:.1f})")
    finally:
        ser.close()
        logf.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
