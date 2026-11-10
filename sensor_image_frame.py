"""
PAW3395 raw frame viewer
========================
Shows the 36x36 image the sensor sees. Needs the paw3395_frame firmware
(921600 baud, 'F' command).

    pip install pyserial numpy opencv-python
    python sensor_image_frame.py --port COM6

Keys in the image window:
    q   quit
    s   save the current frame (PNG, 10x enlarged + raw CSV) to frames/
    n   toggle contrast stretching
Keep the sensor still while capturing.
"""
import argparse
import os
import time

import cv2
import numpy as np
import serial

SYNC = b"\xff\xff\xaa\x55"
N = 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--outdir", default="frames")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)                       # ESP32 resets when the port opens
    ser.reset_input_buffer()
    ser.write(b"x\n")                     # make sure logging is stopped
    time.sleep(0.2)
    ser.reset_input_buffer()
    print("Connected. Window keys: q quit, s save, n contrast on/off")

    stretch = True
    saved = 0
    os.makedirs(args.outdir, exist_ok=True)
    try:
        while True:
            ser.write(b"F\n")
            ser.read_until(SYNC)
            raw = ser.read(N * N)
            if len(raw) != N * N:
                continue
            img = np.frombuffer(raw, dtype=np.uint8).reshape(N, N)
            show = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) if stretch else img
            big = cv2.resize(show, (N * args.scale, N * args.scale),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow("PAW3395 surface view", big)
            print(f"\rmin {img.min():3d}  max {img.max():3d}  mean {img.mean():6.1f}  "
                  f"contrast(sd) {img.std():5.1f}   ", end="")
            k = cv2.waitKey(10) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n"):
                stretch = not stretch
            if k == ord("s"):
                stem = os.path.join(args.outdir, time.strftime("frame_%Y%m%d_%H%M%S"))
                cv2.imwrite(stem + ".png", cv2.resize(img, (N * args.scale, N * args.scale),
                                                      interpolation=cv2.INTER_NEAREST))
                np.savetxt(stem + ".csv", img, fmt="%d", delimiter=",")
                saved += 1
                print(f"\nsaved {stem}.png/.csv")
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print(f"\n{saved} frame(s) saved")


if __name__ == "__main__":
    main()
