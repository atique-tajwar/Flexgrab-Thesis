"""
PAW3395 Characterization Logger (host side)
===========================================

Runs a single fixed-duration capture and writes it to CSV. Fixed duration
rather than a start/stop key, because these are controlled experiments where
the capture length should be identical across runs.

Usage:
    python paw_logger.py --port COM6 --name noise-paper --duration 60
    python paw_logger.py --port COM6 --name dist-100mm --duration 20 --cpi 16000
    python paw_logger.py --port COM6 --name zheight-2.4mm --duration 30

Naming convention (the analysis script parses these):
    noise-<surface>.csv     stationary noise floor,   e.g. noise-glass.csv
    surface-<material>.csv  swipe across a material,  e.g. surface-pet.csv
    zheight-<mm>mm.csv      standoff sweep,           e.g. zheight-2.4mm.csv
    dist-<mm>mm.csv         known-distance move,      e.g. dist-100mm.csv
    slip-<object>.csv       real slip event,          e.g. slip-bottle.csv

Output columns: t_us, dx, dy, motion

Requires: pip install pyserial
"""

import argparse
import csv
import os
import sys
import time

import serial


def sanitize(name):
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "")
    return name.strip() or "unnamed"


def main():
    ap = argparse.ArgumentParser(description="PAW3395 capture")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM6 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--name", required=True, help="output name (without .csv)")
    ap.add_argument("--duration", type=float, default=60.0, help="capture seconds")
    ap.add_argument("--cpi", type=int, default=None, help="set CPI before capture")
    ap.add_argument("--poll", type=int, default=None, help="poll interval in us")
    ap.add_argument("--outdir", default="slip_data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, sanitize(args.name) + ".csv")
    if os.path.exists(path):
        sys.exit(f"Refusing to overwrite existing file: {path}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        sys.exit(f"Could not open {args.port}: {e}")

    time.sleep(2.0)                 # board reset settling
    ser.reset_input_buffer()

    def send(cmd):
        ser.write((cmd + "\n").encode())
        time.sleep(0.15)
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", "ignore").strip()
            if line:
                print("  " + line)

    if args.cpi is not None:
        send(f"c{args.cpi}")
    if args.poll is not None:
        send(f"p{args.poll}")
    send("i")

    print(f"\nCapturing {args.duration:.0f} s -> {path}")
    print("Hold the setup steady / perform the motion now.\n")

    rows = []
    ser.reset_input_buffer()
    send("s")

    t_start = time.time()
    last_report = t_start
    try:
        while time.time() - t_start < args.duration:
            line = ser.readline().decode("utf-8", "ignore").strip()
            if not line or line.startswith("#") or line.startswith("t_us"):
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                            int(parts[3]), int(parts[4]), int(parts[5])))
            except ValueError:
                continue

            now = time.time()
            if now - last_report >= 1.0:
                el = now - t_start
                print(f"\r  {el:5.1f}s / {args.duration:.0f}s   "
                      f"{len(rows)} samples   "
                      f"({len(rows)/max(el,1e-6):.0f}/s)", end="")
                last_report = now
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        ser.write(b"x\n")
        time.sleep(0.2)
        ser.close()

    if not rows:
        sys.exit("\nNo samples captured. Check baud rate and that the board is streaming.")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "dx", "dy", "motion", "squal", "shutter"])
        w.writerows(rows)

    dt = [rows[i+1][0] - rows[i][0] for i in range(len(rows)-1)]
    dt_mean = sum(dt) / len(dt) if dt else 0
    nz = sum(1 for r in rows if r[3])

    print(f"\n\nSaved {len(rows)} samples to {path}")
    print(f"  mean interval : {dt_mean:.1f} us  ({1e6/dt_mean:.0f} Hz)"
          if dt_mean else "  interval: n/a")
    print(f"  polls with motion: {nz} ({100*nz/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
