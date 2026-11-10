"""
FSR trial logger
================
One trial at a time, at your own pace. Works with fsr_stream.ino.

    pip install pyserial
    python fsr_trial_logger.py --port COM11 --name pla2mm_run1

For each trial:
  1. Type the mass in grams and press Enter      e.g.  100
     (optionally a custom hold time in seconds:   100 60)
  2. Logging starts at once. You have a 15 s buffer to place the weight
     (the plate must still be EMPTY when you press Enter; the first
     seconds give the zero reading for this trial).
  3. After the buffer, logging continues for the hold time (default 15 s).
  4. A beep says the trial is done. Remove the weight, then type the next mass.

Type q + Enter to finish.

All trials go into one file, fsr_runs/<name>.csv, with columns
    trial, mass_g, phase, pc_time, t_ms, adc_raw, mv, r_ohm
where phase is "buffer" (first 15 s) or "hold".
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import serial

try:
    import winsound
except ImportError:
    winsound = None


def beep(freq, ms):
    if winsound is not None:
        winsound.Beep(freq, ms)
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--name", required=True, help="run name, e.g. pla2mm_run1")
    ap.add_argument("--outdir", default="fsr_runs")
    ap.add_argument("--buffer", type=float, default=7.0, help="seconds to place the weight")
    ap.add_argument("--hold", type=float, default=15.0, help="default seconds logged after the buffer")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, a.name + ".csv")
    new = not os.path.exists(path)
    if not new:
        print(f"{path} exists - new trials will be APPENDED to it.")

    ser = serial.Serial(a.port, a.baud, timeout=1)
    time.sleep(2.0)

    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["trial", "mass_g", "phase", "pc_time", "t_ms", "adc_raw", "mv", "r_ohm"])

    # continue trial numbering if appending
    trial = 0
    if not new:
        with open(path) as fr:
            for row in csv.reader(fr):
                if row and row[0].isdigit():
                    trial = max(trial, int(row[0]))

    print(f"Logging trials to {path}")
    print(f"Plate EMPTY, type mass in g + Enter (optionally '100 60' for a 60 s hold). q to finish.\n")

    try:
        while True:
            cmd = input(f"Trial {trial + 1} - mass (g): ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            parts = cmd.split()
            try:
                mass = float(parts[0])
                hold = float(parts[1]) if len(parts) > 1 else a.hold
            except (ValueError, IndexError):
                print("  type a number, e.g. 100   or   100 60")
                continue
            trial += 1

            ser.reset_input_buffer()             # drop data from between trials
            t_start = time.time()
            t_buf_end = t_start + a.buffer
            t_end = t_buf_end + hold
            n = 0
            last_shown = None
            beeped_buffer = False
            while True:
                now = time.time()
                if now >= t_end:
                    break
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw or raw.startswith("#"):
                    continue
                p = raw.split(",")
                if len(p) != 4:
                    continue
                phase = "buffer" if now < t_buf_end else "hold"
                if phase == "hold" and not beeped_buffer:
                    beeped_buffer = True
                    beep(1200, 150)                 # buffer over: weight should be on
                w.writerow([trial, f"{mass:g}", phase,
                            datetime.now().isoformat(timespec="milliseconds"), *p])
                n += 1
                mv, r = float(p[2]), float(p[3])
                rtxt = f"{r/1000:7.2f} kOhm" if r > 0 else "  no force  "
                if phase == "buffer":
                    msg = f"PLACE {mass:g} g   ({t_buf_end - now:4.1f} s left to place)"
                else:
                    msg = f"HOLD              ({t_end - now:4.1f} s left)        "
                sys.stdout.write(f"\r  {msg} | {mv:7.1f} mV {rtxt}   ")
                sys.stdout.flush()
            f.flush()
            beep(800, 500)
            print(f"\n  Trial {trial} done ({n} samples). REMOVE the weight.\n")
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        ser.close()
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
