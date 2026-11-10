"""
PAW3395 Z-Height Tool
=====================

Finds the sensor's usable standoff (focal) window using the SQUAL value from
the burst read. SQUAL counts the surface features the sensor can resolve, so
it peaks at correct focus and collapses when the sensor is too close or too
far. That makes it a direct focus indicator -- far better than guessing from
whether tracking "looks right".

Three modes:

  1. LIVE -- watch SQUAL update in the terminal while you move the sensor by
     hand. Use this first to find the rough peak.

         python paw_zheight.py --port COM6 --live

  2. MEASURE -- with a shim stack of known thickness under the sensor, record
     a few seconds and append the result to zheight_results.csv. Repeat for
     each height.

         python paw_zheight.py --port COM6 --measure 2.0
         python paw_zheight.py --port COM6 --measure 2.2
         python paw_zheight.py --port COM6 --measure 2.4
         ...

  3. PLOT -- once several heights are recorded, plot SQUAL vs standoff and
     report the usable window.

         python paw_zheight.py --plot

Requires firmware emitting: t_us,dx,dy,motion,squal,shutter
Requires: numpy, matplotlib, pyserial
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

RESULTS = "zheight_results.csv"


def open_port(port, baud, poll):
    import serial
    ser = serial.Serial(port, baud, timeout=0.2)
    time.sleep(2.0)
    ser.reset_input_buffer()

    def cmd(c):
        ser.write((c + "\n").encode())
        time.sleep(0.15)
        while ser.in_waiting:
            ln = ser.readline().decode("utf-8", "ignore").strip()
            if ln:
                print("  " + ln)

    if poll:
        cmd(f"p{poll}")
    cmd("s")
    ser.reset_input_buffer()
    return ser


def parse(line):
    """Return (squal, shutter, dx, dy) or None."""
    if not line or line.startswith("#") or line.startswith("t_us"):
        return None
    p = line.split(",")
    if len(p) < 6:
        return None
    try:
        return int(p[4]), int(p[5]), int(p[1]), int(p[2])
    except ValueError:
        return None


def bar(v, vmax=128, width=44):
    n = int(round(width * min(v, vmax) / vmax))
    return "#" * n + "." * (width - n)


def mode_live(ser):
    print("\nLIVE MODE -- move the sensor up and down slowly.")
    print("Watch for the height where SQUAL peaks. Ctrl-C to stop.\n")
    peak = 0
    peak_shutter = 0
    hist = []
    last = 0.0
    try:
        while True:
            rec = parse(ser.readline().decode("utf-8", "ignore").strip())
            if rec is None:
                continue
            squal, shutter, dx, dy = rec
            hist.append(squal)
            if len(hist) > 20:
                hist.pop(0)
            sm = float(np.mean(hist))
            if sm > peak:
                peak, peak_shutter = sm, shutter

            now = time.time()
            if now - last > 0.08:
                print(f"\rSQUAL {sm:6.1f} |{bar(sm)}|  "
                      f"shutter {shutter:6d}  peak {peak:6.1f}   ", end="")
                last = now
    except KeyboardInterrupt:
        print(f"\n\nBest SQUAL seen: {peak:.1f} (shutter {peak_shutter})")
        print("Now measure at known heights with --measure <mm>.")


def mode_measure(ser, height_mm, seconds):
    print(f"\nMEASURING at {height_mm} mm for {seconds:.0f} s.")
    print("Keep the sensor still on the shim stack.\n")
    squals, shutters, moved = [], [], 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        rec = parse(ser.readline().decode("utf-8", "ignore").strip())
        if rec is None:
            continue
        squal, shutter, dx, dy = rec
        squals.append(squal)
        shutters.append(shutter)
        if dx or dy:
            moved += 1
        el = time.time() - t0
        print(f"\r  {el:4.1f}s  n={len(squals):5d}  "
              f"SQUAL {np.mean(squals):6.1f}   ", end="")

    if not squals:
        sys.exit("\nNo samples. Is the firmware streaming the 6-column format?")

    row = dict(
        height_mm=height_mm,
        squal_mean=float(np.mean(squals)),
        squal_sd=float(np.std(squals)),
        squal_min=int(np.min(squals)),
        squal_max=int(np.max(squals)),
        shutter_mean=float(np.mean(shutters)),
        n=len(squals),
        motion_pct=100.0 * moved / len(squals),
    )

    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"\n\n  SQUAL    {row['squal_mean']:.1f} +/- {row['squal_sd']:.1f} "
          f"(min {row['squal_min']}, max {row['squal_max']})")
    print(f"  shutter  {row['shutter_mean']:.0f}")
    print(f"  appended to {RESULTS}")


def mode_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} not found. Run --measure at several heights first.")

    rows = []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    if len(rows) < 3:
        sys.exit(f"Only {len(rows)} height(s) recorded. Measure at least 3.")

    rows.sort(key=lambda r: r["height_mm"])
    z = np.array([r["height_mm"] for r in rows])
    s = np.array([r["squal_mean"] for r in rows])
    sd = np.array([r["squal_sd"] for r in rows])
    sh = np.array([r["shutter_mean"] for r in rows])

    best = s.max()
    good = z[s >= 0.8 * best]

    print(f"\n{'Z (mm)':>8} | {'SQUAL':>8} | {'sd':>6} | {'shutter':>9}")
    print("-" * 40)
    for zi, si, sdi, shi in zip(z, s, sd, sh):
        mark = "  <-- peak" if si == best else ""
        print(f"{zi:>8.2f} | {si:>8.1f} | {sdi:>6.1f} | {shi:>9.0f}{mark}")

    print(f"\nPeak SQUAL {best:.1f} at Z = {z[s.argmax()]:.2f} mm")
    if len(good):
        print(f"Usable window (SQUAL >= 80% of peak): "
              f"{good.min():.2f} - {good.max():.2f} mm  "
              f"(width {good.max()-good.min():.2f} mm)")
        print("\nDesign implication: the finger must hold the sensor within this")
        print("window relative to the gripped surface, and hold it there while")
        print("gripping. A narrow window means a rigid standoff is required --")
        print("a compliant fingertip deforming under load would drift out of it.")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(z, s, yerr=sd, fmt="o-", color="#1f6feb", lw=2, ms=7,
                 capsize=3, label="SQUAL")
    ax1.axhline(0.8 * best, color="#d1242f", ls=":", lw=1.2,
                label="80% of peak")
    if len(good):
        ax1.axvspan(good.min(), good.max(), color="#1f6feb", alpha=0.10)
    ax1.axvspan(2.2, 2.6, color="#1a7f37", alpha=0.10,
                label="datasheet 2.2-2.6 mm")
    ax1.set_xlabel("standoff Z (mm)")
    ax1.set_ylabel("SQUAL (surface features)", color="#1f6feb")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(z, sh, "s--", color="#8250df", lw=1.2, ms=5, alpha=0.7)
    ax2.set_ylabel("shutter (exposure)", color="#8250df")

    ax1.legend(fontsize=8, loc="lower center")
    ax1.set_title("Sensor focal window from SQUAL")
    fig.tight_layout()
    fig.savefig("zheight_squal.png", dpi=150)
    print("\nsaved zheight_squal.png")


def main():
    ap = argparse.ArgumentParser(description="PAW3395 Z-height / focus tool")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--poll", type=int, default=5000)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--measure", type=float, metavar="MM",
                    help="record at this standoff in mm")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.plot:
        mode_plot()
        return

    if not args.port:
        sys.exit("Specify --port (or use --plot).")
    if not args.live and args.measure is None:
        sys.exit("Choose --live, --measure <mm>, or --plot.")

    try:
        ser = open_port(args.port, args.baud, args.poll)
    except Exception as e:
        sys.exit(f"Could not open {args.port}: {e}")

    try:
        if args.live:
            mode_live(ser)
        else:
            mode_measure(ser, args.measure, args.seconds)
    finally:
        try:
            ser.write(b"x\n")
            time.sleep(0.1)
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
