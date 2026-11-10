"""
FSR guided logger (hands-free)
==============================
Runs the whole loading test on a timed schedule and TELLS you what to do,
with beeps and a big on-screen prompt. You never type during the test:
just place / remove the weight when it beeps. Works with fsr_stream.ino.

Prepare one container per mass in advance (e.g. four bottles filled to
50, 100, 200, 500 g) so you only need to set them down and lift them off.
Weigh each container and pass the exact masses with --masses.

    pip install pyserial
    python fsr_guided_logger.py --port COM11 --name pla2mm_run1 ^
        --masses 50.2 99.8 200.4 500.1 --cycles 5

Default schedule:
    10 s empty baseline
    each cycle: masses up then down (50,100,200,500,500,200,100,50)
        3-2-1 countdown -> PLACE  -> 10 s loaded
        3-2-1 countdown -> REMOVE -> 15 s empty (time to grab the next one)
    final hold: PLACE --hold-mass for --hold seconds, then REMOVE

Keys during the run (no Enter needed):
    space   pause / resume the schedule (if you fumble a weight)
    q       stop early and save

The log marks each PLACE / REMOVE prompt. The analysis also finds the
real moment of loading from the data, so a second of reaction delay
does not matter.

Output: fsr_runs/<name>.csv
    pc_time, t_ms, adc_raw, mv, r_ohm, event, mass_g, step
"""
import argparse
import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime

import serial

try:
    import msvcrt
    import winsound
except ImportError:            # not Windows: run without keys / sound
    msvcrt = None
    winsound = None


def beep(freq, ms):
    if winsound is not None:
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    else:
        sys.stdout.write("\a")


def build_schedule(a):
    """List of (action, mass, duration_after) in order."""
    steps = [("baseline", 0.0, a.baseline)]
    order = list(a.masses) + list(reversed(a.masses))
    for c in range(a.cycles):
        for m in order:
            steps.append(("place", m, a.on))
            steps.append(("remove", 0.0, a.off))
    if a.hold > 0:
        steps.append(("place", a.hold_mass, a.hold))
        steps.append(("remove", 0.0, a.off))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default="fsr_runs")
    ap.add_argument("--masses", type=float, nargs="+", default=[50, 100, 200, 500],
                    help="exact container masses in g, lightest first")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--on", type=float, default=10.0, help="seconds loaded")
    ap.add_argument("--off", type=float, default=15.0, help="seconds empty between loads")
    ap.add_argument("--baseline", type=float, default=10.0)
    ap.add_argument("--hold-mass", type=float, default=None,
                    help="mass for the final long hold (default: 2nd lightest)")
    ap.add_argument("--hold", type=float, default=60.0, help="final hold seconds (0 = skip)")
    ap.add_argument("--countdown", type=int, default=3)
    a = ap.parse_args()
    a.masses = sorted(a.masses)
    if a.hold_mass is None:
        a.hold_mass = a.masses[1] if len(a.masses) > 1 else a.masses[0]

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, a.name + ".csv")
    if os.path.exists(path):
        sys.exit(f"{path} already exists - choose another --name")

    steps = build_schedule(a)
    total = sum(d + (a.countdown if s != "baseline" else 0) for s, _, d in steps)
    print(f"Schedule: {len(steps)} steps, about {total/60:.1f} min. Masses {a.masses} g, "
          f"{a.cycles} cycles, hold {a.hold_mass:g} g for {a.hold:g} s.")
    print("Put the EMPTY plate on the sensor. Starting in 5 s ...  (space = pause, q = stop)")

    ser = serial.Serial(a.port, a.baud, timeout=1)
    time.sleep(2.0)
    ser.reset_input_buffer()
    lines = queue.Queue()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if raw:
                lines.put(raw)

    threading.Thread(target=reader, daemon=True).start()

    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["pc_time", "t_ms", "adc_raw", "mv", "r_ohm", "event", "mass_g", "step"])
    for line in [f"# masses_g={a.masses} cycles={a.cycles} on_s={a.on} off_s={a.off} "
                 f"hold_mass_g={a.hold_mass} hold_s={a.hold}"]:
        w.writerow([line, "", "", "", "", "", "", ""])

    # schedule state
    idx = -1
    phase = "lead"            # lead -> countdown -> act -> wait
    phase_end = time.time() + 5.0
    paused_at = None
    last_tick = None
    pending = ("", "", "")
    status = "get ready"
    rows = 0

    def advance(now):
        nonlocal idx, phase, phase_end, status, pending, last_tick
        if phase in ("lead", "wait"):
            idx += 1
            if idx >= len(steps):
                raise StopIteration
            action, m, dur = steps[idx]
            if action == "baseline":
                pending = ("baseline", "0", str(idx))
                phase, phase_end = "wait", now + dur
                status = "BASELINE - plate empty, hands off"
                beep(600, 300)
            else:
                phase, phase_end = "countdown", now + a.countdown
                last_tick = None
                status = (f"get ready: PLACE {m:g} g" if action == "place"
                          else "get ready: REMOVE")
        elif phase == "countdown":
            action, m, dur = steps[idx]
            if action == "place":
                pending = ("load", f"{m:g}", str(idx))
                status = f">>> PLACE {m:g} g NOW <<<   (hold {dur:g} s)"
                beep(1500, 400)
            else:
                pending = ("unload", "0", str(idx))
                status = f">>> REMOVE NOW <<<   (empty {dur:g} s)"
                beep(900, 400)
            phase, phase_end = "wait", now + dur

    try:
        while True:
            now = time.time()
            # keys
            if msvcrt is not None and msvcrt.kbhit():
                k = msvcrt.getwch().lower()
                if k == "q":
                    break
                if k == " ":
                    if paused_at is None:
                        paused_at = now
                        beep(400, 600)
                    else:
                        phase_end += now - paused_at
                        paused_at = None
            # schedule
            if paused_at is None:
                if phase == "countdown":
                    left = int(phase_end - now) + 1
                    if left != last_tick and left > 0:
                        last_tick = left
                        beep(1000, 120)
                if now >= phase_end:
                    try:
                        advance(now)
                    except StopIteration:
                        beep(1200, 800)
                        print("\n\nSchedule finished.")
                        break
            # data
            try:
                raw = lines.get(timeout=0.02)
            except queue.Empty:
                continue
            if raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) != 4:
                continue
            ev, m, st = pending
            w.writerow([datetime.now().isoformat(timespec="milliseconds"), *parts, ev, m, st])
            pending = ("", "", "")
            rows += 1
            if rows % 20 == 0:
                f.flush()
            mv, r = float(parts[2]), float(parts[3])
            rtxt = f"{r/1000:7.2f} kOhm" if r > 0 else "  no force  "
            remaining = max(0.0, phase_end - time.time())
            shown = "PAUSED (space to resume)" if paused_at else status
            sys.stdout.write(f"\r[{max(idx,0)+1:3d}/{len(steps)}] {shown:<44} {remaining:4.1f}s"
                             f" | {mv:7.1f} mV {rtxt}   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        f.close()
        ser.close()
        print(f"\nSaved {rows} rows to {path}")


if __name__ == "__main__":
    main()
