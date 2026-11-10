"""
FSR session logger (with load / unload event markers)
=====================================================
Records the fsr_stream firmware continuously into ONE CSV per run, and lets
you mark when you put a weight on or take it off, by typing in this window.

    pip install pyserial
    python fsr_session_logger.py --port COM11 --name pla2mm_run1

While it runs (type, then press Enter):
    <number>   you just PLACED this mass on the plate, in grams (e.g. 100)
    0          you just REMOVED the weight (plate empty)
    h <number> start of a long HOLD with this mass (e.g. h 100)
    n <text>   free note (e.g. n puck re-centred)
    q          stop and save

The screen shows the live voltage, resistance, and seconds since your last
event, so you know when 5 s have passed and when to change weights.
Output: fsr_runs/<name>.csv with columns
    pc_time, t_ms, adc_raw, mv, r_ohm, event, mass_g
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
    import msvcrt                      # Windows: non-blocking keyboard
except ImportError:
    msvcrt = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--name", required=True, help="run name, e.g. pla2mm_run1")
    ap.add_argument("--outdir", default="fsr_runs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, args.name + ".csv")
    if os.path.exists(path):
        sys.exit(f"{path} already exists - choose another --name")

    ser = serial.Serial(args.port, args.baud, timeout=1)
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
    w.writerow(["pc_time", "t_ms", "adc_raw", "mv", "r_ohm", "event", "mass_g"])

    pending_event, pending_mass = "", ""
    last_event_t = time.time()
    last_event_txt = "start (plate empty?)"
    typed = ""
    rows = 0
    print(f"Logging to {path}")
    print("Type a mass in g + Enter when you PLACE it, 0 + Enter when you REMOVE it, "
          "'h 100' for a long hold, 'n text' for a note, 'q' to stop.\n")

    try:
        while True:
            # keyboard
            if msvcrt is not None:
                while msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        cmd = typed.strip()
                        typed = ""
                        if not cmd:
                            continue
                        if cmd.lower() == "q":
                            raise KeyboardInterrupt
                        if cmd.lower().startswith("n "):
                            pending_event, pending_mass = "note:" + cmd[2:].strip(), ""
                        elif cmd.lower().startswith("h "):
                            pending_event, pending_mass = "hold", cmd[2:].strip()
                        else:
                            try:
                                m = float(cmd)
                            except ValueError:
                                print(f"\n  ? not understood: {cmd!r}")
                                continue
                            pending_event = "unload" if m == 0 else "load"
                            pending_mass = "0" if m == 0 else f"{m:g}"
                        last_event_t = time.time()
                        last_event_txt = f"{pending_event} {pending_mass}".strip()
                        print(f"\n  >> {last_event_txt}")
                    elif ch == "\b":
                        typed = typed[:-1]
                    else:
                        typed += ch

            # serial
            try:
                raw = lines.get(timeout=0.02)
            except queue.Empty:
                continue
            if raw.startswith("#"):
                print("\n" + raw)
                continue
            parts = raw.split(",")
            if len(parts) != 4:
                continue
            w.writerow([datetime.now().isoformat(timespec="milliseconds"), *parts,
                        pending_event, pending_mass])
            pending_event, pending_mass = "", ""
            rows += 1
            if rows % 10 == 0:
                f.flush()
            mv, r = float(parts[2]), float(parts[3])
            rtxt = f"{r/1000:8.2f} kOhm" if r > 0 else "   no force "
            dt = time.time() - last_event_t
            sys.stdout.write(f"\r{mv:7.1f} mV  {rtxt}   {dt:5.1f} s since '{last_event_txt}'"
                             f"   > {typed:<12}")
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
