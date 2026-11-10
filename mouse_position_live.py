"""
PAW3395 Live Monitor (v2)
=========================

Live sanity check: streams dx/dy, accumulates position, plots the path in
MILLIMETRES, prints running totals.

Why v2: v1 read one serial line per plot iteration. The sensor streams at
500 Hz while a matplotlib redraw takes tens of milliseconds, so the serial
buffer grew without bound and the display fell progressively further behind.
Here a background thread drains the port continuously and the plot only ever
reads accumulated state, so rendering speed cannot cause lag.

Also changed:
  - default poll interval 5 ms (200 Hz) -- ample for a visual check, and a
    fifth of the data rate
  - plotted trace is decimated to a fixed budget, so redraw cost stays
    constant no matter how long it runs
  - serial buffer is flushed on reset and on CPI change, so stale samples
    are not mixed with a new scale factor
  - live backlog indicator; if it climbs, the pipeline is not keeping up
  - --flip-x / --flip-y, because Motion_Burst() negates dx
    (*dx = -(int16_t)(...)), which mirrors the trace horizontally

Usage:
    python paw_monitor.py --demo
    python paw_monitor.py --port COM6 --cpi 16000
    python paw_monitor.py --port COM6 --cpi 16000 --flip-x

Keys (plot window focused):
    r   reset origin and clear trace
    1-6 CPI presets: 800 1600 5000 8000 16000 26000
    s   save PNG
    q   quit

Requires: numpy, matplotlib, pyserial
"""

import argparse
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
CPI_PRESETS = {"1": 800, "2": 1600, "3": 5000, "4": 8000, "5": 16000, "6": 26000}
MAX_TRACE_PTS = 4000          # plotted points budget


class Accumulator:
    """Shared state between the reader thread and the plot loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.x_cnt = 0.0        # cumulative counts
            self.y_cnt = 0.0
            self.path_cnt = 0.0
            self.n = 0
            self.n_motion = 0
            self.recent = deque(maxlen=400)
            self.trace = [(0.0, 0.0)]
            self.decim = 1
            self.since_keep = 0
            self.t_first = None
            self.t_last = None

    def add(self, t_us, dx, dy):
        with self.lock:
            self.x_cnt += dx
            self.y_cnt += dy
            self.path_cnt += float(np.hypot(dx, dy))
            self.n += 1
            if dx or dy:
                self.n_motion += 1
            self.recent.append(float(np.hypot(dx, dy)))
            if self.t_first is None:
                self.t_first = t_us
            self.t_last = t_us

            self.since_keep += 1
            if self.since_keep >= self.decim:
                self.since_keep = 0
                self.trace.append((self.x_cnt, self.y_cnt))
                if len(self.trace) > MAX_TRACE_PTS:
                    self.trace = self.trace[::2]
                    self.decim *= 2

    def snapshot(self):
        with self.lock:
            rms = float(np.sqrt(np.mean(np.square(self.recent)))) if self.recent else 0.0
            return dict(
                x=self.x_cnt, y=self.y_cnt, path=self.path_cnt,
                n=self.n, n_motion=self.n_motion, rms=rms,
                trace=list(self.trace),
            )


class SerialReader(threading.Thread):
    def __init__(self, ser, acc, flip_x, flip_y):
        super().__init__(daemon=True)
        self.ser = ser
        self.acc = acc
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.running = True
        self.backlog = 0
        self.rate = 0.0
        self._n = 0
        self._t0 = time.time()

    def run(self):
        buf = b""
        while self.running:
            try:
                waiting = self.ser.in_waiting
                self.backlog = waiting
                chunk = self.ser.read(max(waiting, 1))
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or line.startswith("t_us"):
                        continue
                    if line.startswith("#"):
                        print("\n  " + line)
                        continue
                    p = line.split(",")
                    if len(p) < 4:
                        continue
                    try:
                        t_us, dx, dy = int(p[0]), int(p[1]), int(p[2])
                    except ValueError:
                        continue
                    if self.flip_x:
                        dx = -dx
                    if self.flip_y:
                        dy = -dy
                    self.acc.add(t_us, dx, dy)
                    self._n += 1
                now = time.time()
                if now - self._t0 >= 1.0:
                    self.rate = self._n / (now - self._t0)
                    self._n = 0
                    self._t0 = now
            except Exception:
                time.sleep(0.01)

    def stop(self):
        self.running = False


class DemoReader(threading.Thread):
    def __init__(self, acc, get_cpi):
        super().__init__(daemon=True)
        self.acc = acc
        self.get_cpi = get_cpi
        self.running = True
        self.backlog = 0
        self.rate = 200.0
        self.i = 0
        self.t = 0

    def run(self):
        while self.running:
            cpi = self.get_cpi()
            cpm = cpi / MM_PER_INCH
            per = 60.0 * cpm / 200.0
            side = (self.i // 200) % 4
            step = int(round(per + np.random.normal(0, per * 0.05)))
            dx, dy = [(step, 0), (0, step), (-step, 0), (0, -step)][side]
            self.t += 5000
            self.acc.add(self.t, dx, dy)
            self.i += 1
            time.sleep(0.005)

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser(description="PAW3395 live monitor")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--cpi", type=int, default=16000)
    ap.add_argument("--poll", type=int, default=5000,
                    help="poll interval in us (default 5000 = 200 Hz)")
    ap.add_argument("--flip-x", action="store_true",
                    help="negate dx (Motion_Burst already negates it once)")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if not args.demo and not args.port:
        sys.exit("Specify --port, or use --demo to run without hardware.")

    state = dict(cpi=args.cpi, quit=False)
    acc = Accumulator()
    ser = None

    def send(cmd, flush=False):
        if ser is None:
            return
        ser.write((cmd + "\n").encode())
        time.sleep(0.12)
        if flush:
            ser.reset_input_buffer()

    if args.demo:
        reader = DemoReader(acc, lambda: state["cpi"])
        print("DEMO MODE -- synthetic data, no hardware.")
    else:
        try:
            import serial
            ser = serial.Serial(args.port, args.baud, timeout=0.05)
        except Exception as e:
            sys.exit(f"Could not open {args.port}: {e}")
        time.sleep(2.0)
        ser.reset_input_buffer()
        send(f"c{state['cpi']}")
        send(f"p{args.poll}")
        send("i")
        ser.reset_input_buffer()
        send("s")
        reader = SerialReader(ser, acc, args.flip_x, args.flip_y)

    reader.start()

    plt.ion()
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12, 5.8),
                                  gridspec_kw={"width_ratios": [3, 2]})
    line, = ax.plot([], [], "-", color="#1f6feb", lw=1.4)
    dot, = ax.plot([], [], "o", color="#d1242f", ms=7)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    axt.axis("off")
    txt = axt.text(0.0, 0.99, "", va="top", ha="left", family="monospace",
                   fontsize=10, transform=axt.transAxes)

    def do_reset():
        acc.reset()
        if ser is not None:
            ser.reset_input_buffer()
        print("\n-- origin reset --")

    def on_key(event):
        k = event.key
        if k == "q":
            state["quit"] = True
        elif k == "r":
            do_reset()
        elif k == "s":
            fn = f"paw_trace_{state['cpi']}cpi.png"
            fig.savefig(fn, dpi=150)
            print(f"\nsaved {fn}")
        elif k in CPI_PRESETS:
            state["cpi"] = CPI_PRESETS[k]
            if ser is not None:
                send("x", flush=True)
                send(f"c{state['cpi']}")
                send("s", flush=True)
            do_reset()
            print(f"-- CPI set to {state['cpi']} --")

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\nKeys: r reset | 1-6 CPI | s save | q quit")
    print("Verify CPI: press r, slide a known distance, read 'net displ'.\n")

    last_draw = last_print = 0.0
    try:
        while not state["quit"] and plt.fignum_exists(fig.number):
            now = time.time()
            if now - last_draw > 0.08:
                s = acc.snapshot()
                cpm = state["cpi"] / MM_PER_INCH
                tr = np.asarray(s["trace"], dtype=float) / cpm
                x_mm, y_mm = s["x"] / cpm, s["y"] / cpm
                net = float(np.hypot(x_mm, y_mm))
                count_um = MM_PER_INCH / state["cpi"] * 1000
                backlog = getattr(reader, "backlog", 0)

                if len(tr):
                    line.set_data(tr[:, 0], tr[:, 1])
                    dot.set_data([x_mm], [y_mm])
                    ax.relim(); ax.autoscale_view()
                ax.set_title(f"path @ {state['cpi']} CPI ({count_um:.2f} um/count)")

                warn = ""
                if backlog > 4000:
                    warn = "\n** backlog high: raise --poll"
                txt.set_text(
                    f"CPI          {state['cpi']}\n"
                    f"count size   {count_um:8.2f} um\n\n"
                    f"net displ    {net:8.2f} mm\n"
                    f"  x          {x_mm:8.2f} mm\n"
                    f"  y          {y_mm:8.2f} mm\n"
                    f"path length  {s['path']/cpm:8.2f} mm\n\n"
                    f"samples      {s['n']:8d}\n"
                    f"with motion  {s['n_motion']:8d} "
                    f"({100*s['n_motion']/max(s['n'],1):.1f}%)\n"
                    f"recent RMS   {s['rms']:8.2f} counts\n\n"
                    f"read rate    {getattr(reader,'rate',0.0):8.0f} /s\n"
                    f"backlog      {backlog:8d} bytes{warn}\n\n"
                    f"r reset  1-6 CPI  s save  q quit"
                )
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                last_draw = now

            if now - last_print > 1.0:
                s = acc.snapshot()
                cpm = state["cpi"] / MM_PER_INCH
                net = float(np.hypot(s["x"], s["y"])) / cpm
                print(f"\r  net {net:8.2f} mm | path {s['path']/cpm:8.2f} mm | "
                      f"{s['n']:7d} samples | {getattr(reader,'rate',0):4.0f}/s | "
                      f"backlog {getattr(reader,'backlog',0):5d}B   ", end="")
                last_print = now

            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        time.sleep(0.1)
        if ser is not None:
            try:
                ser.write(b"x\n"); time.sleep(0.1); ser.close()
            except Exception:
                pass
        s = acc.snapshot()
        cpm = state["cpi"] / MM_PER_INCH
        print("\n\nFinal:")
        print(f"  CPI          {state['cpi']}")
        print(f"  net displ    {float(np.hypot(s['x'], s['y']))/cpm:.2f} mm")
        print(f"  path length  {s['path']/cpm:.2f} mm")
        print(f"  samples      {s['n']}")
        if s["n"]:
            print(f"  motion polls {100*s['n_motion']/s['n']:.1f}%")
        plt.ioff()


if __name__ == "__main__":
    main()