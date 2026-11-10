"""
PAW3395 Slip Sensor Characterization -- Figure Generation
=========================================================

Produces the figures and numbers for the slip-sensor section of the Results
chapter, from the capture folder.

Expected files (as logged):
    noise-still-<cpi>.csv        stationary, 60 s
    dist-<mm>mm-<cpi>.csv        known lateral slide, 5 s

Columns: t_us,dx,dy,motion,squal,shutter

Usage:
    python slip_results.py <folder>
    python slip_results.py <folder> --outdir slip_figs --poll-us 5000

Outputs (into --outdir):
    FIG_S1_timing.png          sampling interval determinism
    FIG_S2_noise_vth.png       still-noise floor and v_th vs CPI
    FIG_S3_calibration.png     counts-per-mm calibration
    FIG_S4_velocity.png        velocity profiles and headroom
    FIG_S5_optical.png         SQUAL / shutter relationship
    slip_summary.csv           per-file extracted values

Requires: numpy, pandas, matplotlib
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
COLS = {1600: "#1f6feb", 3200: "#e3820e", 6400: "#1a7f37", 12800: "#d1242f"}


# ----------------------------------------------------------------------

def load(folder):
    still, dist = {}, {}
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        b = os.path.basename(p)
        d = pd.read_csv(p)
        if not {"t_us", "dx", "dy", "squal", "shutter"}.issubset(d.columns):
            continue
        m = re.match(r"^noise-still-(\d+)\.csv$", b)
        if m:
            still[int(m.group(1))] = d
            continue
        m = re.match(r"^dist-(\d+)mm-(\d+)\.csv$", b)
        if m:
            dist[(int(m.group(2)), int(m.group(1)))] = d
    return still, dist


def stats(d, cpi, poll_s):
    mag = np.hypot(d["dx"], d["dy"]).to_numpy()
    cs = MM_PER_INCH / cpi
    return dict(
        n=len(d),
        nonzero=int((mag > 0).sum()),
        max_cnt=float(mag.max()),
        rms_cnt=float(np.sqrt((mag ** 2).mean())),
        net_x=float(d["dx"].sum()),
        net_y=float(d["dy"].sum()),
        net=float(np.hypot(d["dx"].sum(), d["dy"].sum())),
        peak_v=float((mag * cs / poll_s).max()),
        squal=float(d["squal"].mean()),
        shutter=float(d["shutter"].mean()),
    )


# ----------------------------------------------------------------------

def fig_timing(still, dist, outdir, poll_us):
    alldt = []
    for d in list(still.values()) + list(dist.values()):
        alldt.append(np.diff(d["t_us"].to_numpy()))
    dt = np.concatenate(alldt)

    print("\n=== TIMING DETERMINISM ===")
    print(f"  intervals            {len(dt)}")
    print(f"  median               {np.median(dt):.1f} us (commanded {poll_us})")
    print(f"  std deviation        {dt.std():.2f} us ({100*dt.std()/np.median(dt):.3f}%)")
    print(f"  min / max            {dt.min()} / {dt.max()} us")
    print(f"  outside +/-1%        {100*np.mean(np.abs(dt-poll_us) > 0.01*poll_us):.4f}%")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    lo, hi = dt.min() - 1, dt.max() + 1
    ax.hist(dt, bins=np.arange(lo, hi + 1, 1), color="#1f6feb", edgecolor="white")
    ax.axvline(poll_us, color="k", ls="--", lw=1.2, label=f"commanded {poll_us} us")
    ax.set_yscale("log")
    ax.set_xlabel("sampling interval (us)")
    ax.set_ylabel("occurrences (log)")
    ax.set_title(f"Sampling interval determinism\n"
                 f"sd = {dt.std():.2f} us = {100*dt.std()/np.median(dt):.3f}% of interval")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S1_timing.png"), dpi=150)
    plt.close(fig)
    return dt


def fig_noise(still, outdir, poll_s):
    if not still:
        return None
    cpis = sorted(still)
    print("\n=== STILL NOISE FLOOR ===")
    print(f"{'CPI':>7} | {'n':>6} | {'nonzero':>8} | {'max cnt':>8} | {'v_th mm/s':>10} | {'SQUAL':>6}")
    print("-" * 60)
    rows = []
    for c in cpis:
        s = stats(still[c], c, poll_s)
        vq = (MM_PER_INCH / c) / poll_s
        rows.append((c, s, vq))
        print(f"{c:>7} | {s['n']:>6} | {s['nonzero']:>8} | {s['max_cnt']:>8.0f} | "
              f"{vq:>10.2f} | {s['squal']:>6.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    axs[0].bar([str(c) for c, _, _ in rows], [s["nonzero"] for _, s, _ in rows],
               color=[COLS.get(c, "#666") for c, _, _ in rows])
    axs[0].set_xlabel("CPI")
    axs[0].set_ylabel("polls with non-zero displacement")
    axs[0].set_title("Stationary false counts\n(zero at every resolution)")
    axs[0].grid(alpha=0.3, axis="y")
    tot = sum(s["n"] for _, s, _ in rows)
    axs[0].text(0.5, 0.5, f"0 of {tot:,} samples", transform=axs[0].transAxes,
                ha="center", va="center", fontsize=13, color="#1a7f37")

    vq = [v for _, _, v in rows]
    axs[1].plot([c for c, _, _ in rows], vq, "o-", color="#1f6feb", lw=2, ms=8)
    for c, _, v in rows:
        axs[1].annotate(f"{v:.2f}", (c, v), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_xticks([c for c, _, _ in rows])
    axs[1].set_xticklabels([str(c) for c, _, _ in rows])
    axs[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].set_xlabel("CPI")
    axs[1].set_ylabel("$v_{th}$ = 1 count per poll (mm/s)")
    axs[1].set_title("Deadband threshold set by quantization alone")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S2_noise_vth.png"), dpi=150)
    plt.close(fig)
    return rows


def fig_calibration(dist, outdir):
    if not dist:
        return None
    cpis = sorted({c for c, _ in dist})
    print("\n=== COUNTS-PER-MM CALIBRATION ===")
    print(f"{'CPI':>7} | {'known':>7} | {'counts':>8} | {'meas cpm':>9} | {'exp cpm':>8} | {'dev %':>7}")
    print("-" * 62)
    devs = {c: [] for c in cpis}
    pts = []
    for (c, k), d in sorted(dist.items()):
        net = float(np.hypot(d["dx"].sum(), d["dy"].sum()))
        meas, exp = net / k, c / MM_PER_INCH
        dev = 100 * (meas - exp) / exp
        devs[c].append(dev)
        pts.append((c, k, net, meas, exp, dev))
        print(f"{c:>7} | {k:>5}mm | {net:>8.0f} | {meas:>9.1f} | {exp:>8.1f} | {dev:>+6.1f}%")
    print()
    for c in cpis:
        a = np.array(devs[c])
        print(f"  CPI {c:>5}: mean {a.mean():+6.1f}%   spread {a.min():+.1f}% to {a.max():+.1f}%")
    allv = np.concatenate([np.array(devs[c]) for c in cpis])
    print(f"\n  overall mean deviation {allv.mean():+.1f}%")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for c in cpis:
        sub = [(k, n) for cc, k, n, _, _, _ in pts if cc == c]
        ks = [k for k, _ in sub]
        ns = [n for _, n in sub]
        axs[0].plot(ks, ns, "o", ms=9, color=COLS.get(c, "#666"), label=f"{c} CPI")
        xs = np.linspace(0, max(ks) * 1.05, 30)
        axs[0].plot(xs, xs * c / MM_PER_INCH, "--", lw=1.1,
                    color=COLS.get(c, "#666"), alpha=0.6)
    axs[0].set_xlabel("known displacement (mm)")
    axs[0].set_ylabel("reported counts")
    axs[0].set_title("Counts-per-mm calibration\n(dashed = expected at that CPI)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    w = 0.2
    ks = sorted({k for _, k in dist})
    for i, k in enumerate(ks):
        vals = [next(d for cc, kk, _, _, _, d in pts if cc == c and kk == k) for c in cpis]
        axs[1].bar(np.arange(len(cpis)) + (i - 1) * w, vals, w,
                   label=f"{k} mm")
    axs[1].axhline(0, color="k", lw=1)
    axs[1].axhline(15, color="#d1242f", ls="--", lw=1)
    axs[1].axhline(-15, color="#d1242f", ls="--", lw=1)
    axs[1].set_xticks(np.arange(len(cpis)))
    axs[1].set_xticklabels([str(c) for c in cpis])
    axs[1].set_xlabel("CPI")
    axs[1].set_ylabel("deviation from expected counts/mm (%)")
    axs[1].set_title("Per-distance deviation\n(averaging hides the spread)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S3_calibration.png"), dpi=150)
    plt.close(fig)
    return pts


def fig_velocity(dist, outdir, poll_s):
    if not dist:
        return
    print("\n=== VELOCITY PROFILES AND HEADROOM ===")
    print(f"{'run':>18} | {'peak mm/s':>10} | {'max cnt/poll':>12} | {'cross-axis':>11}")
    print("-" * 60)
    peak_all = 0.0
    cross = []
    for (c, k), d in sorted(dist.items()):
        mag = np.hypot(d["dx"], d["dy"]).to_numpy()
        v = mag * (MM_PER_INCH / c) / poll_s
        sx, sy = d["dx"].sum(), d["dy"].sum()
        r = abs(sy) / max(abs(sx), 1)
        cross.append(r)
        peak_all = max(peak_all, v.max())
        print(f"{f'{k}mm-{c}':>18} | {v.max():>10.1f} | {mag.max():>12.0f} | {100*r:>10.1f}%")
    print(f"\n  fastest observed  {peak_all:.0f} mm/s ({peak_all/MM_PER_INCH:.1f} ips)")
    print(f"  datasheet limit   200 ips (Office mode), 650 ips (HP mode)")
    print(f"  utilisation       {100*peak_all/MM_PER_INCH/200:.1f}% of Office mode")
    print(f"  mean cross-axis   {100*np.mean(cross):.1f}%")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for (c, k), d in sorted(dist.items()):
        if k != 60:
            continue
        mag = np.hypot(d["dx"], d["dy"]).to_numpy()
        t = (d["t_us"].to_numpy() - d["t_us"].iloc[0]) * 1e-6
        v = mag * (MM_PER_INCH / c) / poll_s
        axs[0].plot(t * 1000, v, lw=1.1, color=COLS.get(c, "#666"),
                    alpha=0.85, label=f"{c} CPI")
    axs[0].set_xlabel("time (ms)")
    axs[0].set_ylabel("velocity (mm/s)")
    axs[0].set_title("Velocity profile, 60 mm slide\n(all four resolutions)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].bar(["observed\nmax"], [peak_all / MM_PER_INCH], color="#1f6feb", width=0.5)
    axs[1].bar(["Office\nmode"], [200], color="#8250df", width=0.5, alpha=0.5)
    axs[1].bar(["HP\nmode"], [650], color="#1a7f37", width=0.5, alpha=0.5)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("tracking speed (ips, log)")
    axs[1].set_title(f"Speed headroom\nobserved = {100*peak_all/MM_PER_INCH/200:.1f}% of Office mode")
    axs[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S4_velocity.png"), dpi=150)
    plt.close(fig)


def fig_optical(still, dist, outdir):
    S, SH, lab = [], [], []
    for c, d in sorted(still.items()):
        S.append(d["squal"].mean()); SH.append(d["shutter"].mean()); lab.append("still")
    for (c, k), d in sorted(dist.items()):
        S.append(d["squal"].mean()); SH.append(d["shutter"].mean()); lab.append("slide")
    S, SH = np.array(S), np.array(SH)
    lab = np.array(lab)
    if len(S) < 3:
        return
    r = np.corrcoef(S, SH)[0, 1]
    print("\n=== OPTICAL CONDITIONS ===")
    print(f"  corr(SQUAL, shutter) = {r:+.3f}")
    print(f"  SQUAL   {S.min():.0f} - {S.max():.0f}")
    print(f"  shutter {SH.min():.0f} - {SH.max():.0f}")
    for g in ("still", "slide"):
        m = lab == g
        if m.any():
            print(f"    {g:>6}: SQUAL {S[m].mean():6.1f}   shutter {SH[m].mean():8.0f}")
    print("\n  Groups separating in this plane indicates the runs were not all")
    print("  on the same surface; treat cross-group comparisons with care.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g, col, mk in (("still", "#1f6feb", "o"), ("slide", "#d1242f", "s")):
        m = lab == g
        if m.any():
            ax.plot(SH[m], S[m], mk, color=col, ms=9, alpha=0.8, label=g)
    ax.set_xlabel("shutter (exposure)")
    ax.set_ylabel("SQUAL (features resolved)")
    ax.set_title(f"Optical operating conditions across runs\n"
                 f"corr = {r:+.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S5_optical.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Slip sensor results figures")
    ap.add_argument("folder")
    ap.add_argument("--outdir", default="slip_figs")
    ap.add_argument("--poll-us", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    poll_s = args.poll_us * 1e-6

    still, dist = load(args.folder)
    if not still and not dist:
        sys.exit(f"No recognised captures in {args.folder}.\n"
                 "Expected noise-still-<cpi>.csv and dist-<mm>mm-<cpi>.csv")
    print(f"Loaded {len(still)} still + {len(dist)} slide captures")

    fig_timing(still, dist, args.outdir, args.poll_us)
    fig_noise(still, args.outdir, poll_s)
    fig_calibration(dist, args.outdir)
    fig_velocity(dist, args.outdir, poll_s)
    fig_optical(still, dist, args.outdir)

    rows = []
    for c, d in sorted(still.items()):
        s = stats(d, c, poll_s); s.update(run="still", cpi=c, known_mm=np.nan)
        rows.append(s)
    for (c, k), d in sorted(dist.items()):
        s = stats(d, c, poll_s); s.update(run="slide", cpi=c, known_mm=k)
        rows.append(s)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "slip_summary.csv"), index=False)

    print(f"\nFigures and slip_summary.csv written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
