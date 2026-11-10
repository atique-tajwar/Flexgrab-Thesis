"""
PAW3395 Characterization Analysis
=================================

Analyses a folder of captures from paw_logger.py and produces the figures and
numbers for the slip-sensor characterization section.

Expected filenames:
    noise-<surface>.csv     stationary   -> noise floor, v_th
    surface-<material>.csv  swipe        -> tracking quality by material
    zheight-<mm>mm.csv      standoff     -> usable Z window
    dist-<mm>mm.csv         known move   -> counts-per-mm, verifies CPI
    slip-<object>.csv       real slip    -> measured slip profile

Usage:
    python paw_analysis.py <folder> --cpi 16000
    python paw_analysis.py <folder> --cpi 16000 --outdir paw_plots

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


def load(path):
    d = pd.read_csv(path)
    need = {"t_us", "dx", "dy", "motion"}
    if not need.issubset(d.columns):
        return None
    d = d.dropna().astype({"t_us": np.int64, "dx": int, "dy": int, "motion": int})
    if len(d) < 10:
        return None
    return d


def kind_and_label(stem):
    m = re.match(r"^(noise|surface|zheight|dist|slip)-(.+)$", stem)
    if not m:
        return None, stem
    return m.group(1), m.group(2)


def basic_stats(d, cpi):
    t = d["t_us"].to_numpy()
    dx = d["dx"].to_numpy().astype(float)
    dy = d["dy"].to_numpy().astype(float)
    dt = np.diff(t)
    mag = np.hypot(dx, dy)
    count_mm = MM_PER_INCH / cpi
    return dict(
        n=len(d),
        dt_mean_us=dt.mean() if len(dt) else np.nan,
        dt_jitter_us=dt.std() if len(dt) else np.nan,
        counts_rms=np.sqrt(np.mean(mag ** 2)),
        counts_max=mag.max(),
        motion_frac=d["motion"].mean(),
        total_x_mm=dx.sum() * count_mm,
        total_y_mm=dy.sum() * count_mm,
        path_mm=mag.sum() * count_mm,
    )


# ----------------------------------------------------------------------

def analyse_noise(recs, cpi, outdir):
    """Stationary captures -> noise floor and threshold selection."""
    if not recs:
        return None
    print("\n=== NOISE FLOOR (stationary) ===")
    print(f"{'surface':>16} | {'n':>7} | {'RMS cnt':>8} | {'max cnt':>8} | {'polls w/ motion':>16}")
    print("-" * 68)
    for label, d in recs:
        s = basic_stats(d, cpi)
        print(f"{label:>16} | {s['n']:>7} | {s['counts_rms']:>8.3f} | "
              f"{s['counts_max']:>8.0f} | {100*s['motion_frac']:>14.2f}%")

    # threshold sweep on the pooled stationary data
    allmag = np.concatenate([np.hypot(d["dx"], d["dy"]).to_numpy() for _, d in recs])
    dt = np.concatenate([np.diff(d["t_us"].to_numpy()) for _, d in recs])
    dt_s = np.median(dt) * 1e-6
    count_mm = MM_PER_INCH / cpi

    ths = np.arange(0, 11)
    rates = [(allmag > t).mean() for t in ths]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    axs[0].hist(allmag, bins=np.arange(-0.5, max(6, allmag.max()) + 1.5),
                color="#1f6feb", edgecolor="white")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("|displacement| per poll (counts)")
    axs[0].set_ylabel("occurrences (log)")
    axs[0].set_title("Stationary noise distribution")
    axs[0].grid(alpha=0.3)

    axs[1].semilogy(ths, np.maximum(rates, 1e-6), "o-", color="#d1242f")
    axs[1].set_xlabel("threshold (counts per poll)")
    axs[1].set_ylabel("false-trigger rate (fraction of polls)")
    axs[1].set_title("Threshold vs. false-trigger rate")
    axs[1].grid(alpha=0.3)
    sec = axs[1].secondary_xaxis(
        "top", functions=(lambda c: c * count_mm / dt_s,
                          lambda v: v * dt_s / count_mm))
    sec.set_xlabel("equivalent $v_{th}$ (mm/s)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_floor.png"), dpi=150)
    plt.close(fig)

    print(f"\nPooled: RMS = {np.sqrt(np.mean(allmag**2)):.3f} counts, "
          f"max = {allmag.max():.0f} counts")
    for t, r in zip(ths, rates):
        if r == 0:
            print(f"Zero false triggers at threshold >= {t} counts "
                  f"({t*count_mm/dt_s:.2f} mm/s)  <-- suggested v_th")
            break
    return allmag


def analyse_surface(recs, cpi, outdir):
    if not recs:
        return
    print("\n=== SURFACE DEPENDENCE ===")
    print(f"{'material':>16} | {'path (mm)':>10} | {'RMS cnt':>8} | {'dropout %':>10}")
    print("-" * 54)
    labels, paths, drops = [], [], []
    for label, d in recs:
        s = basic_stats(d, cpi)
        # dropout: polls reporting nothing during an intended sweep
        drop = 100 * (1 - s["motion_frac"])
        print(f"{label:>16} | {s['path_mm']:>10.2f} | {s['counts_rms']:>8.2f} | {drop:>9.1f}%")
        labels.append(label); paths.append(s["path_mm"]); drops.append(drop)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    o = np.argsort(paths)
    axs[0].barh([labels[i] for i in o], [paths[i] for i in o], color="#1f6feb")
    axs[0].set_xlabel("total tracked path (mm)")
    axs[0].set_title("Tracked distance per surface\n(same intended sweep)")
    axs[0].grid(alpha=0.3, axis="x")

    o = np.argsort(drops)
    cols = ["#d1242f" if drops[i] > 50 else "#1f6feb" for i in o]
    axs[1].barh([labels[i] for i in o], [drops[i] for i in o], color=cols)
    axs[1].set_xlabel("polls reporting no motion (%)")
    axs[1].set_title("Dropout rate\n(red: >50%, tracking unreliable)")
    axs[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "surface_dependence.png"), dpi=150)
    plt.close(fig)


def analyse_zheight(recs, cpi, outdir):
    if not recs:
        return
    print("\n=== Z-HEIGHT (standoff) ===")
    zs, paths, drops = [], [], []
    for label, d in recs:
        m = re.match(r"^([\d.]+)mm$", label)
        if not m:
            print(f"  skipping '{label}': expected e.g. 'zheight-2.4mm.csv'")
            continue
        z = float(m.group(1))
        s = basic_stats(d, cpi)
        zs.append(z); paths.append(s["path_mm"]); drops.append(100*(1-s["motion_frac"]))
    if not zs:
        return
    o = np.argsort(zs)
    zs = np.array(zs)[o]; paths = np.array(paths)[o]; drops = np.array(drops)[o]

    print(f"{'Z (mm)':>8} | {'path (mm)':>10} | {'dropout %':>10}")
    print("-" * 34)
    for z, p, dr in zip(zs, paths, drops):
        print(f"{z:>8.2f} | {p:>10.2f} | {dr:>9.1f}%")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(zs, paths, "o-", color="#1f6feb", lw=2, ms=7, label="tracked path")
    ax.axvspan(2.2, 2.6, color="#1a7f37", alpha=0.12,
               label="datasheet window (2.2-2.6 mm)")
    ax.set_xlabel("standoff Z (mm)")
    ax.set_ylabel("tracked path (mm)")
    ax.set_title("Tracking vs. sensor standoff")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "zheight.png"), dpi=150)
    plt.close(fig)

    good = zs[paths > 0.8 * paths.max()]
    if len(good):
        print(f"\nUsable window (>80% of best): {good.min():.2f} - {good.max():.2f} mm")
        print("If this is narrow, the sensor must sit on a rigid sub-frame:")
        print("a compliant fingertip deforming under load would move it out of range.")


def analyse_distance(recs, cpi, outdir):
    if not recs:
        return
    print("\n=== DISTANCE CALIBRATION (verifies CPI / DPI_Config) ===")
    known, measured = [], []
    for label, d in recs:
        m = re.match(r"^([\d.]+)mm$", label)
        if not m:
            continue
        kn = float(m.group(1))
        counts = np.hypot(d["dx"].sum(), d["dy"].sum())
        known.append(kn); measured.append(counts)
    if not known:
        return
    known = np.array(known); measured = np.array(measured)
    cpm = measured / known                     # counts per mm
    expected = cpi / MM_PER_INCH

    print(f"{'known (mm)':>11} | {'counts':>10} | {'counts/mm':>10}")
    print("-" * 38)
    for k, c, r in zip(known, measured, cpm):
        print(f"{k:>11.1f} | {c:>10.0f} | {r:>10.2f}")
    print(f"\nexpected counts/mm at CPI={cpi}: {expected:.2f}")
    print(f"measured (mean):                {cpm.mean():.2f}")
    err = 100 * (cpm.mean() - expected) / expected
    print(f"deviation: {err:+.1f}%")
    if abs(err) > 15:
        print("\n** Large deviation. Either DPI_Config() does not take CPI directly,")
        print("   or the standoff is outside the focal window. Check both before")
        print("   trusting any velocity computed from these counts.")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(known, measured, "o", color="#1f6feb", ms=9, label="measured")
    xs = np.linspace(0, known.max()*1.05, 50)
    ax.plot(xs, xs*expected, "--", color="#1a7f37", lw=1.6,
            label=f"expected at {cpi} CPI")
    p = np.polyfit(known, measured, 1)
    ax.plot(xs, np.polyval(p, xs), "-", color="#d1242f", lw=1.4,
            label=f"fit: {p[0]:.1f} counts/mm")
    ax.set_xlabel("known displacement (mm)")
    ax.set_ylabel("reported counts")
    ax.set_title("Counts-per-millimetre calibration")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "distance_calibration.png"), dpi=150)
    plt.close(fig)


def analyse_slip(recs, cpi, outdir):
    if not recs:
        return
    print("\n=== MEASURED SLIP PROFILES ===")
    count_mm = MM_PER_INCH / cpi
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, d in recs:
        t = (d["t_us"].to_numpy() - d["t_us"].iloc[0]) * 1e-6
        dt = np.gradient(t)
        v = np.hypot(d["dx"], d["dy"]).to_numpy() * count_mm / np.maximum(dt, 1e-9)
        ax.plot(t*1000, v, lw=1.2, label=label)
        print(f"  {label:>14}: peak {v.max():.1f} mm/s, "
              f"total {np.hypot(d['dx'],d['dy']).sum()*count_mm:.2f} mm")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("slip velocity (mm/s)")
    ax.set_title("Measured slip velocity profiles")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "slip_profiles.png"), dpi=150)
    plt.close(fig)
    print("\nCompare the onset shape against the smooth Coulomb model used in")
    print("simulation: a stick-slip jerk at onset would change K_s tuning.")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PAW3395 characterization analysis")
    ap.add_argument("folder")
    ap.add_argument("--cpi", type=int, required=True,
                    help="CPI the captures were taken at")
    ap.add_argument("--outdir", default="paw_plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    buckets = {k: [] for k in ("noise", "surface", "zheight", "dist", "slip")}
    unknown = []
    for p in sorted(glob.glob(os.path.join(args.folder, "*.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        d = load(p)
        if d is None:
            print(f"  skipping {stem}: unreadable or too short")
            continue
        kind, label = kind_and_label(stem)
        if kind is None:
            unknown.append(stem)
            continue
        buckets[kind].append((label, d))

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        sys.exit(f"No recognised captures in {args.folder}. "
                 "Expected names like noise-paper.csv, dist-100mm.csv")
    print(f"Loaded {total} capture(s) at CPI={args.cpi} "
          f"(count = {MM_PER_INCH/args.cpi*1000:.2f} um)")
    if unknown:
        print(f"Unrecognised (ignored): {', '.join(unknown)}")

    # timing sanity check across everything
    print("\n=== TIMING ===")
    for kind, recs in buckets.items():
        for label, d in recs:
            s = basic_stats(d, args.cpi)
            warn = ""
            if s["dt_jitter_us"] > 0.1 * s["dt_mean_us"]:
                warn = "  ** jitter >10% of interval"
            print(f"  {kind}-{label:<14} dt = {s['dt_mean_us']:7.1f} "
                  f"+/- {s['dt_jitter_us']:5.1f} us{warn}")

    analyse_noise(buckets["noise"], args.cpi, args.outdir)
    analyse_surface(buckets["surface"], args.cpi, args.outdir)
    analyse_zheight(buckets["zheight"], args.cpi, args.outdir)
    analyse_distance(buckets["dist"], args.cpi, args.outdir)
    analyse_slip(buckets["slip"], args.cpi, args.outdir)

    print(f"\nFigures written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
