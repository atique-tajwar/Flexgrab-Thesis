"""
PAW3395 Characterization Analysis — multi-CPI version
=======================================================

Like paw_analysis.py, but reads CPI directly out of each filename instead of
taking one --cpi value for the whole folder. This lets you drop every capture
into a single folder and compare CPI settings side by side in one run.

Expected filenames (CPI is now a required trailing field):
    noise-<surface>-<cpi>.csv     stationary   e.g. noise-paper-1600.csv
    surface-<material>-<cpi>.csv  swipe        e.g. surface-pet-3200.csv
    zheight-<mm>mm-<cpi>.csv      standoff     e.g. zheight-5.0mm-3200.csv
    dist-<mm>mm-<cpi>.csv         known move   e.g. dist-20mm-1600.csv
    slip-<object>-<cpi>.csv       real slip    e.g. slip-bottle-3200.csv

Output columns expected: t_us,dx,dy,motion[,squal,shutter]
(squal/shutter are optional — used for noise settle-time trimming if present)

Usage:
    python paw_analysis_multi.py slip_data
    python paw_analysis_multi.py slip_data --outdir paw_plots --settle-s 5

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
CMAP = plt.get_cmap("tab10")

# kind - label - cpi, e.g. "dist-20mm-1600" -> ("dist", "20mm", 1600)
FNAME_RE = re.compile(r"^(noise|surface|zheight|dist|slip)-(.+)-(\d+)$")
MM_RE = re.compile(r"^([\d.]+)mm$")


def load(path):
    d = pd.read_csv(path)
    need = {"t_us", "dx", "dy", "motion"}
    if not need.issubset(d.columns):
        return None
    keep = list(need | ({"squal", "shutter"} & set(d.columns)))
    d = d[keep].dropna()
    base_types = {"t_us": np.int64, "dx": int, "dy": int, "motion": int}
    d = d.astype(base_types)
    if len(d) < 10:
        return None
    return d


def parse_stem(stem):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    kind, label, cpi_str = m.groups()
    return kind, label, int(cpi_str)


def color_for(cpi, cpi_list):
    return CMAP(cpi_list.index(cpi) % 10)


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
        path_mm=mag.sum() * count_mm,
    )


def trim_settle(d, settle_s):
    """Drop the first settle_s seconds — sensor may still be in a
    run-mode / high-frame-rate burst before downshifting to rest mode,
    which pollutes a pooled stationary-noise estimate."""
    if settle_s <= 0 or len(d) == 0:
        return d
    t0 = d["t_us"].iloc[0]
    cutoff = t0 + int(settle_s * 1e6)
    trimmed = d[d["t_us"] >= cutoff]
    return trimmed if len(trimmed) >= 10 else d


# ----------------------------------------------------------------------

def analyse_noise_multi(cpi_groups, outdir, settle_s):
    if not cpi_groups:
        return
    print("\n=== NOISE FLOOR (stationary), by CPI ===")
    if settle_s > 0:
        print(f"(discarding first {settle_s:.0f}s of each capture — "
              f"sensor may still be settling out of a run-mode burst)")
    print(f"{'cpi':>7} | {'surface':>10} | {'n':>7} | {'RMS cnt':>8} | "
          f"{'max cnt':>8} | {'polls w/ motion':>16}")
    print("-" * 72)

    cpi_list = sorted(cpi_groups.keys())
    pooled = {}
    for cpi in cpi_list:
        mags, dts = [], []
        for label, d in cpi_groups[cpi]:
            dt_trim = trim_settle(d, settle_s)
            s = basic_stats(dt_trim, cpi)
            print(f"{cpi:>7} | {label:>10} | {s['n']:>7} | {s['counts_rms']:>8.3f} | "
                  f"{s['counts_max']:>8.0f} | {100*s['motion_frac']:>14.2f}%")
            mags.append(np.hypot(dt_trim["dx"], dt_trim["dy"]).to_numpy())
            dts.append(np.diff(dt_trim["t_us"].to_numpy()))
        pooled[cpi] = (np.concatenate(mags), np.concatenate(dts))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for cpi in cpi_list:
        allmag, dt = pooled[cpi]
        c = color_for(cpi, cpi_list)
        axs[0].hist(allmag, bins=np.arange(-0.5, max(6, allmag.max()) + 1.5),
                    color=c, alpha=0.5, label=f"{cpi} CPI", edgecolor="white")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("|displacement| per poll (counts)")
    axs[0].set_ylabel("occurrences (log)")
    axs[0].set_title("Stationary noise distribution by CPI")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    ths = np.arange(0, 11)
    print(f"\n{'cpi':>7} | {'RMS cnt':>8} | {'max cnt':>8} | {'suggested v_th (mm/s)':>22}")
    print("-" * 54)
    for cpi in cpi_list:
        allmag, dt = pooled[cpi]
        c = color_for(cpi, cpi_list)
        dt_s = np.median(dt) * 1e-6
        count_mm = MM_PER_INCH / cpi
        rates = [(allmag > t).mean() for t in ths]
        axs[1].semilogy(ths, np.maximum(rates, 1e-6), "o-", color=c, label=f"{cpi} CPI")

        vth_str = "none found <=10 cnt"
        for t, r in zip(ths, rates):
            if r == 0:
                vth_str = f"{t*count_mm/dt_s:.2f} (thr={t} cnt)"
                break
        print(f"{cpi:>7} | {np.sqrt(np.mean(allmag**2)):>8.3f} | "
              f"{allmag.max():>8.0f} | {vth_str:>22}")

    axs[1].set_xlabel("threshold (counts per poll)")
    axs[1].set_ylabel("false-trigger rate (fraction of polls)")
    axs[1].set_title("Threshold vs. false-trigger rate, by CPI")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_floor_multicpi.png"), dpi=150)
    plt.close(fig)


def analyse_distance_multi(cpi_groups, outdir):
    if not cpi_groups:
        return
    print("\n=== DISTANCE CALIBRATION (verifies CPI / DPI_Config), by CPI ===")
    cpi_list = sorted(cpi_groups.keys())

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    deviations = {}

    for cpi in cpi_list:
        known, measured = [], []
        for label, d in cpi_groups[cpi]:
            m = MM_RE.match(label)
            if not m:
                print(f"  skipping dist-{label}-{cpi}: expected e.g. '20mm'")
                continue
            kn = float(m.group(1))
            counts = np.hypot(d["dx"].sum(), d["dy"].sum())
            known.append(kn); measured.append(counts)
        if not known:
            continue
        known = np.array(known); measured = np.array(measured)
        cpm = measured / known
        expected = cpi / MM_PER_INCH
        err = 100 * (cpm.mean() - expected) / expected
        deviations[cpi] = err

        print(f"\n  CPI {cpi} (expected {expected:.2f} counts/mm):")
        for k, c, r in zip(known, measured, cpm):
            print(f"    {k:>6.1f} mm | {c:>8.0f} counts | {r:>7.2f} counts/mm")
        print(f"    measured mean: {cpm.mean():.2f} counts/mm   deviation: {err:+.1f}%")
        if abs(err) > 15:
            print("    ** large deviation — check DPI_Config() semantics / standoff focus")

        c = color_for(cpi, cpi_list)
        axs[0].plot(known, measured, "o", color=c, ms=8, label=f"{cpi} CPI measured")
        xs = np.linspace(0, known.max()*1.05, 50)
        axs[0].plot(xs, xs*expected, "--", color=c, lw=1.2, alpha=0.6)

    axs[0].set_xlabel("known displacement (mm)")
    axs[0].set_ylabel("reported counts")
    axs[0].set_title("Counts-per-mm calibration\n(dashed = expected at that CPI)")
    axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)

    if deviations:
        cpis_sorted = sorted(deviations.keys())
        errs = [deviations[c] for c in cpis_sorted]
        cols = [color_for(c, cpi_list) for c in cpis_sorted]
        axs[1].bar([str(c) for c in cpis_sorted], errs, color=cols)
        axs[1].axhline(0, color="black", lw=0.8)
        axs[1].axhline(15, color="#d1242f", lw=0.8, ls="--")
        axs[1].axhline(-15, color="#d1242f", lw=0.8, ls="--")
        axs[1].set_xlabel("CPI")
        axs[1].set_ylabel("deviation from expected counts/mm (%)")
        axs[1].set_title("CPI calibration accuracy\n(red dashed = 15% flag line)")
        axs[1].grid(alpha=0.3, axis="y")

        best = min(deviations, key=lambda c: abs(deviations[c]))
        print(f"\nBest calibration match: CPI {best} "
              f"({deviations[best]:+.1f}% deviation)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "distance_calibration_multicpi.png"), dpi=150)
    plt.close(fig)
    return deviations


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PAW3395 multi-CPI characterization analysis")
    ap.add_argument("folder")
    ap.add_argument("--outdir", default="paw_plots")
    ap.add_argument("--settle-s", type=float, default=5.0,
                    help="seconds to discard from the start of each noise "
                         "capture before pooling (default 5.0, 0 to disable)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # buckets[kind][cpi] = [(label, df), ...]
    buckets = {k: {} for k in ("noise", "surface", "zheight", "dist", "slip")}
    unknown = []
    for p in sorted(glob.glob(os.path.join(args.folder, "*.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        parsed = parse_stem(stem)
        if parsed is None:
            unknown.append(stem)
            continue
        kind, label, cpi = parsed
        d = load(p)
        if d is None:
            print(f"  skipping {stem}: unreadable or too short")
            continue
        buckets[kind].setdefault(cpi, []).append((label, d))

    total = sum(len(v) for group in buckets.values() for v in group.values())
    if total == 0:
        sys.exit(f"No recognised captures in {args.folder}. "
                 "Expected names like noise-paper-1600.csv, dist-20mm-3200.csv")
    print(f"Loaded {total} capture(s) across "
          f"{len({c for g in buckets.values() for c in g})} CPI setting(s)")
    if unknown:
        print(f"Unrecognised (ignored — missing trailing -<cpi>?): {', '.join(unknown)}")

    print("\n=== TIMING ===")
    for kind, group in buckets.items():
        for cpi, recs in sorted(group.items()):
            for label, d in recs:
                s = basic_stats(d, cpi)
                warn = "  ** jitter >10% of interval" if (
                    s["dt_jitter_us"] > 0.1 * s["dt_mean_us"]) else ""
                print(f"  {kind}-{label}-{cpi:<6} dt = {s['dt_mean_us']:7.1f} "
                      f"+/- {s['dt_jitter_us']:5.1f} us{warn}")

    analyse_noise_multi(buckets["noise"], args.outdir, args.settle_s)
    analyse_distance_multi(buckets["dist"], args.outdir)
    # surface / zheight / slip follow the same per-CPI grouping pattern and
    # can be extended the same way once you have captures for them.

    print(f"\nFigures written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
