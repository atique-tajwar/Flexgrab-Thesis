"""
FSR Calibration Analysis
========================

Reads a folder of CSVs produced by the FSR logger, extracts a steady-state
reading from each, and produces calibration + diagnostic plots.

Expected CSV schema (as written by the logger):
    Time, Voltage (V), Resistance (ohm)

Expected filename convention:
    <object>-<mass>g.csv        e.g.  bottle-490g.csv, perfume-185g.csv

The mass is parsed from the filename and converted to force as F = m*g,
i.e. this assumes DEAD-WEIGHT loading (object resting on the sensor).

Usage:
    python fsr_calibration.py <folder>
    python fsr_calibration.py <folder> --outdir plots --group bottle
    python fsr_calibration.py <folder> --settle 0.5 --exclude 305

Outputs (into --outdir):
    per-file/<name>.png      time series for each CSV
    raw_response.png         V vs F, all objects, grouped by prefix
    diagnostics.png          sensitivity, creep, leave-one-out residuals
    calibration.png          fitted calibration curve
    calibration_summary.csv  extracted steady-state values + fit residuals

Requires: numpy, pandas, matplotlib
"""

import argparse
import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81

VOLT_COL = "Voltage (V)"
RES_COL = "Resistance (ohm)"

# time axis reconstruction (the logged wall-clock stamps are not used)
SAMPLE_MS = 200.0
JITTER_MS = 15.0


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def parse_name(path):
    """Return (object_name, mass_grams) from '<object>-<mass>g.csv'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"^(.*?)-(\d+(?:\.\d+)?)g$", stem)
    if not m:
        return stem, None
    return m.group(1), float(m.group(2))


def synth_time(n, rng):
    """Reconstructed time axis: uniform sampling plus bounded jitter."""
    base = np.arange(n) * SAMPLE_MS
    return (base + rng.uniform(0.0, JITTER_MS, size=n)) / 1000.0


def load_folder(folder, settle):
    """Load every CSV, returning a list of per-file record dicts."""
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        sys.exit(f"No CSV files found in {folder}")

    rng = np.random.default_rng(0)
    records = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  skipping {os.path.basename(p)}: {e}")
            continue

        missing = [c for c in (VOLT_COL, RES_COL) if c not in df.columns]
        if missing:
            print(f"  skipping {os.path.basename(p)}: missing column(s) {missing}")
            continue

        obj, mass = parse_name(p)
        V = df[VOLT_COL].to_numpy(dtype=float)
        R = df[RES_COL].to_numpy(dtype=float)
        ok = np.isfinite(V) & np.isfinite(R)
        V, R = V[ok], R[ok]
        if len(V) < 8:
            print(f"  skipping {os.path.basename(p)}: too few valid rows")
            continue

        t = synth_time(len(V), rng)
        cut = int(len(V) * settle)          # steady-state window
        q = max(1, len(V) // 4)             # quarter window, for creep

        records.append(dict(
            path=p,
            stem=os.path.splitext(os.path.basename(p))[0],
            obj=obj,
            mass=mass,
            force=(mass * G / 1000.0) if mass is not None else np.nan,
            t=t, V=V, R=R,
            V_ss=V[cut:].mean(),
            V_sd=V[cut:].std(),
            R_ss=R[cut:].mean(),
            V_early=V[:q].mean(),
            V_late=V[-q:].mean(),
            n=len(V),
        ))

    records.sort(key=lambda r: (r["obj"], r["mass"] if r["mass"] is not None else 0))
    return records


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def fit_models(F, V, R):
    """Try several calibration forms; return list of (label, predict_fn, r2, maxerr)."""
    out = []

    def score(label, pred):
        res = F - pred
        ss_tot = np.sum((F - F.mean()) ** 2)
        r2 = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else np.nan
        out.append((label, pred, r2, np.max(np.abs(res))))

    # linear in V
    p = np.polyfit(V, F, 1)
    score(f"F = {p[0]:.3f}·V {p[1]:+.3f}", np.polyval(p, V))

    # power law F = a·V^n + c  (grid search on n)
    best = None
    for n in np.arange(0.2, 4.001, 0.01):
        A = np.vstack([V ** n, np.ones_like(V)]).T
        coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        ssr = np.sum((F - A @ coef) ** 2)
        if best is None or ssr < best[0]:
            best = (ssr, n, coef)
    _, n, coef = best
    score(f"F = {coef[0]:.3f}·V^{n:.2f} {coef[1]:+.3f}", coef[0] * V ** n + coef[1])

    # linear in conductance
    Gc = 1.0 / R
    p2 = np.polyfit(Gc, F, 1)
    score(f"F = {p2[0]:.0f}/R {p2[1]:+.3f}", np.polyval(p2, Gc))

    # power law in conductance
    c3 = np.polyfit(np.log(Gc), np.log(F), 1)
    score(f"F = {np.exp(c3[1]):.2f}·(1/R)^{c3[0]:.3f}", np.exp(c3[1]) * Gc ** c3[0])

    out.sort(key=lambda x: -x[2])
    return out


def loo_residuals(F, V):
    """Leave-one-out residuals under a linear F(V) fit."""
    r = np.full(len(F), np.nan)
    if len(F) < 3:
        return r
    for i in range(len(F)):
        k = np.ones(len(F), bool)
        k[i] = False
        p = np.polyfit(V[k], F[k], 1)
        r[i] = F[i] - np.polyval(p, V[i])
    return r


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_timeseries(rec, outdir):
    d = os.path.join(outdir, "per-file")
    os.makedirs(d, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axs[0].plot(rec["t"], rec["V"], "-", color="blue", lw=1.4)
    axs[0].set_ylabel("Voltage (V)")
    axs[0].grid(alpha=0.3)
    axs[0].set_title(rec["stem"])

    axs[1].plot(rec["t"], rec["R"], "--", color="red", lw=1.4)
    axs[1].set_ylabel("Resistance (Ω)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(d, rec["stem"] + ".png"), dpi=150)
    plt.close(fig)


def plot_raw(records, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    groups = sorted({r["obj"] for r in records if r["mass"] is not None})
    cmap = plt.get_cmap("tab10")

    for i, gname in enumerate(groups):
        pts = [r for r in records if r["obj"] == gname and r["mass"] is not None]
        pts.sort(key=lambda r: r["mass"])
        F = [r["force"] for r in pts]
        V = [r["V_ss"] for r in pts]
        style = "o-" if len(pts) > 2 else "s"
        ax.plot(F, V, style, color=cmap(i % 10), ms=7, lw=1.5, label=gname)
        for r in pts:
            ax.annotate(f"{r['mass']:.0f}g", (r["force"], r["V_ss"]),
                        textcoords="offset points", xytext=(5, -9), fontsize=6.5)

    ax.set_xlabel("Applied force  $F = mg$  (N)")
    ax.set_ylabel("Steady-state FSR voltage (V)")
    ax.set_title("Raw response by object")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "raw_response.png"), dpi=150)
    plt.close(fig)


def plot_diagnostics(recs, outdir):
    """recs: single-group, mass-sorted records used for calibration."""
    F = np.array([r["force"] for r in recs])
    V = np.array([r["V_ss"] for r in recs])
    mass = np.array([r["mass"] for r in recs])

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))

    # local sensitivity
    if len(F) > 1:
        sens = np.diff(V) / np.diff(F)
        mid = (F[:-1] + F[1:]) / 2
        axs[0].axhline(0, color="k", lw=0.8)
        axs[0].plot(mid, sens, "s-", color="#8250df", ms=7, lw=1.5)
        for x, s in zip(mid, sens):
            if s < 0.15:
                axs[0].annotate("flat/neg", (x, s), textcoords="offset points",
                                xytext=(0, -14), fontsize=6.5, ha="center",
                                color="#d1242f")
    axs[0].set_xlabel("$F$ (N)")
    axs[0].set_ylabel("local $dV/dF$ (V/N)")
    axs[0].set_title("Local sensitivity")
    axs[0].grid(alpha=0.3)

    # creep
    drift = np.array([100 * (r["V_late"] - r["V_early"]) / r["V_early"] for r in recs])
    axs[1].bar([f"{m:.0f}" for m in mass], drift, color="#1f6feb")
    axs[1].set_xlabel("mass (g)")
    axs[1].set_ylabel("drift, first→last quarter (%)")
    axs[1].set_title("Within-recording creep")
    axs[1].tick_params(axis="x", rotation=45, labelsize=7)
    axs[1].grid(alpha=0.3, axis="y")

    # leave-one-out residuals
    r_loo = loo_residuals(F, V)
    if np.isfinite(r_loo).any():
        sd = np.nanstd(r_loo)
        colors = ["#d1242f" if abs(x) > 2 * sd else
                  "#e3b341" if abs(x) > 1.5 * sd else "#1f6feb" for x in r_loo]
        axs[2].bar([f"{m:.0f}" for m in mass], r_loo, color=colors)
        axs[2].axhline(2 * sd, color="#d1242f", ls=":", lw=1)
        axs[2].axhline(-2 * sd, color="#d1242f", ls=":", lw=1)
    axs[2].set_xlabel("mass (g)")
    axs[2].set_ylabel("LOO residual (N)")
    axs[2].set_title("Leave-one-out residuals\n(red > 2σ, amber > 1.5σ)")
    axs[2].tick_params(axis="x", rotation=45, labelsize=7)
    axs[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diagnostics.png"), dpi=150)
    plt.close(fig)
    return r_loo


def plot_calibration(recs, excluded, outdir):
    F = np.array([r["force"] for r in recs])
    V = np.array([r["V_ss"] for r in recs])
    R = np.array([r["R_ss"] for r in recs])
    mass = np.array([r["mass"] for r in recs])

    keep = ~np.isin(mass, excluded) if excluded else np.ones(len(F), bool)
    if keep.sum() < 2:
        print("  too few points left after exclusions; skipping calibration plot")
        return None

    models = fit_models(F[keep], V[keep], R[keep])
    label, _, r2, maxerr = models[0]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(V[keep], F[keep], "o", color="#1f6feb", ms=8, label="used in fit")
    if (~keep).any():
        ax.plot(V[~keep], F[~keep], "o", color="#d1242f", ms=8, label="excluded")

    # draw the best model over a dense V sweep, refitting in V-space only
    if "/R" not in label:
        xs = np.linspace(V.min(), V.max(), 200)
        p = np.polyfit(V[keep], F[keep], 1)
        ax.plot(xs, np.polyval(p, xs), "-", color="#1f6feb", lw=1.8,
                label=f"F = {p[0]:.3f}·V {p[1]:+.3f}")

    for m_, v_, f_ in zip(mass, V, F):
        ax.annotate(f"{m_:.0f}g", (v_, f_), textcoords="offset points",
                    xytext=(6, -4), fontsize=6.5)

    ax.set_xlabel("Steady-state FSR voltage (V)")
    ax.set_ylabel("Applied force $F = mg$ (N)")
    ax.set_title(f"Calibration  —  best model: {label}\n$R^2$={r2:.4f}, max|err|={maxerr:.3f} N")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "calibration.png"), dpi=150)
    plt.close(fig)
    return models


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FSR calibration analysis")
    ap.add_argument("folder", help="folder containing the CSV files")
    ap.add_argument("--outdir", default="plots", help="output folder (default: plots)")
    ap.add_argument("--group", default=None,
                    help="object prefix to calibrate on, e.g. 'bottle'. "
                         "Default: the group with the most files.")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="fraction of each record discarded before averaging "
                         "(default 0.5 = use last half)")
    ap.add_argument("--exclude", type=float, nargs="*", default=[],
                    help="mass values (g) to exclude from the fit")
    ap.add_argument("--no-timeseries", action="store_true",
                    help="skip the per-file time-series plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading {args.folder} ...")
    records = load_folder(args.folder, args.settle)
    print(f"  loaded {len(records)} file(s)\n")

    print(f"{'file':>24} | {'mass g':>7} | {'F (N)':>6} | {'V_ss':>6} | "
          f"{'V sd':>6} | {'R_ss':>9} | {'creep %':>8}")
    print("-" * 82)
    for r in records:
        creep = 100 * (r["V_late"] - r["V_early"]) / r["V_early"]
        mass_s = f"{r['mass']:.0f}" if r["mass"] is not None else "n/a"
        force_s = f"{r['force']:.3f}" if r["mass"] is not None else "n/a"
        print(f"{r['stem']:>24} | {mass_s:>7} | {force_s:>6} | {r['V_ss']:>6.3f} | "
              f"{r['V_sd']:>6.4f} | {r['R_ss']:>9.1f} | {creep:>+7.2f}%")

    if not args.no_timeseries:
        print("\nWriting per-file time series ...")
        for r in records:
            plot_timeseries(r, args.outdir)

    plot_raw(records, args.outdir)

    # choose calibration group
    massed = [r for r in records if r["mass"] is not None]
    if not massed:
        sys.exit("\nNo filenames matched '<object>-<mass>g.csv'; cannot calibrate.")

    if args.group:
        recs = [r for r in massed if r["obj"] == args.group]
        if not recs:
            sys.exit(f"\nNo files with object prefix '{args.group}'.")
    else:
        counts = {}
        for r in massed:
            counts[r["obj"]] = counts.get(r["obj"], 0) + 1
        best = max(counts, key=counts.get)
        recs = [r for r in massed if r["obj"] == best]
        print(f"\nCalibrating on group '{best}' ({len(recs)} points). "
              f"Override with --group.")

    recs.sort(key=lambda r: r["mass"])
    if len(recs) < 3:
        sys.exit("\nNeed at least 3 points in the calibration group.")

    print("\nNOTE: only same-object points are used. FSRs respond to pressure\n"
          "distribution, so mixing objects with different contact footprints\n"
          "invalidates the calibration.\n")

    r_loo = plot_diagnostics(recs, args.outdir)
    models = plot_calibration(recs, args.exclude, args.outdir)

    if models:
        print("Candidate models (best first):")
        for label, _, r2, maxerr in models:
            print(f"  {label:<34}  R²={r2:>7.4f}   max|err|={maxerr:.3f} N")

    # summary CSV
    sd = np.nanstd(r_loo) if np.isfinite(r_loo).any() else np.nan
    rows = []
    for r, res in zip(recs, r_loo):
        rows.append(dict(
            file=r["stem"], object=r["obj"], mass_g=r["mass"],
            force_N=r["force"], V_ss=r["V_ss"], V_sd=r["V_sd"], R_ss=r["R_ss"],
            creep_pct=100 * (r["V_late"] - r["V_early"]) / r["V_early"],
            loo_residual_N=res,
            loo_sigma=abs(res) / sd if np.isfinite(sd) and sd > 0 else np.nan,
            excluded=r["mass"] in args.exclude,
        ))
    out_csv = os.path.join(args.outdir, "calibration_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    flagged = [r["file"] for r in rows if np.isfinite(r["loo_sigma"]) and r["loo_sigma"] > 1.5]
    if flagged:
        print(f"\nPoints with LOO residual > 1.5σ (inspect before trusting): {', '.join(flagged)}")

    print(f"\nDone. Outputs written to '{args.outdir}/'")


if __name__ == "__main__":
    main()