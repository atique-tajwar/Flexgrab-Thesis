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

def detect_poll_us(t_us):
    """Median sampling interval of a capture, correcting uint32 rollover."""
    dt = np.diff(np.asarray(t_us, dtype=np.int64))
    dt = dt % (1 << 32)
    return float(np.median(dt))


def load(folder):
    """Group captures by condition, collecting repeats.

    Each repeat is stored as (name, dataframe, poll_s) with its OWN measured
    sampling interval. A dataset may legitimately contain several poll rates:
    counts-per-mm is independent of sampling rate, so those captures remain
    valid for calibration even though their velocities must be computed with
    the correct interval.
    """
    still, dist = {}, {}
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        b = os.path.basename(p)
        d = pd.read_csv(p)
        if not {"t_us", "dx", "dy", "squal", "shutter"}.issubset(d.columns):
            continue
        if len(d) < 10:
            continue
        poll_s = detect_poll_us(d["t_us"]) * 1e-6
        m = re.match(r"^noise-still-(\d+)(?:-r(\d+))?\.csv$", b)
        if m:
            still.setdefault(int(m.group(1)), []).append((b, d, poll_s))
            continue
        m = re.match(r"^dist-(\d+)mm-(\d+)(?:-r(\d+))?\.csv$", b)
        if m:
            dist.setdefault((int(m.group(2)), int(m.group(1))), []).append((b, d, poll_s))
    return still, dist


def all_frames(group):
    """Flatten to a list of (dataframe, poll_s)."""
    out = []
    for v in group.values():
        out.extend((df, ps) for _, df, ps in v)
    return out


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

def unwrap_dt(t_us):
    """Intervals from a uint32 micros() timebase, correcting rollover.

    micros() wraps every 2^32 us (~71.6 min). A wrap makes a normal interval
    appear as a ~-4.29e9 jump, so any negative difference is corrected by
    adding 2^32.
    """
    dt = np.diff(t_us.astype(np.int64))
    wraps = int((dt < 0).sum())
    dt[dt < 0] += 2 ** 32
    return dt, wraps


def fig_timing(still, dist, outdir):
    """Report timing per detected poll-rate group rather than pooling."""
    recs = ([(nm, df, ps) for v in still.values() for nm, df, ps in v] +
            [(nm, df, ps) for v in dist.values() for nm, df, ps in v])

    groups = {}
    for nm, df, ps in recs:
        key = int(round(ps * 1e6))
        groups.setdefault(key, []).append((nm, df))

    print("\n=== TIMING ===")
    print(f"  poll-rate groups detected: "
          f"{', '.join(f'{k} us ({len(v)} files)' for k, v in sorted(groups.items()))}")
    print("\n  Counts-per-mm is independent of sampling rate, so every capture")
    print("  remains valid for calibration. Velocity-derived quantities are")
    print("  computed with each file's own interval.\n")

    print(f"{'nominal':>9} | {'files':>6} | {'intervals':>10} | {'median us':>9} | "
          f"{'sd us':>8} | {'sd %':>7} | {'wraps':>6}")
    print("-" * 74)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k in sorted(groups):
        dts, wraps = [], 0
        for _, d in groups[k]:
            raw = np.diff(d["t_us"].to_numpy().astype(np.int64))
            wraps += int((raw < 0).sum())
            dts.append(raw % (1 << 32))
        dt = np.concatenate(dts)
        print(f"{k:>9} | {len(groups[k]):>6} | {len(dt):>10} | {np.median(dt):>9.0f} | "
              f"{dt.std():>8.2f} | {100*dt.std()/np.median(dt):>6.3f}% | {wraps:>6}")
        lo, hi = k - 30, k + 30
        ax.hist(np.clip(dt, lo, hi), bins=np.arange(lo, hi + 1, 1),
                alpha=0.75, label=f"{k} us ({len(groups[k])} files)")
    ax.set_yscale("log")
    ax.set_xlabel("sampling interval (us)")
    ax.set_ylabel("occurrences (log)")
    ax.set_title("Sampling interval determinism, by configured rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S1_timing.png"), dpi=150)
    plt.close(fig)
    return groups


def fig_noise(still, outdir):
    if not still:
        return None
    cpis = sorted(still)
    print("\n=== STILL NOISE FLOOR ===")
    print(f"{'CPI':>7} | {'reps':>5} | {'samples':>8} | {'nonzero':>8} | "
          f"{'v_th mm/s':>10} | {'SQUAL mean+/-sd':>16}")
    print("-" * 72)
    rows = []
    for c in cpis:
        reps = still[c]
        n_tot = sum(len(d) for _, d, _ in reps)
        nz_tot = sum(int((np.hypot(d["dx"], d["dy"]) > 0).sum()) for _, d, _ in reps)
        sq = np.array([d["squal"].mean() for _, d, _ in reps])
        ps = np.array([p for _, _, p in reps])
        vq = (MM_PER_INCH / c) / ps.mean()
        rows.append(dict(cpi=c, reps=len(reps), n=n_tot, nz=nz_tot, vq=vq,
                         squal_m=sq.mean(), squal_sd=sq.std()))
        print(f"{c:>7} | {len(reps):>5} | {n_tot:>8} | {nz_tot:>8} | "
              f"{vq:>10.2f} | {sq.mean():>7.1f} +/- {sq.std():<5.1f}")
    tot = sum(r["n"] for r in rows)
    totnz = sum(r["nz"] for r in rows)
    print(f"\n  pooled: {totnz} non-zero of {tot:,} samples")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    axs[0].bar([str(r["cpi"]) for r in rows], [r["nz"] for r in rows],
               color=[COLS.get(r["cpi"], "#666") for r in rows])
    axs[0].set_xlabel("CPI")
    axs[0].set_ylabel("polls with non-zero displacement")
    axs[0].set_title("Stationary false counts\n(all repeats pooled)")
    axs[0].grid(alpha=0.3, axis="y")
    axs[0].text(0.5, 0.5, f"{totnz} of {tot:,} samples",
                transform=axs[0].transAxes, ha="center", va="center",
                fontsize=13, color="#1a7f37" if totnz == 0 else "#d1242f")

    cs = [r["cpi"] for r in rows]
    vq = [r["vq"] for r in rows]
    axs[1].plot(cs, vq, "o-", color="#1f6feb", lw=2, ms=8)
    for c, v in zip(cs, vq):
        axs[1].annotate(f"{v:.2f}", (c, v), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_xticks(cs)
    axs[1].set_xticklabels([str(c) for c in cs])
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
    """With repeats, separate systematic bias from repeat-to-repeat scatter."""
    if not dist:
        return None
    cpis = sorted({c for c, _ in dist})
    knowns = sorted({k for _, k in dist})

    print("\n=== COUNTS-PER-MM CALIBRATION ===")
    print(f"{'CPI':>7} | {'known':>7} | {'reps':>5} | {'counts mean+/-sd':>18} | "
          f"{'meas cpm':>9} | {'exp cpm':>8} | {'dev %':>7} | {'rep CV':>7}")
    print("-" * 92)
    cell = {}
    for c in cpis:
        for k in knowns:
            reps = dist.get((c, k))
            if not reps:
                continue
            # counts-per-mm does not involve time, so captures at any
            # poll rate are valid here
            nets = np.array([float(np.hypot(d["dx"].sum(), d["dy"].sum()))
                             for _, d, _ in reps])
            meas = nets.mean() / k
            exp = c / MM_PER_INCH
            dev = 100 * (meas - exp) / exp
            cv = 100 * nets.std() / nets.mean() if nets.mean() else np.nan
            cell[(c, k)] = dict(nets=nets, meas=meas, exp=exp, dev=dev, cv=cv)
            print(f"{c:>7} | {k:>5}mm | {len(reps):>5} | "
                  f"{nets.mean():>9.0f} +/- {nets.std():<6.0f} | "
                  f"{meas:>9.1f} | {exp:>8.1f} | {dev:>+6.1f}% | {cv:>6.1f}%")

    print()
    for c in cpis:
        dv = np.array([cell[(c, k)]["dev"] for k in knowns if (c, k) in cell])
        print(f"  CPI {c:>5}: mean {dv.mean():+6.1f}%   spread {dv.min():+.1f}% to {dv.max():+.1f}%")
    alldev = np.array([v["dev"] for v in cell.values()])
    allcv = np.array([v["cv"] for v in cell.values() if np.isfinite(v["cv"])])
    print(f"\n  overall mean deviation {alldev.mean():+.1f}%")
    if len(allcv):
        print(f"  mean repeat-to-repeat CV {allcv.mean():.1f}%")
        print("\n  Interpretation: repeat CV measures technique scatter; the")
        print("  deviation measures fixture bias. If the per-distance pattern")
        print("  repeats across CPI while CV stays small, the bias is systematic.")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for c in cpis:
        ks = [k for k in knowns if (c, k) in cell]
        mu = [cell[(c, k)]["nets"].mean() for k in ks]
        sd = [cell[(c, k)]["nets"].std() for k in ks]
        axs[0].errorbar(ks, mu, yerr=sd, fmt="o", ms=8, capsize=4,
                        color=COLS.get(c, "#666"), label=f"{c} CPI")
        xs = np.linspace(0, max(ks) * 1.05, 30)
        axs[0].plot(xs, xs * c / MM_PER_INCH, "--", lw=1.1,
                    color=COLS.get(c, "#666"), alpha=0.6)
    axs[0].set_xlabel("known displacement (mm)")
    axs[0].set_ylabel("reported counts")
    axs[0].set_title("Counts-per-mm calibration\n(error bars = spread across repeats)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    w = 0.8 / max(len(knowns), 1)
    for i, k in enumerate(knowns):
        vals, errs, xs = [], [], []
        for j, c in enumerate(cpis):
            if (c, k) not in cell:
                continue
            v = cell[(c, k)]
            nets = v["nets"]
            devs = 100 * (nets / k - v["exp"]) / v["exp"]
            vals.append(devs.mean()); errs.append(devs.std())
            xs.append(j + (i - (len(knowns) - 1) / 2) * w)
        axs[1].bar(xs, vals, w, yerr=errs, capsize=3, label=f"{k} mm")
    axs[1].axhline(0, color="k", lw=1)
    axs[1].axhline(15, color="#d1242f", ls="--", lw=1)
    axs[1].axhline(-15, color="#d1242f", ls="--", lw=1)
    axs[1].set_xticks(range(len(cpis)))
    axs[1].set_xticklabels([str(c) for c in cpis])
    axs[1].set_xlabel("CPI")
    axs[1].set_ylabel("deviation from expected counts/mm (%)")
    axs[1].set_title("Per-distance deviation\n(error bars = repeat scatter)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S3_calibration.png"), dpi=150)
    plt.close(fig)
    return cell


def fig_velocity(dist, outdir):
    """Velocity uses each capture's own sampling interval."""
    if not dist:
        return
    print("\n=== VELOCITY PROFILES AND HEADROOM ===")
    print("  (each capture uses its own measured sampling interval)\n")
    print(f"{'run':>18} | {'reps':>5} | {'peak mm/s':>10} | {'max cnt/poll':>12} | {'cross-axis':>11}")
    print("-" * 70)
    peak_all = 0.0
    cross = []
    for (c, k), reps in sorted(dist.items()):
        pk, mc, cr = [], [], []
        for _, d, ps in reps:
            mag = np.hypot(d["dx"], d["dy"]).to_numpy()
            pk.append((mag * (MM_PER_INCH / c) / ps).max())
            mc.append(mag.max())
            sx, sy = d["dx"].sum(), d["dy"].sum()
            cr.append(abs(sy) / max(abs(sx), 1))
        cross.extend(cr)
        peak_all = max(peak_all, max(pk))
        print(f"{f'{k}mm-{c}':>18} | {len(reps):>5} | {np.mean(pk):>10.1f} | "
              f"{np.mean(mc):>12.0f} | {100*np.mean(cr):>10.1f}%")
    print(f"\n  fastest observed  {peak_all:.0f} mm/s ({peak_all/MM_PER_INCH:.1f} ips)")
    print(f"  datasheet limit   200 ips (Office mode), 650 ips (HP mode)")
    print(f"  utilisation       {100*peak_all/MM_PER_INCH/200:.1f}% of Office mode")
    print(f"  mean cross-axis   {100*np.mean(cross):.1f}% (sd {100*np.std(cross):.1f}%)")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    kmax = max(k for _, k in dist)
    for (c, k), reps in sorted(dist.items()):
        if k != kmax:
            continue
        _, d, ps = reps[0]
        mag = np.hypot(d["dx"], d["dy"]).to_numpy()
        t = (d["t_us"].to_numpy() - d["t_us"].iloc[0]) * 1e-6
        v = mag * (MM_PER_INCH / c) / ps
        axs[0].plot(t * 1000, v, lw=1.1, color=COLS.get(c, "#666"),
                    alpha=0.85, label=f"{c} CPI")
    axs[0].set_xlabel("time (ms)")
    axs[0].set_ylabel("velocity (mm/s)")
    axs[0].set_title(f"Velocity profile, {kmax} mm slide\n(first repeat, all resolutions)")
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
    for c, reps in sorted(still.items()):
        for _, d, _ in reps:
            S.append(d["squal"].mean()); SH.append(d["shutter"].mean()); lab.append("still")
    for (c, k), reps in sorted(dist.items()):
        for _, d, _ in reps:
            S.append(d["squal"].mean()); SH.append(d["shutter"].mean()); lab.append("slide")
    S, SH = np.array(S), np.array(SH)
    lab = np.array(lab)
    if len(S) < 3:
        return
    r = np.corrcoef(S, SH)[0, 1]
    print("\n=== OPTICAL CONDITIONS ===")
    print(f"  captures             {len(S)}")
    print(f"  corr(SQUAL, shutter) = {r:+.3f}")
    print(f"  SQUAL   {S.min():.0f} - {S.max():.0f}")
    print(f"  shutter {SH.min():.0f} - {SH.max():.0f}")
    for g in ("still", "slide"):
        m = lab == g
        if m.any():
            print(f"    {g:>6}: SQUAL {S[m].mean():6.1f} +/- {S[m].std():<5.1f}  "
                  f"shutter {SH[m].mean():8.0f} +/- {SH[m].std():<7.0f}")
    print("\n  Groups separating in this plane indicates the runs were not all")
    print("  on the same surface; treat cross-group comparisons with care.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g, col, mk in (("still", "#1f6feb", "o"), ("slide", "#d1242f", "s")):
        m = lab == g
        if m.any():
            ax.plot(SH[m], S[m], mk, color=col, ms=8, alpha=0.75, label=g)
    ax.set_xlabel("shutter (exposure)")
    ax.set_ylabel("SQUAL (features resolved)")
    ax.set_title(f"Optical operating conditions across {len(S)} captures\ncorr = {r:+.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "FIG_S5_optical.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Slip sensor results figures")
    ap.add_argument("folder")
    ap.add_argument("--outdir", default="slip_figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    still, dist = load(args.folder)
    if not still and not dist:
        sys.exit(f"No recognised captures in {args.folder}.\n"
                 "Expected noise-still-<cpi>.csv and dist-<mm>mm-<cpi>.csv")
    ns=sum(len(v) for v in still.values()); nd=sum(len(v) for v in dist.values())
    print(f"Loaded {ns} still + {nd} slide captures "
          f"({len(still)} still conditions, {len(dist)} slide conditions)")

    fig_timing(still, dist, args.outdir)
    fig_noise(still, args.outdir)
    fig_calibration(dist, args.outdir)
    fig_velocity(dist, args.outdir)
    fig_optical(still, dist, args.outdir)

    rows = []
    for c, reps in sorted(still.items()):
        for name, d, ps in reps:
            st = stats(d, c, ps)
            st.update(file=name, run="still", cpi=c, known_mm=np.nan,
                      poll_us=round(ps*1e6))
            rows.append(st)
    for (c, k), reps in sorted(dist.items()):
        for name, d, ps in reps:
            st = stats(d, c, ps)
            st.update(file=name, run="slide", cpi=c, known_mm=k,
                      poll_us=round(ps*1e6))
            rows.append(st)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "slip_summary.csv"), index=False)

    print(f"\nFigures and slip_summary.csv written to '{args.outdir}/'")


if __name__ == "__main__":
    main()