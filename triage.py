"""
PAW3395 Capture Triage
======================

Detects the actual sampling interval of every capture and flags those that do
not match the intended rate. A mixture of poll rates in one dataset silently
corrupts every velocity-derived quantity, because velocity is counts divided
by the sampling interval.

Usage:
    python triage.py slip_data                        # report only
    python triage.py slip_data --expect 5000          # flag mismatches
    python triage.py slip_data --expect 5000 --move   # quarantine them

With --move, mismatched files are moved to <folder>/wrong_poll/ rather than
deleted, so nothing is lost.

Also reports rollover events, intra-file timing instability, and SQUAL, so a
capture can be rejected for optical reasons as well as timing ones.

Requires: numpy, pandas
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd


def intervals_us(t_us):
    """Sample intervals from a uint32 micros() timebase, correcting rollover."""
    dt = np.diff(np.asarray(t_us, dtype=np.int64))
    wraps = int((dt < 0).sum())
    dt = dt % (1 << 32)
    return dt, wraps


def classify(median_dt, candidates):
    """Nearest nominal rate, and the relative error to it."""
    best = min(candidates, key=lambda c: abs(median_dt - c))
    return best, 100.0 * (median_dt - best) / best


def main():
    ap = argparse.ArgumentParser(description="Triage PAW3395 captures by poll rate")
    ap.add_argument("folder")
    ap.add_argument("--expect", type=int, default=None,
                    help="intended poll interval in us; files not matching are flagged")
    ap.add_argument("--candidates", type=int, nargs="*",
                    default=[250, 500, 1000, 2000, 5000, 10000, 20000],
                    help="nominal rates the firmware might have been left at")
    ap.add_argument("--tol", type=float, default=2.0,
                    help="percent tolerance around the expected rate")
    ap.add_argument("--move", action="store_true",
                    help="move mismatched files to <folder>/wrong_poll/")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "*.csv")))
    if not paths:
        sys.exit(f"No CSVs in {args.folder}")

    print(f"{'file':>32} | {'n':>6} | {'median us':>9} | {'nominal':>8} | "
          f"{'err %':>7} | {'jitter':>7} | {'wrap':>4} | {'SQUAL':>6}")
    print("-" * 100)

    rows = []
    for p in paths:
        b = os.path.basename(p)
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"{b:>32} | unreadable: {e}")
            continue
        if "t_us" not in d.columns or len(d) < 10:
            continue

        dt, wraps = intervals_us(d["t_us"])
        med = float(np.median(dt))
        nominal, err = classify(med, args.candidates)
        # jitter measured only within the dominant mode, so a single
        # outlier does not dominate the statistic
        core = dt[np.abs(dt - med) < 0.5 * med]
        jit = 100.0 * core.std() / med if len(core) else np.nan
        sq = float(d["squal"].mean()) if "squal" in d.columns else np.nan

        mismatch = args.expect is not None and abs(med - args.expect) > args.tol / 100 * args.expect
        flag = "  ** MISMATCH" if mismatch else ""
        rows.append(dict(file=b, path=p, n=len(d), med_us=med, nominal=nominal,
                         err=err, jitter=jit, wraps=wraps, squal=sq,
                         mismatch=mismatch))
        print(f"{b:>32} | {len(d):>6} | {med:>9.0f} | {nominal:>8} | "
              f"{err:>+6.2f}% | {jit:>6.2f}% | {wraps:>4} | {sq:>6.1f}{flag}")

    if not rows:
        sys.exit("\nNo usable captures found.")

    df = pd.DataFrame(rows)

    print("\n=== POLL RATE GROUPS ===")
    for nom, grp in df.groupby("nominal"):
        n_samp = int(grp["n"].sum())
        print(f"  {nom:>6} us : {len(grp):>2} file(s), {n_samp:>7,} samples")
        if len(grp) <= 12:
            for f in grp["file"]:
                print(f"             {f}")

    if df["wraps"].sum():
        print(f"\n  micros() rollovers detected in "
              f"{int((df['wraps'] > 0).sum())} file(s) -- corrected automatically.")

    bad_jit = df[df["jitter"] > 5]
    if len(bad_jit):
        print("\n=== UNSTABLE TIMING WITHIN FILE (jitter > 5%) ===")
        for _, r in bad_jit.iterrows():
            print(f"  {r['file']:<34} jitter {r['jitter']:.1f}%")
        print("  Likely serial backlog forcing deadline resynchronisation.")

    if args.expect is None:
        print("\nPass --expect <us> to flag mismatches, and --move to quarantine them.")
        return

    mm = df[df["mismatch"]]
    print(f"\n=== AGAINST EXPECTED {args.expect} us (+/-{args.tol}%) ===")
    print(f"  matching   : {len(df) - len(mm)} file(s)")
    print(f"  mismatched : {len(mm)} file(s)")

    if len(mm) == 0:
        print("\n  All captures consistent. Nothing to remove.")
        return

    for _, r in mm.iterrows():
        print(f"    {r['file']:<34} {r['med_us']:.0f} us (nominal {r['nominal']})")

    if not args.move:
        print("\n  Re-run with --move to quarantine these into wrong_poll/")
        return

    dest = os.path.join(args.folder, "wrong_poll")
    os.makedirs(dest, exist_ok=True)
    for _, r in mm.iterrows():
        shutil.move(r['path'], os.path.join(dest, r['file']))
    print(f"\n  Moved {len(mm)} file(s) to {dest}")
    print("  They are not deleted; re-analyse them separately with the correct")
    print("  --poll-us if you want to keep the data.")


if __name__ == "__main__":
    main()
