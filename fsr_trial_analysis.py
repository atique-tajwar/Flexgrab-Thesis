"""
FSR trial analysis
==================
Analyses runs recorded with fsr_trial_logger.py (one CSV per run, many trials).
Uses only measured data and the ESP32's own time stamps.

    python fsr_trial_analysis.py fsr_runs/pla2mm_run1.csv fsr_runs/tpu1mm_run1.csv ^
        --plate 0 --t-read 10 --outdir fsr_analysis

For each trial it:
  * finds the moment the weight lands (first sustained rise above the empty
    reading),
  * takes the reading T_READ seconds after landing (mean of a 1 s window),
  * converts it to conductance G = 1/R (micro-siemens), which is close to
    proportional to force for an FSR.
Per run it fits G = a * F^n, inverts the fit to estimate force, and reports
force error by load band, loading-direction (up vs down) bias, the trend over
the session, and drift during the long holds (trials longer than 40 s).

Outputs (in --outdir): trials_<run>.csv, summary.csv, fsr_comparison.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G_ACC = 9.81


def load_run(path):
    d = pd.read_csv(path)
    d = d[pd.to_numeric(d["t_ms"], errors="coerce").notna()].copy()
    for c in ("trial", "mass_g", "t_ms", "adc_raw", "mv", "r_ohm"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def analyse_trial(x, t_read, rise_mv, sustain):
    x = x.sort_values("t_ms")
    t = (x["t_ms"].to_numpy() - x["t_ms"].iloc[0]) / 1000.0
    mv = x["mv"].to_numpy()
    r = x["r_ohm"].to_numpy()
    base = np.median(mv[: max(5, int(0.8 / 0.05))])          # first ~0.8 s
    above = mv > base + rise_mv
    land = None
    for i in range(len(above) - sustain):
        if above[i: i + sustain].all():
            land = i
            break
    out = dict(trial=int(x["trial"].iloc[0]), mass_g=float(x["mass_g"].iloc[0]),
               base_mv=base, t_land=np.nan, mv_read=np.nan, r_read=np.nan,
               G_read=np.nan, t_avail=np.nan, long=False)
    if land is None:
        return out, None
    tl = t[land]
    out["t_land"] = tl
    out["t_avail"] = t[-1] - tl
    out["long"] = out["t_avail"] > 40
    w = (t >= tl + t_read - 0.5) & (t <= tl + t_read + 0.5)
    if w.sum() < 5:
        return out, None
    rr = r[w]
    rr = rr[rr > 0]
    if len(rr) == 0:
        return out, None
    out["mv_read"] = mv[w].mean()
    out["r_read"] = rr.mean()
    out["G_read"] = 1e6 / rr.mean()
    # drift trace in conductance after landing
    keep = (t >= tl) & (r > 0)
    trace = pd.DataFrame({"t": t[keep] - tl, "G": 1e6 / r[keep]})
    return out, trace


def fit_power(F, G):
    ok = (F > 0) & (G > 0)
    n, lna = np.polyfit(np.log(F[ok]), np.log(G[ok]), 1)
    return np.exp(lna), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--plate", type=float, default=0.0,
                    help="mass (g) permanently on the FSR, added to every trial")
    ap.add_argument("--t-read", type=float, default=10.0,
                    help="seconds after landing at which the reading is taken")
    ap.add_argument("--rise-mv", type=float, default=30.0)
    ap.add_argument("--sustain", type=int, default=10, help="samples above threshold")
    ap.add_argument("--min-mass", type=float, default=0.0,
                    help="ignore trials lighter than this (g)")
    ap.add_argument("--outdir", default="fsr_analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    summary, allres, traces = [], {}, {}
    for path in a.runs:
        name = os.path.splitext(os.path.basename(path))[0]
        d = load_run(path)
        rows, tr = [], {}
        for _, x in d.groupby("trial"):
            o, trace = analyse_trial(x, a.t_read, a.rise_mv, a.sustain)
            rows.append(o)
            if trace is not None:
                tr[o["trial"]] = trace
        df = pd.DataFrame(rows).sort_values("trial")
        df["F"] = (df["mass_g"] + a.plate) * G_ACC / 1000.0
        # loading direction relative to the previous trial
        df["dir"] = np.where(df["mass_g"].diff() > 0, "up",
                             np.where(df["mass_g"].diff() < 0, "down", "same"))
        ok = df["G_read"].notna() & (df["mass_g"] >= a.min_mass)
        A, n = fit_power(df.loc[ok, "F"].to_numpy(), df.loc[ok, "G_read"].to_numpy())
        df["F_est"] = (df["G_read"] / A) ** (1.0 / n)
        df["err_N"] = df["F_est"] - df["F"]
        df["err_pct"] = 100 * df["err_N"] / df["F"]
        logres = np.log(df.loc[ok, "G_read"]) - np.log(A * df.loc[ok, "F"] ** n)
        r2_log = 1 - logres.var() / np.log(df.loc[ok, "G_read"]).var()

        bands = [(0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 10)]
        s = dict(run=name, n_trials=len(df), n_used=int(ok.sum()),
                 n_no_landing=int(df["t_land"].isna().sum()),
                 fit=f"G = {A:.2f} * F^{n:.3f}  (uS, N)", r2_log=r2_log,
                 rms_err_N=np.sqrt(np.nanmean(df.loc[ok, "err_N"] ** 2)))
        for lo, hi in bands:
            m = ok & (df["F"] >= lo) & (df["F"] < hi)
            s[f"sd_err_pct_{lo:g}-{hi:g}N"] = df.loc[m, "err_pct"].std()
            s[f"n_{lo:g}-{hi:g}N"] = int(m.sum())
        for dname in ("up", "down"):
            m = ok & (df["dir"] == dname)
            s[f"mean_err_pct_{dname}"] = df.loc[m, "err_pct"].mean()
        # trend over the session: slope of error % vs trial number
        tt = df.loc[ok]
        s["trend_pct_per_10_trials"] = 10 * np.polyfit(tt["trial"], tt["err_pct"], 1)[0]
        # long holds: change in G from 5 s to 60 s (or end) after landing
        holds = []
        for _, row in df[df["long"]].iterrows():
            trc = tr.get(row["trial"])
            if trc is None:
                continue
            g5 = trc[(trc.t >= 4.5) & (trc.t <= 5.5)].G.mean()
            tend = min(60.0, trc.t.max())
            gend = trc[(trc.t >= tend - 1) & (trc.t <= tend)].G.mean()
            holds.append(100 * (gend - g5) / g5)
        s["hold_drift_pct_5_to_60s_mean"] = np.mean(holds) if holds else np.nan
        s["hold_drift_pct_list"] = ", ".join(f"{h:+.1f}" for h in holds)
        # short-trial drift: 5 s -> 15 s after landing
        sd = []
        for _, row in df[~df["long"] & ok].iterrows():
            trc = tr.get(row["trial"])
            if trc is None or trc.t.max() < 15:
                continue
            g5 = trc[(trc.t >= 4.5) & (trc.t <= 5.5)].G.mean()
            g15 = trc[(trc.t >= 14.5) & (trc.t <= 15.5)].G.mean()
            sd.append(100 * (g15 - g5) / g5)
        s["drift_pct_5_to_15s_median"] = np.median(sd) if sd else np.nan
        # sticky ADC codes: share of samples on the 3 most common non-zero raw codes
        raw = d.loc[d["adc_raw"] > 1, "adc_raw"].round()
        s["pct_samples_on_top3_codes"] = 100 * raw.value_counts().head(3).sum() / len(raw)
        s["top3_codes"] = ", ".join(f"{int(k)}" for k in raw.value_counts().head(3).index)
        summary.append(s)
        df.to_csv(os.path.join(a.outdir, f"trials_{name}.csv"), index=False)
        allres[name] = (df, ok, A, n)
        traces[name] = (df, tr)

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(sm.T.to_string())

    # ---------------- figure ----------------
    cols = ["#1f6feb", "#d1242f", "#2da44e", "#8250df"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    for k, (name, (df, ok, A, n)) in enumerate(allres.items()):
        c = cols[k % len(cols)]
        up = ok & (df["dir"] == "up")
        dn = ok & (df["dir"] != "up")
        ax[0, 0].plot(df.loc[up, "F"], df.loc[up, "G_read"], "^", color=c, ms=5, alpha=.8,
                      label=f"{name} (up)")
        ax[0, 0].plot(df.loc[dn, "F"], df.loc[dn, "G_read"], "v", color=c, ms=5, alpha=.5,
                      mfc="none", label=f"{name} (down)")
        Fs = np.linspace(df.loc[ok, "F"].min(), df.loc[ok, "F"].max(), 200)
        ax[0, 0].plot(Fs, A * Fs ** n, "-", color=c, lw=1.5)
        ax[0, 1].plot(df.loc[ok, "F"], df.loc[ok, "err_pct"], "o", color=c, ms=4, alpha=.7, label=name)
        ax[0, 2].plot(df.loc[ok, "trial"], df.loc[ok, "err_pct"], "o-", color=c, ms=3, lw=.6,
                      alpha=.8, label=name)
        dfl, tr = traces[name]
        for tid in dfl.loc[dfl["long"], "trial"]:
            trc = tr.get(tid)
            if trc is None:
                continue
            g5 = trc[(trc.t >= 4.5) & (trc.t <= 5.5)].G.mean()
            ax[1, 0].plot(trc.t, 100 * (trc.G / g5 - 1), color=c, lw=.9, alpha=.8)
    ax[1, 0].plot([], [], color=cols[0], label=list(allres)[0])
    if len(allres) > 1:
        ax[1, 0].plot([], [], color=cols[1], label=list(allres)[1])
    ax[0, 0].set(xlabel="Applied force F (N)", ylabel=f"Conductance 1/R at {a.t_read:g} s (µS)",
                 title="Calibration (lines: power-law fit)")
    ax[0, 1].axhline(0, color="k", lw=.8)
    ax[0, 1].set(xlabel="Applied force F (N)", ylabel="Force error (%)",
                 title="Error of the fitted calibration")
    ax[0, 2].axhline(0, color="k", lw=.8)
    ax[0, 2].set(xlabel="Trial number (order in session)", ylabel="Force error (%)",
                 title="Error vs order (trend = settling / history)")
    ax[1, 0].axhline(0, color="k", lw=.8)
    ax[1, 0].set(xlabel="Time after landing (s)", ylabel="Change in 1/R from 5 s (%)",
                 title="Long holds: drift", xlim=(0, 65))
    # raw ADC code histogram
    for k, path in enumerate(a.runs):
        d = load_run(path)
        raw = d.loc[d["adc_raw"] > 1, "adc_raw"]
        ax[1, 1].hist(raw, bins=np.arange(0, 4096, 8), color=cols[k % len(cols)], alpha=.5,
                      label=os.path.splitext(os.path.basename(path))[0])
    ax[1, 1].set(xlabel="Raw ADC code", ylabel="Samples", yscale="log",
                 title="ADC codes (spikes = sticky codes)")
    # example trial
    name0 = list(traces)[0]
    dfl, tr = traces[name0]
    ex = dfl.loc[dfl["mass_g"].sub(dfl["mass_g"].median()).abs().idxmin(), "trial"]
    d0 = load_run(a.runs[0])
    x = d0[d0["trial"] == ex]
    t = (x["t_ms"] - x["t_ms"].iloc[0]) / 1000
    ax[1, 2].plot(t, x["mv"], color=cols[0], lw=1)
    tl = dfl.loc[dfl["trial"] == ex, "t_land"].iloc[0]
    if np.isfinite(tl):
        ax[1, 2].axvline(tl, color="k", ls=":", lw=1)
        ax[1, 2].axvspan(tl + a.t_read - .5, tl + a.t_read + .5, color="orange", alpha=.3)
    ax[1, 2].set(xlabel="Time in trial (s, ESP32 clock)", ylabel="Pin voltage (mV)",
                 title=f"Example raw trial: {name0} #{ex} "
                       f"({dfl.loc[dfl['trial'] == ex, 'mass_g'].iloc[0]:g} g)")
    for axx in ax.flat:
        axx.grid(alpha=.3)
        if axx.get_legend_handles_labels()[0]:
            axx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fsr_comparison.png"), dpi=140)
    print(f"\nWrote {a.outdir}/summary.csv, trials_*.csv, fsr_comparison.png")


if __name__ == "__main__":
    main()
