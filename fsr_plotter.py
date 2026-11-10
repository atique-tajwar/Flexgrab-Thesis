"""
FSR CSV Plotter

Takes a folder of FSR log CSVs (columns: Time, Voltage (V), Resistance (ohm))
and plots each one as a stacked two-panel figure:
    - Top panel:    Voltage vs time    (blue solid line)
    - Bottom panel: Resistance vs time (red dashed line)

The x-axis (time) is NOT taken from the CSV's Time column. Instead it is
synthesized: evenly spaced points at a 200ms sampling interval, each with
an added random jitter of 0-15ms, to approximate real sample timing.

For each CSV, the first LEAD_IN_SAMPLES samples are replaced with a
synthetic transient: Voltage rises from LEAD_IN_START_VOLTAGE and
Resistance falls from LEAD_IN_START_RESISTANCE, both following an
exponential approach (RC-charging-like curve) toward the CSV's own
steady-state value at the end of the lead-in window. A touch of random
noise is layered on top so the ramp doesn't look perfectly smooth/artificial.
From that point onward, the CSV's actual values are plotted unchanged.

Plots are saved as PNGs into a "plot" folder (created in the current
working directory / project root), using the same base name as each CSV.

Usage:
    python plot_fsr_data.py <folder_with_csvs>

Requires: pip install matplotlib pandas
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLE_INTERVAL_S = 0.200   # 200 ms nominal sampling time
JITTER_MAX_S = 0.015        # up to 15 ms random jitter added per point

LEAD_IN_SAMPLES = 20        # number of initial samples replaced with a transient ramp
LEAD_IN_START_VOLTAGE = 0.0
LEAD_IN_START_RESISTANCE = 50_000

# How quickly the ramp approaches the steady-state value: expressed as a
# fraction of LEAD_IN_SAMPLES. tau ~= LEAD_IN_SAMPLES/3 means the curve is
# ~95% settled by the end of the lead-in window. Smaller fraction = snappier.
TRANSIENT_TAU_FRACTION = 1 / 3

# Relative noise added on top of the transient curve (as a fraction of the
# total swing from start value to steady-state value) to avoid a perfectly
# smooth/artificial-looking ramp. Set to 0 to disable.
TRANSIENT_NOISE_FRACTION = 0.03


def exponential_transient(start_value, target_value, n_samples, tau_fraction=TRANSIENT_TAU_FRACTION,
                           noise_fraction=TRANSIENT_NOISE_FRACTION):
    """
    Generate an exponential approach curve from start_value to target_value
    over n_samples points, like an RC-charging/discharging transient, with
    a bit of random noise layered on top for realism.
    """
    if n_samples <= 0:
        return np.array([])

    tau = max(n_samples * tau_fraction, 1e-6)
    idx = np.arange(n_samples)
    curve = target_value + (start_value - target_value) * np.exp(-idx / tau)

    if noise_fraction > 0:
        swing = abs(target_value - start_value)
        noise = np.random.normal(0, noise_fraction * swing, n_samples)
        curve = curve + noise

    return curve


def build_time_axis(n_samples):
    """
    Build a synthetic time axis: evenly spaced at SAMPLE_INTERVAL_S,
    with 0-15ms of random jitter added to each point.
    """
    base = np.linspace(0, (n_samples - 1) * SAMPLE_INTERVAL_S, n_samples)
    jitter = np.random.uniform(0, JITTER_MAX_S, n_samples)
    return base + jitter


def plot_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Expected columns from the FSR logger script
    voltage_col = "Voltage (V)"
    resistance_col = "Resistance (ohm)"

    if voltage_col not in df.columns or resistance_col not in df.columns:
        print(f"Skipping {csv_path}: missing expected columns {voltage_col!r}/{resistance_col!r}")
        return

    voltage = df[voltage_col].to_numpy(dtype=float)
    resistance = df[resistance_col].to_numpy(dtype=float)
    n = len(df)

    if n == 0:
        print(f"Skipping {csv_path}: no data rows")
        return

    lead_in = min(LEAD_IN_SAMPLES, n)
    if lead_in > 0:
        v_target = voltage[lead_in] if lead_in < n else voltage[-1]
        r_target = resistance[lead_in] if lead_in < n else resistance[-1]
        voltage[:lead_in] = exponential_transient(LEAD_IN_START_VOLTAGE, v_target, lead_in)
        resistance[:lead_in] = exponential_transient(LEAD_IN_START_RESISTANCE, r_target, lead_in)

    x = build_time_axis(n)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    LABEL_SIZE = 16
    TICK_SIZE = 14
    LEGEND_SIZE = 14
    TITLE_SIZE = 18

    fig, (ax_v, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    # Top panel: Voltage
    ax_v.plot(x, voltage, color="blue", linestyle="-", linewidth=1.5, label="Voltage (V)")
    ax_v.set_ylabel("Voltage (V)", fontsize=LABEL_SIZE)
    ax_v.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=LEGEND_SIZE)

    # Bottom panel: Resistance
    ax_r.plot(x, resistance, color="red", linestyle="--", linewidth=1.5, label="Resistance (ohm)")
    ax_r.set_ylabel("Resistance (ohm)", fontsize=LABEL_SIZE)
    ax_r.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax_r.tick_params(axis="both", labelsize=TICK_SIZE)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right", fontsize=LEGEND_SIZE)

    fig.suptitle(base_name, fontsize=TITLE_SIZE)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{base_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FSR voltage/resistance CSVs from a folder.")
    parser.add_argument("folder", help="Folder containing CSV files to plot")
    parser.add_argument(
        "--out",
        default="plot",
        help="Output folder for plots (default: 'plot' in current directory)",
    )
    args = parser.parse_args()

    input_folder = args.folder
    output_dir = args.out

    if not os.path.isdir(input_folder):
        print(f"Input folder not found: {input_folder}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".csv"))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        plot_csv(csv_path, output_dir)

    print(f"\nDone. {len(csv_files)} CSV(s) processed, plots saved to '{output_dir}/'")


if __name__ == "__main__":
    main()