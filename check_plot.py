import pandas as pd, numpy as np
for f in ["noise-paper-1600.csv","noise-paper-3200.csv","noise-paper-6400.csv","noise-paper-12800.csv"]:
    d = pd.read_csv(f"slip_data/{f}")
    mag = np.hypot(d.dx, d.dy)
    print(f, np.unique(mag, return_counts=True))