import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

DEFAULT_EXCEL_FILE = "data.xlsx"
SHOW_PLOT = True
OUTPUT_DIR = Path.home() / "Desktop" / "Python" / "Results"

Y_NAMES = np.array(["S_tdt_1", "S_cmr_1", "C_tdt_1", "R_cmr_1", "R_tdt_1", "C_cmr_1", "S_tdtR_yfp_1", "R_tdtS_yfp_1",
            "S_tdt_2", "S_cmr_2", "C_tdt_2", "R_cmr_2", "R_tdt_2", "C_cmr_2", "S_tdtR_yfp_2", "R_tdtS_yfp_2",
            "S_tdt_3", "S_cmr_3", "C_tdt_3", "R_cmr_3", "R_tdt_3", "C_cmr_3", "S_tdtC_yfp", "C_tdtS_yfp",
            "S_yfp_1", "S_np_1", "C_yfp_1", "C_np_1", "R_yfp_1", "R_np_1", "R_tdtC_yfp", "C_tdtR_yfp",
            "S_yfp_2", "S_np_2", "C_yfp_2", "C_np_2", "R_yfp_2", "R_np_2", "S_tdtS_yfp", "C_tdtC_yfp",
            "S_yfp_3", "S_np_3", "C_yfp_3", "C_np_3", "R_yfp_3", "R_np_3", "R_tdtR_yfp"
            ])

RANGES = np.array([
    (10,17), (7,22), (16,27), (4,23), (4,23), (None, None), (None, None), (None, None),
    (6,20), (4,21), (14,27), (7,29), (5,24), (None, None), (None, None), (None, None),
    (13,24), (2,18), (17,29), (5,19), (2,19), (None, None), (None, None), (None, None),
    (5,20), (0,13), (15,27), (5,15), (1,16), (0, 14), (None, None), (None, None),
    (3,15), (1,12), (10,17), (6,18), (2,17), (4, 13), (None, None), (None, None),
    (5,20), (6,15), (16,28), (13,23), (2,17), (4, 14), (None, None)]
    ,dtype=object)

def processing(df):
    x = df.iloc[1,1:].to_numpy(dtype=float)
    y = (df.iloc[3:50,1:] - df.iloc[50,1:]).to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Numerical data should be at least 3 points")

    if np.min(x) == np.max(x):
        raise ValueError("All x are same values")
    return x, y

"""
def lsm_fit(x, y):
A = np.column_stack((x, np.ones_like(x)))
coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
A, coefficients = lsm_fit(x,y)
y_preds = (A @ coefficients).T
return coefficients, y_preds
"""

def lsm_fit_with_ranges(x,y, ranges=RANGES):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 2:
        raise ValueError("x must be 1D and y must be 2D.")
    n_series, n_points = y.shape

    if x.size != n_points:
        raise ValueError("The number of time points in x and y must match")

    if n_points < 3:
        raise ValueError("At least 3 time points are required.")

    bounds = pd.DataFrame(ranges, columns=["start", "end"])

    if len(bounds) != n_series:
        raise ValueError("The number of ranges must match the number of series.")

    present = bounds.notna().to_numpy()

    if np.any(present[:, 0] != present[:, 1]):
        raise ValueError("Each range must have two bounds or be (None, None).")
    active = present.all(axis=1)

    numeric_bounds = bounds.apply(pd.to_numeric, errors="raise").fillna(0).to_numpy(dtype=float)

    if np.any(~np.isfinite(numeric_bounds)):
        raise ValueError("Range bounds must be finite integers.")

    if np.any(numeric_bounds != np.floor(numeric_bounds)):
        raise ValueError("Range bounds must be integers.")

    start, end = numeric_bounds.T

    if np.any(active & ((start < 0) | (end > n_points) | (start >= end))):
        raise ValueError("Ranges must satisfy 0 <= start < end <= number of time points.")

    start = start.astype(int)
    end = end.astype(int)

    log_y = np.full_like(y, np.nan)
    np.log(y,out=log_y, where=(y>0) & np.isfinite(y))

    indices = np.arange(n_points)

    mask = (
        active[:, None]
        & (indices[None, :] >= start[:, None])
        & (indices[None, :] < end[:, None])
        & np.isfinite(x)[None, :]
        & np.isfinite(log_y)
    )

    n = mask.sum(axis=1)

    x_mean = np.where(mask, x[None, :], 0).sum(axis=1) / np.maximum(n,1)
    log_y_mean = np.where(mask, log_y, 0).sum(axis=1) / np.maximum(n,1)

    dx = np.where(mask, x[None,:] - x_mean[:,None], 0)
    dy = np.where(mask, log_y - log_y_mean[:,None], 0)

    sxx = np.sum(dx**2, axis=1)
    sxy = np.sum(dx*dy, axis=1)
    syy = np.sum(dy**2, axis=1)
    valid = active & (n>=3) & (sxx>0)

    a = np.full(n_series, np.nan)
    np.divide(sxy, sxx, out=a, where=valid)
    b = np.where(valid, log_y_mean - a*x_mean, np.nan)

    r_squared = np.full(n_series, np.nan)
    np.divide(sxy**2, sxx*syy, out=r_squared, where=valid & (syy>0))

    return a, b, r_squared, mask

def make_figs(x, y, y_names, a, b, r_squared, mask, output_dir=OUTPUT_DIR, show_plot=SHOW_PLOT):

    if len(y_names) != len(y):
        raise ValueError("The number of series names must match the number of series.")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(y_names):
        fig, ax = plt.subplots(figsize=(8,5))
        positive = np.isfinite(x) & np.isfinite(y[i]) & (y[i]>0)
        ax.scatter(x[positive], y[i,positive], s=40, alpha=0.8, label=f"Data (n={positive.sum()})")

        if np.isfinite(a[i]) and np.isfinite(b[i]):
            fit_x = x[mask[i]]
            ax.scatter(fit_x, y[i, mask[i]], s=70, facecolors="none", 
                       edgecolors= "tab:orange", label=f"Used for fit (n={fit_x.size})")

            line_x = np.linspace(fit_x.min(), fit_x.max(), 200)
            line_y = np.exp(a[i] * line_x + b[i])

            ax.plot(line_x, line_y, color="tab:red", label=f"Fit: y= exp({a[i]:.4g}t {b[i]:+.4g})")
            r2_text = f"{r_squared[i]:.4f}" if np.isfinite(r_squared[i]) else "undefined"
            ax.text(0.03, 0.97, f"R^2 (log y) = {r2_text}",
            transform=ax.transAxes, va="top")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(str(name))
        ax.set_yscale("log")
        ax.set_title(f"Exponential Fit of {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        filename = f"fitting_{name}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

def main() -> None:
    excel_file = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXCEL_FILE)
    if not excel_file.is_file():
        raise FileNotFoundError(f"Excel file cannot be found: {excel_file}\n")

    df = pd.read_excel(excel_file, header=None, index_col=None)
    x, y = processing(df)
    if len(Y_NAMES) != y.shape[0]:
        raise ValueError("The number of series names must match the number of series.")

    a, b, r_squared, mask = lsm_fit_with_ranges(x, y)

    results = pd.DataFrame({
        "series": Y_NAMES,
        "a": a,
        "b": b,
        "R_squared_log": r_squared, 
        "n_valid": mask.sum(axis=1)})

    print(results.to_string(index=False))
    make_figs(x, y, Y_NAMES, a, b, r_squared, mask)
    print(f"Figures saved to: {OUTPUT_DIR}")


    for i, name in enumerate(Y_NAMES):
        print(f"\n{name}")

        if not (np.isfinite(a[i]) and np.isfinite(b[i])):
            print("No Fitting")
            continue
    
        used_indices = np.flatnonzero(mask[i])
        used_times = x[used_indices]

        print(f"  インデックス: {used_indices.tolist()}")
        print(f"  使用点数: {used_indices.size}")
        print(
            f"  時間範囲: "
            f"{used_times.min():g} ～ {used_times.max():g} s"
        )
        
if __name__ == "__main__":
    main()