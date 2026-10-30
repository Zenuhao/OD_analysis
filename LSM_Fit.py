import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

DEFAULT_EXCEL_FILE = "data.xlsx"
SHOW_PLOT = True
OUTPUT_DIR = Path.home() / "Desktop" / "Python" / "Results"

Label = np.array([
"Scmr",
"Stdt",
"Syfp",
"Snp",
"Ctdt",
"Cyfp",
"Cnp",
"Rcmr",
"Rtdt",
"Ryfp",
"Rnp"
])

Y_NAMES = np.array(["S_tdt_1", "S_cmr_1", "C_tdt_1", "R_cmr_1", "R_tdt_1", "C_cmr_1", "S_tdtR_yfp_1", "R_tdtS_yfp_1",
            "S_tdt_2", "S_cmr_2", "C_tdt_2", "R_cmr_2", "R_tdt_2", "C_cmr_2", "S_tdtR_yfp_2", "R_tdtS_yfp_2",
            "S_tdt_3", "S_cmr_3", "C_tdt_3", "R_cmr_3", "R_tdt_3", "C_cmr_3", "S_tdtC_yfp", "C_tdtS_yfp",
            "S_yfp_1", "S_np_1", "C_yfp_1", "C_np_1", "R_yfp_1", "R_np_1", "R_tdtC_yfp", "C_tdtR_yfp",
            "S_yfp_2", "S_np_2", "C_yfp_2", "C_np_2", "R_yfp_2", "R_np_2", "S_tdtS_yfp", "C_tdtC_yfp",
            "S_yfp_3", "S_np_3", "C_yfp_3", "C_np_3", "R_yfp_3", "R_np_3", "R_tdtR_yfp"
            ])

RANGES = np.array([
    (12,21), (9,22), (16,27), (8,18), (7,20), (None, None), (None, None), (None, None),
    (6,20), (6,21), (15,24), (12,23), (6,24), (None, None), (None, None), (None, None),
    (10,20), (4,18), (20,28), (10,16), (6,19), (None, None), (None, None), (None, None),
    (5,16), (3,11), (16,26), (6,11), (5,12), (0,13), (None, None), (None, None),
    (3,12), (3,12), (7,17), (6,12), (7,17), (6,13), (None, None), (None, None),
    (5,14), (7,15), (20,28), (13,23), (3,13), (4,13), (None, None)]
    ,dtype=object)

def processing(df):
    x = df.iloc[1,1:].to_numpy(dtype=float)
    y = (df.iloc[3:50,1:] - df.iloc[50,1:]).to_numpy(dtype=float)

    if len(x) < 3:
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

    n_series, n_points = y.shape

    bounds = pd.DataFrame(ranges, columns=["start", "end"])
 
    present = bounds.notna().to_numpy()
  
    active = present.all(axis=1)

    numeric_bounds = bounds.apply(pd.to_numeric, errors="raise").fillna(0).to_numpy(dtype=float)

    start, end = numeric_bounds.T

    positive = np.isfinite(x)[None, :] & np.isfinite(y) & (y > 0)

    start = np.maximum(start, 1).astype(int)
    end = end.astype(int)

    log_y = np.full_like(y, np.nan)
    np.log(y,out=log_y, where=(y>0) & np.isfinite(y))

    # 元配列を削除・詰め直さず、表示される点の番号で選択します。
    visible_number = np.cumsum(positive, axis=1)
    mask = (
        active[:, None]
        & positive
        & (visible_number >= start[:, None])
        & (visible_number <= end[:, None])
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
            visible_number = np.cumsum(positive)
            used_numbers = visible_number[mask[i]]
            ax.text(0.03, 0.97,
                    f"R^2 (log y) = {r2_text}\n"
                    f"Visible points: {used_numbers[0]} to {used_numbers[-1]} (1-based)",
                    transform=ax.transAxes, va="top", fontsize=9)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(str(name))
        ax.set_yscale("log")
        ax.set_title(f"Exponential Fit of {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        filename = f"fitting_{name}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

def main() -> None:
    excel_file = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXCEL_FILE)

    df = pd.read_excel(excel_file, header=None, index_col=None)
    x, y = processing(df)

    a, b, r_squared, mask = lsm_fit_with_ranges(x, y)

    results = pd.DataFrame({
        "series": Y_NAMES,
        "a": a,
        "b": b,
        "R_squared_log": r_squared, 
        "n_valid": mask.sum(axis=1)})

    make_figs(x, y, Y_NAMES, a, b, r_squared, mask)
    print(f"Figures saved to: {OUTPUT_DIR}")

    for i, name in enumerate(Y_NAMES):
        print(f"\n{name}")

        used_indices = np.flatnonzero(mask[i])

        if used_indices.size == 0:
            print(f"{name}: 使用点が0点です。RANGES={RANGES[i]}")
            continue

        used_times = x[used_indices]
        positive = np.isfinite(x) & np.isfinite(y[i]) & (y[i] > 0)
        visible_number = np.cumsum(positive)
        valid = ~np.isnan(a) & ~np.isnan(b)

        a_valid = a[valid]
        b_valid = b[valid]

        print("a:", a_valid)
        print("b:", b_valid)
    

    Stdt = (a_valid[0] + a_valid[5] + a_valid[10]) / 3
    Scmr = (a_valid[1] + a_valid[6] + a_valid[11]) / 3
    Ctdt = (a_valid[2] + a_valid[7] + a_valid[12]) / 3
    Rcmr = (a_valid[3] + a_valid[8] + a_valid[13]) / 3
    Rtdt = (a_valid[4] + a_valid[9] + a_valid[14]) / 3
    Syfp = (a_valid[15] + a_valid[21] + a_valid[27]) / 3
    Snp = (a_valid[16] + a_valid[22] + a_valid[28]) / 3
    Cyfp = (a_valid[17] + a_valid[23] + a_valid[29]) / 3
    Cnp = (a_valid[18] + a_valid[24] + a_valid[30]) / 3
    Ryfp = (a_valid[19] + a_valid[25] + a_valid[31]) / 3
    Rnp = (a_valid[20] + a_valid[26] + a_valid[32]) / 3

    Mean = np.array([Stdt, Scmr, Ctdt, Rcmr, Rtdt, Syfp, Snp, Cyfp, Cnp, Ryfp, Rnp])

    Stdt_var = ((a_valid[0] - Stdt)**2 + (a_valid[5] - Stdt)**2 + (a_valid[10] - Stdt)**2) / 3
    Scmr_var = ((a_valid[1] - Scmr)**2 + (a_valid[6] - Scmr)**2 + (a_valid[11] - Scmr)**2) / 3
    Ctdt_var = ((a_valid[2] - Ctdt)**2 + (a_valid[7] - Ctdt)**2 + (a_valid[12] - Ctdt)**2) / 3
    Rcmr_var = ((a_valid[3] - Rcmr)**2 + (a_valid[8] - Rcmr)**2 + (a_valid[13] - Rcmr)**2) / 3
    Rtdt_var = ((a_valid[4] - Rtdt)**2 + (a_valid[9] - Rtdt)**2 + (a_valid[14] - Rtdt)**2) / 3
    Syfp_var = ((a_valid[15] - Syfp)**2 + (a_valid[21] - Syfp)**2 + (a_valid[27] - Syfp)**2) / 3
    Snp_var = ((a_valid[16] - Snp)**2 + (a_valid[22] - Snp)**2 + (a_valid[28] - Snp)**2) / 3
    Cyfp_var = ((a_valid[17] - Cyfp)**2 + (a_valid[23] - Cyfp)**2 + (a_valid[29] - Cyfp)**2) / 3
    Cnp_var = ((a_valid[18] - Cnp)**2 + (a_valid[24] - Cnp)**2 + (a_valid[30] - Cnp)**2) / 3
    Ryfp_var = ((a_valid[19] - Ryfp)**2 + (a_valid[25] - Ryfp)**2 + (a_valid[31] - Ryfp)**2) / 3
    Rnp_var = ((a_valid[20] - Rnp)**2 + (a_valid[26] - Rnp)**2 + (a_valid[32] - Rnp)**2) / 3

    Std = np.sqrt(np.array([Stdt_var, Scmr_var, Ctdt_var, Rcmr_var, Rtdt_var, Syfp_var, Snp_var, Cyfp_var, Cnp_var, Ryfp_var, Rnp_var]))

    Dbt = np.log(2) / Mean

    plt.subplot(1,2,1)
    plt.errorbar(Label, Mean, yerr = Std, capsize=5, fmt='o', markersize=5, ecolor='blue', markeredgecolor = "blue", color='b')
    plt.xlabel("Strain", fontsize = 24)
    plt.ylabel("Growth Rate [1/s]", fontsize = 24)

    plt.subplot(1,2,2)
    plt.scatter(Label, Dbt, color='b')
    plt.xlabel("Strain", fontsize = 24)
    plt.ylabel("Doubling Time [s]", fontsize = 24)

    plt.show()

if __name__ == "__main__":
    main()