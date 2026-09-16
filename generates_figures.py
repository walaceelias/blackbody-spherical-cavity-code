import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from scipy.special import spherical_jn
from scipy.optimize import brentq
import pandas as pd

# Use 'Agg' backend to avoid display errors on headless servers or notebooks
matplotlib.use("Agg")

"""
Figure generator for the article (Alternative A: scalar field), consolidated version.
[Code blinded for peer review]

Physics and structural features incorporated:
  (3) g(w) = 2 R0^3 w^2 / (3 pi c^3) [Correct Weyl's law for scalar field]
  (5) True spectral cutoff at x_nl <= x_cut, rather than a rectangular cutoff in (n,l)
  (6) Figures 2-4 display actual density (coarse-grained / stem), rather than 
      (2l+1)<E_nl> connected by lines, ensuring correct units (energy/frequency).
"""

# ----------------------------------------------------------------------
# 1. Numerical Core: Spherical Bessel zeros (scalar field, Dirichlet
#    boundary conditions), Weyl density, average energy
# ----------------------------------------------------------------------

def get_first_zero(l: int) -> float:
    """Finds the first positive zero of j_l(x) (used to decide when to truncate l)."""
    x = l + 0.1
    dx = 0.5
    while True:
        if spherical_jn(l, x) * spherical_jn(l, x + dx) < 0:
            return brentq(lambda z: spherical_jn(l, z), x, x + dx)
        x += dx / 5

def get_zeros_up_to_cutoff(l: int, x_cut: float) -> list:
    """
    Returns all positive zeros of j_l(x) where x <= x_cut.
    Implements a true spectral cutoff by including all zeros below the cutoff 
    for each l, rather than a fixed number of zeros.
    """
    zeros = []
    x = l + 0.1
    dx = 0.5
    while x < x_cut + dx:
        if spherical_jn(l, x) * spherical_jn(l, x + dx) < 0:
            zero = brentq(lambda z: spherical_jn(l, z), x, x + dx)
            if zero <= x_cut:
                zeros.append(zero)
            x += dx
        else:
            x += dx / 5
    return zeros

def generate_spectral_cutoff_table(x_cut: float, R0: float = 1.0) -> pd.DataFrame:
    """
    Generates a table (l, n, x, omega) including ALL modes with x_nl <= x_cut
    (true spectral cutoff).
    
    The loop over l breaks as soon as the FIRST zero of j_l (n=1) exceeds x_cut,
    because x_{1,l} grows monotonically with l (no higher l will contribute).
    """
    rows = []
    l = 0
    while True:
        x1 = get_first_zero(l)
        if x1 > x_cut:
            break
        zeros_l = get_zeros_up_to_cutoff(l, x_cut)
        for n, zero in enumerate(zeros_l, start=1):
            rows.append([l, n, zero])
        l += 1
        
    df = pd.DataFrame(rows, columns=["l", "n", "x"])
    df["omega"] = df["x"] / R0
    return df

def get_average_energy(omega: float, T: float) -> float:
    """<E> = hbar*omega / (e^(hbar*omega/kB T) - 1) in natural units."""
    return omega / (np.exp(omega / T) - 1)

def get_weyl_density(omega: float, R0: float = 1.0, c: float = 1.0) -> float:
    """Correct density of states (Weyl's law, scalar field): 2R0^3 w^2/(3 pi c^3)."""
    return (2 * R0 ** 3 / (3 * np.pi * c ** 3)) * omega ** 2

def get_planck_density(omega: float, T: float, R0: float = 1.0, c: float = 1.0) -> float:
    """Correct continuous spectral density: g_weyl(omega) * <E(omega,T)>."""
    return get_weyl_density(omega, R0, c) * get_average_energy(omega, T)

def get_analytical_energy(T: float, R0: float = 1.0) -> float:
    """Correct analytical total energy (scalar field): (2 pi^3/45) R0^3 T^4."""
    return (2 * np.pi ** 3 / 45) * R0 ** 3 * T ** 4

# ----------------------------------------------------------------------
# 2. Figure 1: Schematic diagram of the triangular region in the (n, l) plane
# ----------------------------------------------------------------------

def generate_fig1_triangular_region(omega: float = 60.0, R0: float = 1.0, c: float = 1.0, out_file: str = "fig1_triangular_region.png"):
    """
    Qualitative diagram (Fig. 1): Geometric interpretation of mode counting.
    The line n + l/2 = R0*omega/(pi*c) bounds the admissible triangular region.
    """
    rho = R0 * omega / (np.pi * c)  # n + l/2 <= rho

    l_vals = np.arange(0, int(2 * rho) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))

    for l in l_vals:
        n_max_l = rho - l / 2
        if n_max_l < 0:
            continue
        n_vals = np.arange(0, int(np.floor(n_max_l)) + 1)
        ax.scatter(n_vals, [l] * len(n_vals), color="black", s=15, zorder=3)

    l_line = np.linspace(0, 2 * rho, 200)
    n_line = rho - l_line / 2
    ax.plot(n_line, l_line, "r--", linewidth=1.5,
            label=r"$n + \ell/2 = R_0\omega/\pi c$")

    poly_pts = [(0, 0), (rho, 0), (0, 2 * rho)]
    ax.add_patch(Polygon(poly_pts, closed=True, facecolor="steelblue", alpha=0.25, zorder=1))

    ax.set_xlim(0, rho * 1.15)
    ax.set_ylim(0, 2 * rho * 1.15)
    ax.set_xlabel(r"$n$", fontsize=14)
    ax.set_ylabel(r"$\ell$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")

# ----------------------------------------------------------------------
# 3. Figure 2 (Sparse sector): Stem plot for individual modes
# ----------------------------------------------------------------------

def generate_fig_stem(df: pd.DataFrame, T: float, out_file: str):
    """
    For a small number of modes: Displays each mode individually using a stem plot,
    weighted by its degeneracy (2l+1)*<E_nl> = E^mult_{n,l}. 
    This emphasizes the discrete "weight" of each mode-family rather than a continuous density.
    """
    weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))
    fig, ax = plt.subplots(figsize=(10, 5))
    markerline, stemlines, baseline = ax.stem(df["omega"], weights, basefmt=" ")
    plt.setp(stemlines, linewidth=1.5, color="steelblue")
    plt.setp(markerline, markersize=7, color="darkblue")
    
    ax.set_xlabel(r"$\omega_{n,\ell}$", fontsize=14)
    ax.set_ylabel(r"$E^{\mathrm{mult}}_{n,\ell}$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes)")

# ----------------------------------------------------------------------
# 4. Figure 3 (Intermediate sector): Coarse-grained density, single panel
# ----------------------------------------------------------------------

def generate_fig_coarse_grained(df: pd.DataFrame, T: float, R0: float, delta_omega: float, out_file: str):
    """
    Constructs the binned density u_{Delta w}(w_j) = (1/Delta w) * sum E^mult_{n,l}
    over the bin, which correctly yields energy/frequency units. Overlays the
    continuous analytical curve u(w) = g_weyl(w)*<E(w,T)>.
    """
    omega_max = df["omega"].max()
    bins = np.arange(0, omega_max + delta_omega, delta_omega)
    weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))

    bin_idx = np.digitize(df["omega"], bins)
    u_binned = np.array([weights[bin_idx == i].sum() for i in range(1, len(bins))]) / delta_omega
    centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, u_binned, width=delta_omega * 0.9, color="steelblue",
           alpha=0.6, label="coarse-grained")

    omega_cont = np.linspace(0.01, omega_max, 500)
    ax.plot(omega_cont, get_planck_density(omega_cont, T, R0), color="darkred", linewidth=2,
            label="leading Weyl")

    ax.set_xlabel(r"$\omega$", fontsize=14)
    ax.set_ylabel(r"$u(\omega)$  (energy per unit frequency)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes, bin={delta_omega})")

# ----------------------------------------------------------------------
# 5. Figure 4 (Dense sector): Panel with 3 bin widths
# ----------------------------------------------------------------------

def generate_fig4_panel_deltaomega(T: float, R0: float, x_cut: float, bin_widths: tuple = (2, 5, 10), out_file: str = "fig4_panel_deltaomega.png"):
    """
    Creates a panel with a 2 (top) + 1 (bottom, double width) layout, comparing
    the coarse-grained density for three different bin widths calculated from 
    the SAME mode set.
    """
    assert len(bin_widths) == 3, "This function expects exactly 3 bin widths."
    dw1, dw2, dw3 = bin_widths

    df = generate_spectral_cutoff_table(x_cut, R0)
    weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))
    omega_max = df["omega"].max()

    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.15)
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1], sharey=ax_top_left)
    ax_bottom = fig.add_subplot(gs[1, :], sharey=ax_top_left)

    axes_and_bins = [(ax_top_left, dw1), (ax_top_right, dw2), (ax_bottom, dw3)]

    data = {}
    max_values = []
    for _, dw in axes_and_bins:
        bins = np.arange(0, omega_max + dw, dw)
        bin_idx = np.digitize(df["omega"], bins)
        u_binned = np.array([weights[bin_idx == i].sum() for i in range(1, len(bins))]) / dw
        centers = (bins[:-1] + bins[1:]) / 2
        data[dw] = (centers, u_binned)
        max_values.append(u_binned.max())

    omega_cont = np.linspace(0.01, omega_max, 500)
    analytical_curve = get_planck_density(omega_cont, T, R0)
    y_max = 1.1 * max(max(max_values), analytical_curve.max())

    bar_handle = line_handle = None
    for i, (ax, dw) in enumerate(axes_and_bins):
        centers, u_binned = data[dw]
        bar_handle = ax.bar(centers, u_binned, width=dw * 0.9, color="steelblue", alpha=0.6)
        line_handle, = ax.plot(omega_cont, analytical_curve, color="darkred", linewidth=2)
        
        ax.set_xlabel(r"$\omega$", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, y_max)
        
        # Bin width label INSIDE the panel
        ax.text(0.97, 0.95, rf"$\Delta\omega = {dw}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))

    ax_top_left.set_ylabel(r"$u(\omega)$", fontsize=13)
    ax_bottom.set_ylabel(r"$u(\omega)$", fontsize=13)
    plt.setp(ax_top_right.get_yticklabels(), visible=False)

    # Compact single legend inside the first panel
    ax_top_left.legend([bar_handle, line_handle], ["coarse-grained", "leading Weyl"],
                       loc="upper left", fontsize=11, framealpha=0.9)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {out_file}  ({len(df)} modes, x_cut={x_cut}, bins={bin_widths})")

# ----------------------------------------------------------------------
# 6. Figure 5: Convergence with true spectral cutoff
# ----------------------------------------------------------------------

def generate_fig5_spectral_convergence(T: float, R0: float, x_cutoffs: list, out_file: str):
    """
    For each spectral cutoff x_cut in `x_cutoffs`, generates the complete mode table
    where x_nl <= x_cut, sums the numerical energy, and compares it with U_analytical.
    This implements a genuine spectral cutoff, not a rectangular cutoff in (n, l).
    """
    results = []
    U_analytical = get_analytical_energy(T, R0)
    
    for x_cut in x_cutoffs:
        df = generate_spectral_cutoff_table(x_cut, R0)
        weights = (2 * df["l"] + 1) * df["omega"].apply(lambda w: get_average_energy(w, T))
        U_num = weights.sum()
        error = abs(U_num - U_analytical) / U_analytical
        results.append((x_cut, len(df), U_num, error))
        print(f"  x_cut={x_cut:6.1f}  modes={len(df):6d}  U_num={U_num:12.1f}  error={error:.4f}")

    df_conv = pd.DataFrame(results, columns=["x_cut", "n_modes", "U_numerical", "relative_error"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_conv["x_cut"], df_conv["relative_error"], "o-", color="darkred", markersize=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Spectral cutoff $x_{\mathrm{cut}}=\omega_{\max}R_0/c$", fontsize=14)
    ax.set_ylabel(r"Relative error $|U_{\rm numerical} - U_{\rm W}^{(0)}| / U_{\rm W}^{(0)}$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")
    return df_conv

# ----------------------------------------------------------------------
# 7. Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR = "figures_v2"
    os.makedirs(OUT_DIR, exist_ok=True)

    T = 35.0
    R0 = 1.0

    print("=== Figure 1 (Schematic diagram of the triangular region) ===")
    generate_fig1_triangular_region(omega=60.0, R0=R0, out_file=f"{OUT_DIR}/fig1_triangular_region.png")

    print("=== Figure 2 (Sparse sector, few modes) ===")
    df_low = generate_spectral_cutoff_table(x_cut=10, R0=R0)
    generate_fig_stem(df_low, T, out_file=f"{OUT_DIR}/fig2_sparse_sector.png")

    print("=== Figure 3 (Intermediate sector) ===")
    df_mid = generate_spectral_cutoff_table(x_cut=60, R0=R0)
    generate_fig_coarse_grained(df_mid, T, R0, delta_omega=3.0, out_file=f"{OUT_DIR}/fig3_intermediate_sector.png")

    print("=== Figure 4 (Dense sector: Panel with Delta_omega = 2, 5, 10) ===")
    generate_fig4_panel_deltaomega(T, R0, x_cut=300, bin_widths=(2, 5, 10),
                                   out_file=f"{OUT_DIR}/fig4_panel_deltaomega.png")

    print("=== Figure 5 (Convergence with proper spectral cutoff) ===")
    x_cutoffs_list = [10, 20, 40, 60, 100, 150, 200, 300, 400]
    df_conv = generate_fig5_spectral_convergence(T, R0, x_cutoffs_list, f"{OUT_DIR}/fig5_convergence.png")
    
    print("\nConvergence Table:")
    print(df_conv.to_string(index=False))

    print(f"\nU_analytical(T=35, R0=1) = {get_analytical_energy(T, R0):.1f} (Corrected, scalar field)")
    print(f"\nAll figures successfully saved in: {os.path.abspath(OUT_DIR)}")
