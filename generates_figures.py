import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.special import spherical_jn
from scipy.optimize import brentq

# Use 'Agg' backend to avoid display errors on headless servers or notebooks
matplotlib.use('Agg')

"""
Generates all figures for the article:
"Blackbody Radiation in Spherical Cavities: A Theoretical Study of the Discrete Spectrum"
[Code blinded for peer review]

Generated figures (all in natural units, hbar = c = kB = 1):
  fig1_triangular_region.png   -> Fig. 1: Geometric interpretation of mode counting
  fig2_discrete_regime.png     -> Fig. 2: Discrete regime, n,l <= 5, T = 35
  fig3_intermediate_regime.png -> Fig. 3: Intermediate regime, n,l <= 20, T = 35
  fig4_semiclassical_regime.png-> Fig. 4: Semiclassical regime, n,l <= 200, T = 35
  fig5_convergence.png         -> Fig. 5: Convergence of U_numerical -> U_analytical

Requirements:
    pip install numpy scipy matplotlib pandas
"""

# ----------------------------------------------------------------------
# 1. Numerical Core: Spherical Bessel zeros and mode energies
# ----------------------------------------------------------------------

def get_spherical_bessel_zeros(l: int, n_max: int) -> list:
    """
    Finds the first 'n_max' positive zeros of the spherical Bessel function j_l(x).
    It uses sign scanning followed by root refinement via Brent's method.
    """
    zeros = []
    x = l + 0.1
    dx = 0.5
    # The upper limit scales with n_max (approx. l + n_max * pi)
    x_max = l + n_max * np.pi * 1.3 + 50 
    
    while len(zeros) < n_max and x < x_max:
        if spherical_jn(l, x) * spherical_jn(l, x + dx) < 0:
            zero = brentq(lambda z: spherical_jn(l, z), x, x + dx)
            zeros.append(zero)
            x += dx
        else:
            # Smaller step if no sign change is detected
            x += dx / 5
            
    return zeros

def get_multiplet_energy(omega: float, T: float, l: int) -> float:
    """Returns <E_{n,l}> * (2l+1) in natural units (hbar = c = kB = 1)."""
    return (2 * l + 1) * omega / (np.exp(omega / T) - 1)

def generate_spectrum_table(T: float, R: float, n_max: int, l_max: int) -> pd.DataFrame:
    """Generates a table with (l, n, omega, energy) for all modes up to n_max, l_max."""
    rows = []
    for l in range(l_max + 1):
        bessel_zeros = get_spherical_bessel_zeros(l, n_max)
        for n, zero in enumerate(bessel_zeros, start=1):
            omega = zero / R
            energy = get_multiplet_energy(omega, T, l)
            rows.append([l, n, omega, energy])
            
    return pd.DataFrame(rows, columns=["l", "n", "omega", "energy"])

def get_analytical_energy(T: float, R: float = 1.0) -> float:
    """U_analytical(T) = (8 pi / 15) * T^4 for R = 1 in natural units."""
    return (8 * np.pi / 15) * (T ** 4)

# ----------------------------------------------------------------------
# 2. Figure 1: Schematic diagram of the triangular region
# ----------------------------------------------------------------------

def generate_fig1_triangular_region(omega: float = 60.0, R: float = 1.0, c: float = 1.0, out_file: str = "fig1_triangular_region.png"):
    """Reproduces the geometric interpretation of mode counting in the (n, l) plane."""
    rho = R * omega / (np.pi * c)  # Condition: n + l/2 <= rho

    l_vals = np.arange(0, int(2 * rho) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Admissible (n, l) grid points
    for l in l_vals:
        n_max_l = rho - l / 2
        if n_max_l < 0:
            continue
        n_vals = np.arange(0, int(np.floor(n_max_l)) + 1)
        ax.scatter(n_vals, [l] * len(n_vals), color="black", s=15, zorder=3)

    # Boundary line and shaded region
    l_line = np.linspace(0, 2 * rho, 200)
    n_line = rho - l_line / 2
    ax.plot(n_line, l_line, "r--", label=r"$n + \ell/2 = R\omega/\pi c$")

    poly_pts = [(0, 0), (rho, 0), (0, 2 * rho)]
    ax.add_patch(Polygon(poly_pts, closed=True, facecolor="steelblue", alpha=0.25, zorder=1))

    ax.set_xlim(0, rho * 1.15)
    ax.set_ylim(0, 2 * rho * 1.15)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\ell$")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Geometric interpretation of mode counting")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")

# ----------------------------------------------------------------------
# 3. Figures 2, 3, 4: Spectral distributions
# ----------------------------------------------------------------------

def plot_spectrum(df: pd.DataFrame, title: str, out_file: str, markersize: float = 3, linewidth: float = 0.6):
    """Plots the multiplet thermal energy ordered by frequency."""
    df_sorted = df.sort_values(by="omega")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df_sorted["omega"], df_sorted["energy"],
        color="blue", marker="o", markersize=markersize, linewidth=linewidth,
        label=r"$\langle E_{n,\ell} \rangle \cdot (2\ell + 1)$",
    )
    ax.set_xlabel(r"$\omega_{n,\ell}$")
    ax.set_ylabel(r"$\langle E_{n,\ell} \rangle \cdot (2\ell + 1)$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")

def generate_fig2_discrete_regime(T: float = 35, R: float = 1.0, cutoff: int = 5) -> pd.DataFrame:
    df = generate_spectrum_table(T, R, cutoff, cutoff)
    plot_spectrum(df, f"Discrete regime ($n,\\ell \\leq {cutoff}$, $T={T}$)", "fig2_discrete_regime.png", markersize=5, linewidth=1.0)
    return df

def generate_fig3_intermediate_regime(T: float = 35, R: float = 1.0, cutoff: int = 20) -> pd.DataFrame:
    df = generate_spectrum_table(T, R, cutoff, cutoff)
    plot_spectrum(df, f"Intermediate regime ($n,\\ell \\leq {cutoff}$, $T={T}$)", "fig3_intermediate_regime.png", markersize=3, linewidth=0.8)
    return df

def generate_fig4_semiclassical_regime(T: float = 35, R: float = 1.0, cutoff: int = 200) -> pd.DataFrame:
    df = generate_spectrum_table(T, R, cutoff, cutoff)
    plot_spectrum(df, f"Semiclassical regime ($n,\\ell \\leq {cutoff}$, $T={T}$)", "fig4_semiclassical_regime.png", markersize=1.5, linewidth=0.3)
    return df

# ----------------------------------------------------------------------
# 4. Figure 5: Convergence to the analytical limit
# ----------------------------------------------------------------------

def generate_fig5_convergence(full_df: pd.DataFrame, T: float, R: float, cutoffs: list, out_file: str = "fig5_convergence.png"):
    """
    Reuses the largest calculated table. For each cutoff 'K', it sums the energy 
    of modes with n<=K and l<=K, comparing it with the analytical value.
    """
    u_analytical = get_analytical_energy(T, R)
    results = []
    
    for K in cutoffs:
        subset = full_df[(full_df["n"] <= K) & (full_df["l"] <= K)]
        u_numerical = subset["energy"].sum()
        relative_error = abs(u_numerical - u_analytical) / u_analytical
        results.append((K, u_numerical, relative_error))

    df_conv = pd.DataFrame(results, columns=["cutoff", "U_numerical", "relative_error"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_conv["cutoff"], df_conv["relative_error"], "o-", color="darkred")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cutoff ($n_{\max} = \ell_{\max}$)")
    ax.set_ylabel(r"Relative error $|U_{\mathrm{num}} - U_{\mathrm{analytical}}| / U_{\mathrm{analytical}}$")
    ax.set_title(f"Convergence of total energy (T = {T}, R = {R})")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[Generated] {out_file}")
    
    return df_conv

# ----------------------------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR = "article_figures"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(OUT_DIR)

    T = 35.0
    R = 1.0

    print("Generating Figure 1 (Schematic diagram)...")
    generate_fig1_triangular_region(omega=60.0, R=R)

    print("Generating Figure 2 (Discrete regime, n,l <= 5)...")
    generate_fig2_discrete_regime(T=T, R=R, cutoff=5)

    print("Generating Figure 3 (Intermediate regime, n,l <= 20)...")
    generate_fig3_intermediate_regime(T=T, R=R, cutoff=20)

    print("Generating Figure 4 (Semiclassical regime, n,l <= 200)...")
    print("  (This step takes the longest: computing ~40,000 modes)")
    df_200 = generate_fig4_semiclassical_regime(T=T, R=R, cutoff=200)

    print("Generating Figure 5 (Convergence, reusing the n,l <= 200 table)...")
    cutoffs_list = [5, 10, 20, 35, 50, 75, 100, 150, 200]
    df_conv = generate_fig5_convergence(df_200, T=T, R=R, cutoffs=cutoffs_list)
    
    print("\nConvergence Table:")
    print(df_conv.to_string(index=False))

    print(f"\nAll figures successfully saved in: {os.getcwd()}")
