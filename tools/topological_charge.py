#!/usr/bin/env python3
"""
topological_charge.py — Paper C §3.3: Explicit Computation of χ_DND

Computes the D-ND topological charge on a 2D emergence landscape:
1. Double-well potential V(Z) = Z²(1-Z)²
2. Gaussian curvature K at each point
3. Gauss-Bonnet integral: χ_DND = (1/2π) ∫∫ K dA
4. Time evolution through bifurcation → verify χ ∈ ℤ

References:
- Paper C §3.1-3.3: Topological classification via Gauss-Bonnet
- Paper C §3.2: Quantization theorem
- DND_METHOD_AXIOMS §VIII: Entropy and the plane

Author: TM3 (Claude Code)
Date: 2026-02-26
"""

import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent.parent / "papers" / "figures"
DATA_DIR = Path(__file__).parent / "data"


def dnd_potential(Z, theta_NT=1.0, lam=0.5):
    """
    D-ND double-well potential.
    V(Z, θ_NT, λ) = Z²(1-Z)² + λ·θ_NT·Z·(1-Z)

    From sim_lagrangian_v2.py — the potential landscape of emergence.
    Z ∈ [0,1]: order parameter (0 = NT state, 1 = fully emerged)
    """
    return Z**2 * (1 - Z)**2 + lam * theta_NT * Z * (1 - Z)


def emergence_landscape_2d(nx=200, ny=200, t=0.0, theta_NT=1.0, lam=0.5):
    """
    Build a 2D emergence surface parametrized by (x, y) where:
    - x represents the spatial/configuration coordinate
    - y represents a second degree of freedom (e.g., momentum or field)

    The surface height h(x,y) is given by the D-ND potential
    modulated by a time-dependent coupling.
    """
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Map (x,y) to order parameter Z via sigmoid
    Z = 1.0 / (1.0 + np.exp(-X))

    # Surface height: potential + y-dependent modulation
    # The y-direction adds a second well structure
    coupling = 1.0 + 0.3 * np.sin(2 * np.pi * t / 50.0)  # Time-dependent
    h = dnd_potential(Z, theta_NT, lam) + 0.5 * Y**2 * coupling

    return X, Y, h, x, y


def gaussian_curvature_2d(h, x, y):
    """
    Compute Gaussian curvature K of the surface z = h(x,y).

    K = (h_xx * h_yy - h_xy²) / (1 + h_x² + h_y²)²

    Uses central finite differences for derivatives.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # First derivatives
    h_x = np.gradient(h, dx, axis=1)
    h_y = np.gradient(h, dy, axis=0)

    # Second derivatives
    h_xx = np.gradient(h_x, dx, axis=1)
    h_yy = np.gradient(h_y, dy, axis=0)
    h_xy = np.gradient(h_x, dy, axis=0)

    # Gaussian curvature
    numerator = h_xx * h_yy - h_xy**2
    denominator = (1.0 + h_x**2 + h_y**2)**2

    K = numerator / denominator

    return K, h_x, h_y


def gauss_bonnet_integral(K, x, y):
    """
    Compute the Gauss-Bonnet topological charge:
    χ_DND = (1/2π) ∫∫ K dA

    where dA is the area element on the surface.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Numerical integration via trapezoidal rule
    integral = np.trapezoid(np.trapezoid(K, dx=dx, axis=1), dx=dy)
    chi = integral / (2 * np.pi)

    return chi, integral


def time_evolution_chi(n_times=100, t_max=100.0, theta_NT=1.0):
    """
    Evolve χ_DND through time and track topological transitions.

    As the coupling parameter changes, the landscape topology changes,
    and χ should jump between integer values.
    """
    times = np.linspace(0, t_max, n_times)
    chi_values = np.zeros(n_times)
    K_max_values = np.zeros(n_times)
    K_mean_values = np.zeros(n_times)

    # Vary λ through a range that includes bifurcation
    lambda_values = 0.1 + 0.8 * np.sin(2 * np.pi * times / t_max)**2

    for i, (t, lam) in enumerate(zip(times, lambda_values)):
        X, Y, h, x, y = emergence_landscape_2d(nx=150, ny=150, t=t,
                                                 theta_NT=theta_NT, lam=lam)
        K, _, _ = gaussian_curvature_2d(h, x, y)
        chi, _ = gauss_bonnet_integral(K, x, y)

        chi_values[i] = chi
        K_max_values[i] = np.max(np.abs(K))
        K_mean_values[i] = np.mean(K)

        if (i + 1) % 25 == 0:
            print(f"  t={t:.1f}, λ={lam:.3f}, χ={chi:.4f}, "
                  f"|K|_max={K_max_values[i]:.4e}")

    return times, chi_values, K_max_values, K_mean_values, lambda_values


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def generate_figures(times, chi_values, K_max_values, K_mean_values,
                     lambda_values):
    """Generate figures for Paper C §3.3."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    # ─── Figure 7: Topological charge evolution ───
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(times, chi_values, 'b-', linewidth=2)
    # Mark nearest integers
    chi_rounded = np.round(chi_values)
    for val in np.unique(chi_rounded):
        ax1.axhline(y=val, color='r', linestyle=':', alpha=0.3)
    ax1.set_ylabel(r'$\chi_{\mathrm{DND}}$')
    ax1.set_title(r'Topological Charge Evolution: $\chi_{\mathrm{DND}} = \frac{1}{2\pi}\oint K_{\mathrm{gen}}\,dA$')
    ax1.grid(True, alpha=0.3)

    # Quantization check: distance to nearest integer
    dist_to_int = np.abs(chi_values - chi_rounded)
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(times, dist_to_int, alpha=0.2, color='orange')
    ax1_twin.set_ylabel(r'$|\chi - \mathrm{round}(\chi)|$', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1_twin.set_ylim(0, 0.5)

    ax2.plot(times, K_max_values, 'g-', linewidth=1.5, label=r'$|K|_{\max}$')
    ax2.plot(times, np.abs(K_mean_values), 'r-', linewidth=1.5, alpha=0.7,
             label=r'$|\langle K \rangle|$')
    ax2.set_ylabel('Curvature')
    ax2.set_title('Curvature Statistics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    ax3.plot(times, lambda_values, 'purple', linewidth=2)
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel(r'$\lambda$')
    ax3.set_title('Coupling Parameter')
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_C7_topological_charge.pdf')
    fig.savefig(OUTPUT_DIR / 'fig_C7_topological_charge.svg')
    plt.close(fig)
    print(f"  Saved fig_C7_topological_charge.pdf/svg")

    # ─── Figure 8: Curvature landscape snapshots ───
    fig2, axes = plt.subplots(2, 3, figsize=(15, 9))

    snapshot_times = [0, 20, 40, 60, 80, 99]
    snapshot_lambdas = [lambda_values[i] for i in snapshot_times]

    for ax, t_idx, lam in zip(axes.flat, snapshot_times, snapshot_lambdas):
        t = times[t_idx]
        X, Y, h, x, y = emergence_landscape_2d(nx=100, ny=100, t=t, lam=lam)
        K, _, _ = gaussian_curvature_2d(h, x, y)

        # Symmetric colormap centered at zero
        vmax = np.percentile(np.abs(K), 95)
        im = ax.contourf(X, Y, K, levels=30, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
        ax.set_title(f't={t:.0f}, λ={lam:.2f}\nχ={chi_values[t_idx]:.3f}',
                     fontsize=10)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle('Gaussian Curvature Landscape at Different Times',
                  fontsize=14, y=1.01)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'fig_C8_curvature_landscape.pdf')
    fig2.savefig(OUTPUT_DIR / 'fig_C8_curvature_landscape.svg')
    plt.close(fig2)
    print(f"  Saved fig_C8_curvature_landscape.pdf/svg")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TOPOLOGICAL CHARGE COMPUTATION — Paper C §3.3")
    print("=" * 70)

    # Single snapshot analysis
    print("\nSingle snapshot (t=0, λ=0.5):")
    X, Y, h, x, y = emergence_landscape_2d(nx=200, ny=200, t=0, lam=0.5)
    K, _, _ = gaussian_curvature_2d(h, x, y)
    chi, integral = gauss_bonnet_integral(K, x, y)
    print(f"  χ_DND = {chi:.6f}")
    print(f"  Nearest integer: {round(chi)}")
    print(f"  Distance to integer: {abs(chi - round(chi)):.6f}")
    print(f"  Raw integral: {integral:.6f}")
    print(f"  K range: [{K.min():.4e}, {K.max():.4e}]")
    print(f"  <K> = {K.mean():.4e}")

    # Time evolution
    print(f"\nTime evolution (100 steps):")
    times, chi_values, K_max, K_mean, lam_values = time_evolution_chi(
        n_times=100, t_max=100.0
    )

    # Quantization analysis
    chi_rounded = np.round(chi_values)
    dist_to_int = np.abs(chi_values - chi_rounded)
    print(f"\nQuantization analysis:")
    print(f"  Mean distance to nearest integer: {np.mean(dist_to_int):.6f}")
    print(f"  Max distance to nearest integer:  {np.max(dist_to_int):.6f}")
    print(f"  Fraction within 0.1 of integer:   {np.mean(dist_to_int < 0.1)*100:.1f}%")
    print(f"  Fraction within 0.2 of integer:   {np.mean(dist_to_int < 0.2)*100:.1f}%")
    print(f"  χ range: [{chi_values.min():.4f}, {chi_values.max():.4f}]")
    print(f"  Unique nearest integers: {np.unique(chi_rounded).astype(int).tolist()}")

    # Figures
    print("\nGenerating figures...")
    generate_figures(times, chi_values, K_max, K_mean, lam_values)

    # Save results
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'single_snapshot': {
            'chi': float(chi),
            'integral': float(integral),
            'K_range': [float(K.min()), float(K.max())],
            'K_mean': float(K.mean()),
        },
        'time_evolution': {
            'mean_dist_to_integer': float(np.mean(dist_to_int)),
            'max_dist_to_integer': float(np.max(dist_to_int)),
            'fraction_within_0.1': float(np.mean(dist_to_int < 0.1)),
            'fraction_within_0.2': float(np.mean(dist_to_int < 0.2)),
            'chi_range': [float(chi_values.min()), float(chi_values.max())],
            'unique_nearest_integers': np.unique(chi_rounded).astype(int).tolist(),
        }
    }
    with open(DATA_DIR / 'topological_charge_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'topological_charge_results.json'}")


if __name__ == '__main__':
    main()
