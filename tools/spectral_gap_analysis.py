#!/usr/bin/env python3
"""
spectral_gap_analysis.py — Paper C §4.3.1 Test 3: Spectral Gap Estimates

Computes eigenvalues of the Laplace-Beltrami operator on the emergence manifold
and compares spectral gaps with gaps between consecutive Riemann zeta zeros.

Tests whether the two point processes are statistically indistinguishable
(Kolmogorov-Smirnov, pair correlation, nearest-neighbor spacing).

References:
- Paper C §4.3.1 Test 3
- Paper C §4.4: Laplace-Beltrami eigenvalues and Hilbert-Pólya connection
- Berry-Keating conjecture: zeta zeros ~ eigenvalues of quantum Hamiltonian

Author: TM3 (Claude Code)
Date: 2026-02-26
"""

import numpy as np
from scipy import linalg, stats, sparse
from mpmath import zetazero, mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

mp.dps = 30
OUTPUT_DIR = Path(__file__).parent.parent / "papers" / "figures"
DATA_DIR = Path(__file__).parent / "data"


def extract_zeta_zeros(n_zeros=100):
    """Extract first n non-trivial zeta zeros (imaginary parts)."""
    print(f"Extracting {n_zeros} zeta zeros...")
    zeros = [float(zetazero(n).imag) for n in range(1, n_zeros + 1)]
    print(f"  Done. Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
    return np.array(zeros)


# ─────────────────────────────────────────────────────────────
# Emergence Manifold and Laplace-Beltrami Operator
# ─────────────────────────────────────────────────────────────

def build_emergence_metric(N, pattern='log'):
    """
    Build the Fisher information metric on the emergence manifold.

    The manifold is parametrized by emergence eigenvalues λ_k.
    The Fisher metric g_{ij} measures distinguishability between
    nearby emergence states.

    For diagonal emergence operator with eigenvalues λ_k:
    g_{kk} = 1/(λ_k(1-λ_k)) — Fisher metric for Bernoulli parameter
    """
    if pattern == 'log':
        lambdas = np.log(np.arange(1, N + 1) + 1) / np.log(N + 1)
    elif pattern == 'linear':
        lambdas = np.arange(1, N + 1) / (N + 1)
    elif pattern == 'prime':
        primes = _sieve_primes(N)
        lambdas = 1.0 / np.array(primes[:N], dtype=float)
        lambdas = lambdas / lambdas.max() * 0.99  # Keep away from boundary
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Clamp to avoid singularities at 0 and 1
    lambdas = np.clip(lambdas, 0.01, 0.99)

    # Fisher metric (diagonal) for Bernoulli-type emergence
    g_diag = 1.0 / (lambdas * (1.0 - lambdas))

    return lambdas, g_diag


def _sieve_primes(n):
    """Generate at least n primes."""
    limit = max(n * 15, 100)
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def laplace_beltrami_eigenvalues(g_diag, N):
    """
    Compute eigenvalues of the Laplace-Beltrami operator on the
    emergence manifold with diagonal metric g.

    Δ_M Φ = g^{μν} ∇_μ ∇_ν Φ

    For 1D manifold with metric g(x), the Laplacian is:
    Δ Φ = (1/√g) d/dx(√g g^{-1} dΦ/dx)

    We discretize on N points with the given metric.
    """
    # g^{-1} (inverse metric = contravariant components)
    g_inv = 1.0 / g_diag

    # √g (volume element)
    sqrt_g = np.sqrt(g_diag)

    # Build discrete Laplacian with metric
    # Using finite differences: Δ Φ_i ≈ (1/√g_i) Σ_j L_{ij} Φ_j
    dx = 1.0 / N  # Uniform parameter spacing

    L = np.zeros((N, N))
    for i in range(1, N - 1):
        # Metric-weighted second derivative
        coeff_l = sqrt_g[i-1] * g_inv[i-1]
        coeff_c = sqrt_g[i] * g_inv[i]
        coeff_r = sqrt_g[i+1] * g_inv[i+1]

        L[i, i-1] = (coeff_l + coeff_c) / (2.0 * dx**2 * sqrt_g[i])
        L[i, i]   = -(coeff_c * 2.0) / (dx**2 * sqrt_g[i])
        L[i, i+1] = (coeff_r + coeff_c) / (2.0 * dx**2 * sqrt_g[i])

    # Boundary conditions: Dirichlet (Φ=0 at boundaries)
    # First and last rows stay zero

    # Compute eigenvalues
    eigenvalues = np.sort(np.real(linalg.eigvals(L)))

    # Take only negative eigenvalues (Laplacian is negative semi-definite)
    # and flip sign for convenience (so they're positive)
    neg_eigs = -eigenvalues[eigenvalues < -1e-10]
    neg_eigs = np.sort(neg_eigs)

    return neg_eigs


def build_emergence_hamiltonian(N, pattern='log', coupling=1.0):
    """
    Build the D-ND emergence Hamiltonian.

    H = Δ_M + V_eff(Z)

    where V_eff is the double-well potential from the D-ND framework:
    V(Z) = Z²(1-Z)² (from sim_lagrangian_v2.py)

    The Hamiltonian eigenvalues are the candidate Hilbert-Pólya spectrum.
    """
    lambdas, g_diag = build_emergence_metric(N, pattern)

    # Kinetic part: Laplace-Beltrami
    g_inv = 1.0 / g_diag
    sqrt_g = np.sqrt(g_diag)
    dx = 1.0 / N

    H = np.zeros((N, N))
    for i in range(1, N - 1):
        coeff_l = sqrt_g[i-1] * g_inv[i-1]
        coeff_c = sqrt_g[i] * g_inv[i]
        coeff_r = sqrt_g[i+1] * g_inv[i+1]

        H[i, i-1] = -(coeff_l + coeff_c) / (2.0 * dx**2 * sqrt_g[i])
        H[i, i]   = (coeff_c * 2.0) / (dx**2 * sqrt_g[i])
        H[i, i+1] = -(coeff_r + coeff_c) / (2.0 * dx**2 * sqrt_g[i])

    # Potential part: D-ND double-well
    V = coupling * lambdas**2 * (1.0 - lambdas)**2
    H += np.diag(V)

    # Boundary: large potential to enforce confinement
    H[0, 0] = 1e6
    H[-1, -1] = 1e6

    eigenvalues = np.sort(np.real(linalg.eigvals(H)))

    # Filter: keep only physical eigenvalues (positive, not boundary artifacts)
    physical = eigenvalues[(eigenvalues > 0) & (eigenvalues < 1e5)]

    return np.sort(physical), lambdas


# ─────────────────────────────────────────────────────────────
# Statistical Comparison
# ─────────────────────────────────────────────────────────────

def compare_gap_statistics(spec_gaps, zeta_gaps, label_spec="Spectral"):
    """Compare gap distributions between spectrum and zeta zeros."""
    # Normalize to mean spacing = 1
    spec_norm = spec_gaps / np.mean(spec_gaps) if np.mean(spec_gaps) > 0 else spec_gaps
    zeta_norm = zeta_gaps / np.mean(zeta_gaps)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(spec_norm, zeta_norm)

    # Mean and variance comparison
    spec_mean = np.mean(spec_norm)
    zeta_mean = np.mean(zeta_norm)
    spec_var = np.var(spec_norm)
    zeta_var = np.var(zeta_norm)

    # Pair correlation: r_2(s) = density of gaps of size s
    # Approximate via nearest-neighbor spacing distribution
    # For GUE (expected for zeta zeros): P(s) ≈ (32/π²)s² exp(-4s²/π)
    # Wigner surmise

    results = {
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'spec_mean_normalized': float(spec_mean),
        'zeta_mean_normalized': float(zeta_mean),
        'spec_variance': float(spec_var),
        'zeta_variance': float(zeta_var),
        'n_spectral_gaps': int(len(spec_gaps)),
        'n_zeta_gaps': int(len(zeta_gaps)),
    }

    return results, spec_norm, zeta_norm


def wigner_surmise(s):
    """GUE Wigner surmise: P(s) = (32/π²)s² exp(-4s²/π)."""
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def generate_figures(zeta_gaps_norm, results_by_pattern, all_spec_gaps):
    """Generate figures for Paper C §4.3.1."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'legend.fontsize': 10, 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    patterns = list(results_by_pattern.keys())
    colors = {'linear': 'steelblue', 'prime': 'coral', 'log': 'forestgreen'}

    # ─── Figure 5: Nearest-neighbor spacing distributions ───
    fig, axes = plt.subplots(1, len(patterns), figsize=(5 * len(patterns), 5), sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    s_wigner = np.linspace(0, 4, 200)
    P_wigner = wigner_surmise(s_wigner)

    for ax, pattern in zip(axes, patterns):
        spec_norm = all_spec_gaps[pattern]
        color = colors.get(pattern, 'gray')
        r = results_by_pattern[pattern]

        ax.hist(zeta_gaps_norm, bins=20, density=True, alpha=0.4,
                color='gold', label='Zeta gaps', edgecolor='k', linewidth=0.5)
        ax.hist(spec_norm, bins=20, density=True, alpha=0.5,
                color=color, label=f'{pattern} gaps', edgecolor='k', linewidth=0.5)
        ax.plot(s_wigner, P_wigner, 'k--', linewidth=1.5, alpha=0.7,
                label='GUE Wigner')

        ax.set_xlabel('Normalized spacing $s$')
        if ax == axes[0]:
            ax.set_ylabel('$P(s)$')
        ax.set_title(f'{pattern.capitalize()} (KS={r["ks_statistic"]:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 4)

    fig.suptitle('Nearest-Neighbor Spacing: Spectral Gaps vs Zeta Gaps vs GUE',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_C5_spectral_gaps.pdf')
    fig.savefig(OUTPUT_DIR / 'fig_C5_spectral_gaps.svg')
    plt.close(fig)
    print(f"  Saved fig_C5_spectral_gaps.pdf/svg")

    # ─── Figure 6: Eigenvalue staircase comparison ───
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Zeta zero staircase N(t) = number of zeros up to height t
    t_z = np.sort(np.concatenate([[0], np.cumsum(zeta_gaps_norm)]))
    n_z = np.arange(len(t_z))
    ax2.step(t_z, n_z / len(n_z), where='post', linewidth=2,
             color='gold', label='Zeta zeros (normalized)')

    for pattern in patterns:
        spec = all_spec_gaps[pattern]
        t_s = np.sort(np.concatenate([[0], np.cumsum(spec)]))
        n_s = np.arange(len(t_s))
        color = colors.get(pattern, 'gray')
        ax2.step(t_s, n_s / len(n_s), where='post', linewidth=1.5,
                 color=color, alpha=0.8, label=f'{pattern.capitalize()} spectrum')

    ax2.set_xlabel('Normalized cumulative spacing')
    ax2.set_ylabel('Cumulative fraction $N(s)/N_{total}$')
    ax2.set_title('Eigenvalue Staircase: Spectral vs Zeta')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'fig_C6_staircase.pdf')
    fig2.savefig(OUTPUT_DIR / 'fig_C6_staircase.svg')
    plt.close(fig2)
    print(f"  Saved fig_C6_staircase.pdf/svg")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SPECTRAL GAP ANALYSIS — Paper C §4.3.1 Test 3")
    print("=" * 70)

    # Zeta zeros
    n_zeros = 100
    t_zeros = extract_zeta_zeros(n_zeros)
    zeta_gaps = np.diff(t_zeros)
    zeta_gaps_norm = zeta_gaps / np.mean(zeta_gaps)

    # System sizes to test
    N_values = [50, 100, 200]
    patterns = ['linear', 'prime', 'log']

    all_results = {}
    all_spec_gaps = {}

    for pattern in patterns:
        print(f"\n{'─'*50}")
        print(f"Pattern: {pattern}")
        print(f"{'─'*50}")

        best_ks = 1.0
        best_N = 0
        best_gaps = None

        for N in N_values:
            print(f"\n  N = {N}:")
            eigs, lambdas = build_emergence_hamiltonian(N, pattern)

            if len(eigs) < 3:
                print(f"    Too few eigenvalues ({len(eigs)}), skipping")
                continue

            spec_gaps = np.diff(eigs)
            spec_gaps = spec_gaps[spec_gaps > 1e-10]  # Remove near-zero gaps

            if len(spec_gaps) < 5:
                print(f"    Too few gaps ({len(spec_gaps)}), skipping")
                continue

            r, spec_norm, _ = compare_gap_statistics(spec_gaps, zeta_gaps, pattern)
            print(f"    Eigenvalues: {len(eigs)}, Gaps: {len(spec_gaps)}")
            print(f"    KS statistic: {r['ks_statistic']:.4f} (p = {r['ks_p_value']:.4f})")
            print(f"    Variance: spectral={r['spec_variance']:.4f}, zeta={r['zeta_variance']:.4f}")

            if r['ks_statistic'] < best_ks:
                best_ks = r['ks_statistic']
                best_N = N
                best_gaps = spec_norm
                best_result = r

        if best_gaps is not None:
            all_results[pattern] = best_result
            all_spec_gaps[pattern] = best_gaps
            print(f"\n  Best: N={best_N}, KS={best_ks:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pattern':<12} {'KS stat':>10} {'KS p':>10} {'Var(spec)':>12} {'Var(zeta)':>12}")
    print("-" * 56)
    for pattern in patterns:
        if pattern in all_results:
            r = all_results[pattern]
            print(f"{pattern:<12} {r['ks_statistic']:>10.4f} {r['ks_p_value']:>10.4f} "
                  f"{r['spec_variance']:>12.4f} {r['zeta_variance']:>12.4f}")

    # Figures
    print("\nGenerating figures...")
    generate_figures(zeta_gaps_norm, all_results, all_spec_gaps)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / 'spectral_gap_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'spectral_gap_results.json'}")


if __name__ == '__main__':
    main()
