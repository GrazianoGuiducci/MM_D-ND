#!/usr/bin/env python3
"""
zeta_validation.py — Numerical Validation for Paper C (§4.3)

Tests the D-ND/Zeta conjecture: critical values of the generalized informational
curvature K_gen correspond to Riemann zeta zeros on the critical line.

Protocol (from Paper C §4.3):
1. Extract first 100 non-trivial zeros of ζ(s) via mpmath
2. Construct simplified emergence operator on N-level Hilbert space
3. Compute K_gen(x, t_n) for each zeta zero t_n
4. Identify critical curvature values K_c and spatial locations x_c
5. Statistical correlation between {t_n} and {K_c^(n)}
6. Publication-quality figures

References:
- Paper C §2.1: K_gen definition
- Paper C §4.3: Numerical protocol
- Paper C §4.3.1: Complementary tests (spectral gaps, Hausdorff)
- archive/sim_canonical/example_E_N_levels.py: N-level emergence model

Author: TM3 (Claude Code)
Date: 2026-02-26
"""

import numpy as np
from scipy import linalg, stats
from mpmath import zetazero, mp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# High precision for zeta zeros
mp.dps = 30

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "papers" / "figures"
DATA_DIR = Path(__file__).parent / "data"


def extract_zeta_zeros(n_zeros=100):
    """
    Extract first n non-trivial zeros of ζ(1/2 + it).

    Returns array of imaginary parts t_n (all positive by convention).
    Uses mpmath's verified computation.
    """
    print(f"Extracting {n_zeros} Riemann zeta zeros...")
    t_start = time.time()

    zeros = []
    for n in range(1, n_zeros + 1):
        z = zetazero(n)
        t_n = float(z.imag)
        zeros.append(t_n)
        if n % 25 == 0:
            print(f"  {n}/{n_zeros} — t_{n} = {t_n:.6f}")

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s")
    return np.array(zeros)


# ─────────────────────────────────────────────────────────────
# Emergence Model (from Paper A / Paper C §4.3 Step 2)
# ─────────────────────────────────────────────────────────────

def create_emergence_system(N, eigenvalue_pattern='prime'):
    """
    Construct the emergence operator E and Hamiltonian H on N-level system.

    |NT⟩ = (1/√N) Σ|k⟩  (uniform superposition — Null-All state)
    E = Σ λ_k |e_k⟩⟨e_k|  (emergence operator)
    H = diagonal Hamiltonian (drives time evolution)

    eigenvalue_pattern:
        'linear':  λ_k = k/N
        'prime':   λ_k ∝ 1/p_k (inverse primes — Paper C §4.3 Step 2)
        'log':     λ_k = log(k+1)/log(N)
    """
    # |NT⟩ state
    NT = np.ones(N, dtype=complex) / np.sqrt(N)

    # Emergence eigenvalues
    if eigenvalue_pattern == 'linear':
        lambdas = np.arange(1, N + 1) / N
    elif eigenvalue_pattern == 'prime':
        primes = _sieve_primes(N)
        lambdas = 1.0 / np.array(primes[:N], dtype=float)
        lambdas = lambdas / lambdas.max()  # Normalize to [0,1]
    elif eigenvalue_pattern == 'log':
        lambdas = np.log(np.arange(1, N + 1) + 1) / np.log(N + 1)
    else:
        raise ValueError(f"Unknown pattern: {eigenvalue_pattern}")

    E = np.diag(lambdas)

    # Hamiltonian: eigenvalues spaced by emergence eigenvalues
    # H encodes the spectral structure of the emergence
    h_eigenvalues = 2 * np.pi * lambdas
    H = np.diag(h_eigenvalues)

    return NT, E, H, lambdas


def _sieve_primes(n):
    """Generate at least n primes via sieve of Eratosthenes."""
    limit = max(n * 15, 100)  # Generous upper bound
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes


def time_evolution(H, t):
    """Compute U(t) = exp(-iHt) via matrix exponential."""
    return linalg.expm(-1j * H * t)


def emerged_state(NT, E, H, t):
    """Compute R(t) = U(t) E |NT⟩."""
    U_t = time_evolution(H, t)
    return U_t @ (E @ NT)


# ─────────────────────────────────────────────────────────────
# Generalized Informational Curvature K_gen (Paper C §2.1)
# ─────────────────────────────────────────────────────────────

def compute_Kgen_profile(NT, E, H, t, x_grid):
    """
    Compute K_gen(x, t) on a spatial grid.

    K_gen = ∇·(J ⊗ F) where:
    - J(x,t) = Im[ψ*(x,t) ∇ψ(x,t)]  (probability current)
    - F(x,t) = -∇V_eff(x,t)            (generalized force)

    For the N-level discrete system, we project onto position representation
    using a Gaussian basis centered at grid points.
    """
    N = len(NT)
    R_t = emerged_state(NT, E, H, t)

    # Build position-space wavefunction via Gaussian basis
    # Each level k is associated with a Gaussian centered at x_k
    sigma = (x_grid[-1] - x_grid[0]) / (2 * N)  # Width scales with system size
    x_centers = np.linspace(x_grid[0] * 0.8, x_grid[-1] * 0.8, N)

    # ψ(x, t) = Σ_k R_k(t) φ_k(x) where φ_k is Gaussian at x_centers[k]
    psi = np.zeros(len(x_grid), dtype=complex)
    dpsi = np.zeros(len(x_grid), dtype=complex)  # spatial derivative

    for k in range(N):
        phi_k = np.exp(-(x_grid - x_centers[k])**2 / (2 * sigma**2))
        dphi_k = -(x_grid - x_centers[k]) / sigma**2 * phi_k
        psi += R_t[k] * phi_k
        dpsi += R_t[k] * dphi_k

    # Normalize
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x_grid))
    if norm > 1e-15:
        psi /= norm
        dpsi /= norm

    # Probability density
    rho = np.abs(psi)**2
    rho = np.maximum(rho, 1e-30)  # Avoid division by zero

    # Probability current J(x) = Im[ψ* ∇ψ]
    J = np.imag(np.conj(psi) * dpsi)

    # Effective potential V_eff from quantum potential
    # V_eff(x) = -ℏ²/(2m) ∇²√ρ / √ρ  (Bohm quantum potential)
    sqrt_rho = np.sqrt(rho)
    d2_sqrt_rho = np.gradient(np.gradient(sqrt_rho, x_grid), x_grid)
    V_eff = np.zeros_like(x_grid)
    mask = sqrt_rho > 1e-15
    V_eff[mask] = -d2_sqrt_rho[mask] / sqrt_rho[mask]

    # Generalized force F = -∇V_eff
    F = -np.gradient(V_eff, x_grid)

    # K_gen = ∇·(J * F) = d/dx(J * F)
    K_gen = np.gradient(J * F, x_grid)

    return K_gen, psi, rho, J, F, V_eff


def find_critical_curvature(K_gen, x_grid):
    """
    Find critical curvature value K_c and its spatial location x_c.

    K_c is the extremal (min or max absolute) value of K_gen
    at a point where ∂K_gen/∂x = 0 (local extremum).
    """
    # Find local extrema via sign changes in derivative
    dK = np.gradient(K_gen, x_grid)
    sign_changes = np.where(np.diff(np.sign(dK)))[0]

    if len(sign_changes) == 0:
        # No local extrema — use global extremum
        idx = np.argmax(np.abs(K_gen))
        return K_gen[idx], x_grid[idx], idx

    # Among local extrema, find the one with largest absolute K_gen
    best_idx = sign_changes[0]
    best_K = np.abs(K_gen[best_idx])
    for idx in sign_changes:
        if np.abs(K_gen[idx]) > best_K:
            best_K = np.abs(K_gen[idx])
            best_idx = idx

    return K_gen[best_idx], x_grid[best_idx], best_idx


# ─────────────────────────────────────────────────────────────
# Main Validation Protocol
# ─────────────────────────────────────────────────────────────

def run_validation(n_zeros=100, N_system=100, eigenvalue_pattern='prime',
                   x_range=(-5, 5), n_x=500):
    """
    Execute the full validation protocol (Paper C §4.3).

    Returns dict with all results for paper integration.
    """
    print("=" * 70)
    print("D-ND/ZETA VALIDATION — Paper C §4.3 Numerical Protocol")
    print("=" * 70)

    # Step 1: Extract zeta zeros
    t_zeros = extract_zeta_zeros(n_zeros)

    # Step 2: Build emergence model
    print(f"\nBuilding {N_system}-level emergence system ({eigenvalue_pattern} pattern)...")
    NT, E, H, lambdas = create_emergence_system(N_system, eigenvalue_pattern)

    # Spatial grid
    x_grid = np.linspace(x_range[0], x_range[1], n_x)

    # Step 3-4: Compute K_gen at each zeta zero and find critical values
    print(f"\nComputing K_gen at {n_zeros} zeta zero locations...")
    K_c_values = np.zeros(n_zeros)
    x_c_values = np.zeros(n_zeros)
    K_gen_profiles = []

    for i, t_n in enumerate(t_zeros):
        K_gen, psi, rho, J, F, V_eff = compute_Kgen_profile(NT, E, H, t_n, x_grid)
        K_c, x_c, _ = find_critical_curvature(K_gen, x_grid)
        K_c_values[i] = K_c
        x_c_values[i] = x_c

        if i < 5:  # Store first 5 profiles for detailed plots
            K_gen_profiles.append({
                'K_gen': K_gen.copy(),
                'psi': psi.copy(),
                'rho': rho.copy(),
                't_n': t_n,
                'K_c': K_c,
                'x_c': x_c
            })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_zeros} — t_{i+1} = {t_n:.4f}, K_c = {K_c:.6e}, x_c = {x_c:.4f}")

    # Step 5: Statistical correlation
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(t_zeros, np.abs(K_c_values))
    # Spearman rank correlation
    r_spearman, p_spearman = stats.spearmanr(t_zeros, np.abs(K_c_values))
    # Kendall tau
    r_kendall, p_kendall = stats.kendalltau(t_zeros, np.abs(K_c_values))

    print(f"\nCorrelation t_n vs |K_c|:")
    print(f"  Pearson:  r = {r_pearson:.6f}, p = {p_pearson:.2e}")
    print(f"  Spearman: ρ = {r_spearman:.6f}, p = {p_spearman:.2e}")
    print(f"  Kendall:  τ = {r_kendall:.6f}, p = {p_kendall:.2e}")

    # Gap analysis (§4.3.1 Test 3)
    zeta_gaps = np.diff(t_zeros)
    K_c_sorted = np.abs(K_c_values)
    K_c_gaps = np.abs(np.diff(K_c_sorted))

    # KS test: do gap distributions match?
    if len(zeta_gaps) > 5 and len(K_c_gaps) > 5:
        # Normalize both to same scale
        zeta_gaps_norm = zeta_gaps / np.mean(zeta_gaps)
        K_c_gaps_norm = K_c_gaps / np.mean(K_c_gaps) if np.mean(K_c_gaps) > 0 else K_c_gaps
        ks_stat, ks_p = stats.ks_2samp(zeta_gaps_norm, K_c_gaps_norm)
        print(f"\nGap distribution KS test:")
        print(f"  KS statistic = {ks_stat:.6f}, p = {ks_p:.4f}")
    else:
        ks_stat, ks_p = np.nan, np.nan

    # Monotonicity analysis
    # Is |K_c| monotonically related to t_n?
    monotonic_increasing = np.sum(np.diff(np.abs(K_c_values)) > 0) / (n_zeros - 1)
    print(f"\nMonotonicity: {monotonic_increasing*100:.1f}% of consecutive pairs have |K_c| increasing with t_n")

    # Compile results
    results = {
        'n_zeros': n_zeros,
        'N_system': N_system,
        'eigenvalue_pattern': eigenvalue_pattern,
        't_zeros': t_zeros.tolist(),
        'K_c_values': K_c_values.tolist(),
        'x_c_values': x_c_values.tolist(),
        'correlation': {
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_rho': float(r_spearman),
            'spearman_p': float(p_spearman),
            'kendall_tau': float(r_kendall),
            'kendall_p': float(p_kendall),
        },
        'gap_analysis': {
            'ks_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
            'ks_p_value': float(ks_p) if not np.isnan(ks_p) else None,
        },
        'monotonicity_fraction': float(monotonic_increasing),
    }

    return results, t_zeros, K_c_values, x_c_values, K_gen_profiles, x_grid


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def generate_figures(results, t_zeros, K_c_values, x_c_values,
                     K_gen_profiles, x_grid):
    """Generate publication-quality figures for Paper C."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # ─── Figure 1: K_c vs t_n scatter ───
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter = ax1.scatter(t_zeros, np.abs(K_c_values),
                          c=np.arange(len(t_zeros)), cmap='viridis',
                          s=25, alpha=0.8, edgecolors='k', linewidths=0.3)

    # Trend line
    z = np.polyfit(t_zeros, np.abs(K_c_values), 2)
    p = np.poly1d(z)
    t_smooth = np.linspace(t_zeros[0], t_zeros[-1], 200)
    ax1.plot(t_smooth, p(t_smooth), 'r--', linewidth=1.5, alpha=0.7,
             label=f'Quadratic fit')

    r = results['correlation']['pearson_r']
    p_val = results['correlation']['pearson_p']
    ax1.text(0.02, 0.98,
             f'Pearson r = {r:.4f}\np = {p_val:.2e}\n'
             f'Spearman ρ = {results["correlation"]["spearman_rho"]:.4f}',
             transform=ax1.transAxes, va='top', ha='left',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_xlabel(r'Zeta zero imaginary part $t_n$')
    ax1.set_ylabel(r'Critical curvature $|K_c^{(n)}|$')
    ax1.set_title(r'D-ND/Zeta Conjecture: $|K_c|$ vs $t_n$ for first 100 zeros')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Zero index n')

    fig1.savefig(OUTPUT_DIR / 'fig_C1_Kc_vs_tn.pdf')
    fig1.savefig(OUTPUT_DIR / 'fig_C1_Kc_vs_tn.svg')
    plt.close(fig1)
    print(f"  Saved fig_C1_Kc_vs_tn.pdf/svg")

    # ─── Figure 2: K_gen profiles at first 5 zeta zeros ───
    fig2, axes2 = plt.subplots(1, min(5, len(K_gen_profiles)),
                                figsize=(16, 4), sharey=True)
    if not hasattr(axes2, '__len__'):
        axes2 = [axes2]

    for i, (ax, prof) in enumerate(zip(axes2, K_gen_profiles)):
        ax.plot(x_grid, prof['K_gen'], 'b-', linewidth=1.5)
        ax.axhline(y=prof['K_c'], color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=prof['x_c'], color='g', linestyle=':', alpha=0.7, linewidth=1)
        ax.fill_between(x_grid, prof['K_gen'], alpha=0.1, color='blue')
        ax.set_xlabel(r'$x$')
        if i == 0:
            ax.set_ylabel(r'$K_{\mathrm{gen}}(x, t_n)$')
        ax.set_title(f'$t_{{{i+1}}} = {prof["t_n"]:.2f}$', fontsize=11)
        ax.grid(True, alpha=0.2)

    fig2.suptitle(r'Informational Curvature Profiles at First 5 Zeta Zeros',
                  fontsize=13, y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'fig_C2_Kgen_profiles.pdf')
    fig2.savefig(OUTPUT_DIR / 'fig_C2_Kgen_profiles.svg')
    plt.close(fig2)
    print(f"  Saved fig_C2_Kgen_profiles.pdf/svg")

    # ─── Figure 3: Gap distribution comparison ───
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    zeta_gaps = np.diff(t_zeros)
    K_c_gaps = np.abs(np.diff(np.abs(K_c_values)))

    # Normalize
    zeta_gaps_norm = zeta_gaps / np.mean(zeta_gaps)
    K_c_mean = np.mean(K_c_gaps)
    K_c_gaps_norm = K_c_gaps / K_c_mean if K_c_mean > 0 else K_c_gaps

    ax3a.hist(zeta_gaps_norm, bins=25, density=True, alpha=0.6,
              color='steelblue', label='Zeta zero gaps', edgecolor='k', linewidth=0.5)
    ax3a.hist(K_c_gaps_norm, bins=25, density=True, alpha=0.6,
              color='coral', label=r'$|K_c|$ gaps', edgecolor='k', linewidth=0.5)
    ax3a.set_xlabel('Normalized gap size')
    ax3a.set_ylabel('Density')
    ax3a.set_title('Gap Distribution Comparison')
    ax3a.legend()
    ax3a.grid(True, alpha=0.2)

    if results['gap_analysis']['ks_statistic'] is not None:
        ax3a.text(0.98, 0.98,
                  f'KS = {results["gap_analysis"]["ks_statistic"]:.4f}\n'
                  f'p = {results["gap_analysis"]["ks_p_value"]:.4f}',
                  transform=ax3a.transAxes, va='top', ha='right',
                  fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # QQ plot
    n_qq = min(len(zeta_gaps_norm), len(K_c_gaps_norm))
    q_zeta = np.sort(zeta_gaps_norm)[:n_qq]
    q_Kc = np.sort(K_c_gaps_norm)[:n_qq]
    # Interpolate to same length
    q_zeta_interp = np.interp(np.linspace(0, 1, n_qq),
                               np.linspace(0, 1, len(q_zeta)), q_zeta)
    q_Kc_interp = np.interp(np.linspace(0, 1, n_qq),
                              np.linspace(0, 1, len(q_Kc)), q_Kc)

    ax3b.scatter(q_zeta_interp, q_Kc_interp, s=15, alpha=0.6, c='purple')
    lims = [0, max(q_zeta_interp.max(), q_Kc_interp.max()) * 1.1]
    ax3b.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    ax3b.set_xlabel('Zeta gap quantiles')
    ax3b.set_ylabel(r'$|K_c|$ gap quantiles')
    ax3b.set_title('QQ Plot: Gap Distributions')
    ax3b.grid(True, alpha=0.2)
    ax3b.set_aspect('equal')

    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'fig_C3_gap_analysis.pdf')
    fig3.savefig(OUTPUT_DIR / 'fig_C3_gap_analysis.svg')
    plt.close(fig3)
    print(f"  Saved fig_C3_gap_analysis.pdf/svg")

    # ─── Figure 4: Critical location x_c evolution ───
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 8))

    ax4a.plot(t_zeros, x_c_values, 'o-', markersize=3, linewidth=0.8,
              color='darkgreen', alpha=0.8)
    ax4a.set_xlabel(r'$t_n$')
    ax4a.set_ylabel(r'Critical location $x_c^{(n)}$')
    ax4a.set_title('Spatial Location of Critical Curvature vs Zeta Zero')
    ax4a.grid(True, alpha=0.3)

    # Phase space: x_c vs K_c
    scatter4 = ax4b.scatter(x_c_values, K_c_values,
                             c=t_zeros, cmap='plasma', s=20, alpha=0.7,
                             edgecolors='k', linewidths=0.3)
    ax4b.set_xlabel(r'$x_c^{(n)}$')
    ax4b.set_ylabel(r'$K_c^{(n)}$')
    ax4b.set_title(r'Phase Space: $(x_c, K_c)$ colored by $t_n$')
    ax4b.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4b, label=r'$t_n$')

    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / 'fig_C4_critical_locations.pdf')
    fig4.savefig(OUTPUT_DIR / 'fig_C4_critical_locations.svg')
    plt.close(fig4)
    print(f"  Saved fig_C4_critical_locations.pdf/svg")

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────
# Multi-Pattern Analysis
# ─────────────────────────────────────────────────────────────

def run_multi_pattern_comparison(n_zeros=100, N_system=100):
    """
    Run validation with different eigenvalue patterns to test robustness.
    Paper C §4.3 Step 2 suggests trying uniform and prime-based spacing.
    """
    print("\n" + "=" * 70)
    print("MULTI-PATTERN COMPARISON")
    print("=" * 70)

    patterns = ['linear', 'prime', 'log']
    all_results = {}

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern} ---")
        results, t_z, K_c, x_c, _, _ = run_validation(
            n_zeros=n_zeros, N_system=N_system,
            eigenvalue_pattern=pattern, n_x=500
        )
        all_results[pattern] = results

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Pattern':<12} {'Pearson r':>12} {'Spearman ρ':>12} {'Kendall τ':>12} {'Monotonicity':>14}")
    print("-" * 62)
    for pattern in patterns:
        r = all_results[pattern]
        c = r['correlation']
        print(f"{pattern:<12} {c['pearson_r']:>12.6f} {c['spearman_rho']:>12.6f} "
              f"{c['kendall_tau']:>12.6f} {r['monotonicity_fraction']*100:>13.1f}%")

    return all_results


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Primary validation with prime-based eigenvalues
    results, t_zeros, K_c_values, x_c_values, profiles, x_grid = run_validation(
        n_zeros=100, N_system=100, eigenvalue_pattern='prime'
    )

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    generate_figures(results, t_zeros, K_c_values, x_c_values, profiles, x_grid)

    # Save numerical data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / 'zeta_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'zeta_validation_results.json'}")

    # Multi-pattern comparison
    all_results = run_multi_pattern_comparison(n_zeros=100, N_system=100)

    with open(DATA_DIR / 'multi_pattern_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Multi-pattern results saved to {DATA_DIR / 'multi_pattern_results.json'}")

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    c = results['correlation']
    print(f"\nPrimary result (prime pattern, N=100):")
    print(f"  Pearson correlation:  r = {c['pearson_r']:.6f} (p = {c['pearson_p']:.2e})")
    print(f"  Spearman correlation: ρ = {c['spearman_rho']:.6f}")

    if abs(c['pearson_r']) > 0.8:
        print("\n  → STRONG correlation: Supporting evidence for conjecture")
    elif abs(c['pearson_r']) > 0.4:
        print("\n  → MODERATE correlation: Neutral — structure unclear")
    else:
        print("\n  → WEAK correlation: Evidence against conjecture as stated")

    print(f"\nNext steps:")
    print(f"  1. Integrate results into Paper C §4.3")
    print(f"  2. Run spectral_gap_analysis.py for Test 3 (§4.3.1)")
    print(f"  3. Run topological_charge.py for §3.3 verification")
