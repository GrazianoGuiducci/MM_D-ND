"""
phase_diagrams.py

Diagrammi di Fase (θ_NT, λ) → Attrattori (NC.2)

Costruisce una mappa di fase bidimensionale che mostra i bacini di attrazione
nel piano (θ_NT, λ) del modello D-ND.

Per ogni coppia (θ_NT, λ), simula il sistema partendo da una condizione iniziale
leggermente perturbata da Z=0.5 (punto instabile) e determina l'attrattore finale:
  - Attrattore verso 0 (stato "Nulla")
  - Attrattore verso 1 (stato "Tutto")
  - Comportamento oscillatorio

Utilizza una griglia di valori e campioni multipli per robustezza.

Il diagramma di fase mostra i bacini di attrazione con codifica a colori.

Autore: Claude Code
Data: 2025-02-12
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Imposta seed per riproducibilità
np.random.seed(42)


def potential_V(Z, theta_NT=1.0, lambda_par=0.1):
    """
    Potenziale informazionale V(Z, θ_NT, λ).

    Args:
        Z (float or np.ndarray): coordinata di stato
        theta_NT (float): momento angolare NT
        lambda_par (float): parametro di transizione D-ND

    Returns:
        float or np.ndarray: valore del potenziale
    """
    V_base = (Z ** 2) * ((1 - Z) ** 2)
    V_coupling = lambda_par * theta_NT * Z * (1 - Z)
    return V_base + V_coupling


def dV_dZ(Z, theta_NT=1.0, lambda_par=0.1):
    """
    Derivata del potenziale rispetto a Z.

    Args:
        Z (float or np.ndarray): coordinata di stato
        theta_NT (float): momento angolare NT
        lambda_par (float): parametro di transizione D-ND

    Returns:
        float or np.ndarray: derivata del potenziale
    """
    dV_base = 2 * Z * (1 - Z) * (1 - 2 * Z)
    dV_coupling = lambda_par * theta_NT * (1 - 2 * Z)
    return dV_base + dV_coupling


def d_state_dt(t, state, theta_NT=1.0, lambda_par=0.1, c_abs=0.5):
    """
    Sistema di ODE del primo ordine per [Z, V_Z].

    Args:
        t (float): tempo
        state (np.ndarray): [Z, V_Z]
        theta_NT (float): momento angolare
        lambda_par (float): parametro di transizione
        c_abs (float): coefficiente di assorbimento

    Returns:
        np.ndarray: derivate [dZ/dt, dV_Z/dt]
    """
    Z, V_Z = state
    dZ_dt = V_Z
    dVZ_dt = -dV_dZ(Z, theta_NT, lambda_par) - c_abs * V_Z
    return np.array([dZ_dt, dVZ_dt])


def identify_attractor(theta_NT, lambda_par, c_abs=0.5, t_max=200.0,
                       Z0=0.5, perturbation=0.05):
    """
    Identifica l'attrattore raggiunto a partire da una condizione iniziale.

    Usa integrazione RK4 diretta.

    Args:
        theta_NT (float): momento angolare NT
        lambda_par (float): parametro di transizione
        c_abs (float): coefficiente di assorbimento
        t_max (float): tempo massimo di simulazione
        Z0 (float): condizione iniziale di base
        perturbation (float): ampiezza della perturbazione

    Returns:
        str: tipo di attrattore ('Z≈0', 'Z≈1', 'oscillation')
        float: valore finale di Z
        float: ampiezza media delle oscillazioni finali
    """
    # Perturba la condizione iniziale
    Z_initial = Z0 + perturbation * np.random.uniform(-1, 1)
    Z_initial = np.clip(Z_initial, 0.01, 0.99)

    # Integrazione RK4 (passo più grande per velocità)
    dt = 0.2  # Passo aumentato per velocità
    state = np.array([Z_initial, 0.0])
    Z_history = [state[0]]

    t = 0.0
    steps = 0
    max_steps = int(t_max / dt) + 100

    while t < t_max and steps < max_steps:
        steps += 1
        k1 = d_state_dt(t, state, theta_NT, lambda_par, c_abs)
        k2 = d_state_dt(t + 0.5 * dt, state + 0.5 * dt * k1, theta_NT, lambda_par, c_abs)
        k3 = d_state_dt(t + 0.5 * dt, state + 0.5 * dt * k2, theta_NT, lambda_par, c_abs)
        k4 = d_state_dt(t + dt, state + dt * k3, theta_NT, lambda_par, c_abs)

        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state[0] = np.clip(state[0], -0.5, 1.5)  # Evita divergenze
        t += dt
        Z_history.append(state[0])

    if len(Z_history) < 10:
        return None, None, None

    # Analizza il comportamento finale (ultimi 20% del tempo)
    final_idx = int(0.8 * len(Z_history))
    Z_final_phase = np.array(Z_history[final_idx:])

    Z_mean = np.mean(Z_final_phase)
    Z_std = np.std(Z_final_phase)
    Z_min = np.min(Z_final_phase)
    Z_max = np.max(Z_final_phase)

    # Classifica l'attrattore
    if Z_std < 0.05:  # Stato stazionario
        if Z_mean < 0.3:
            attractor = 'Z≈0'
        elif Z_mean > 0.7:
            attractor = 'Z≈1'
        else:
            attractor = 'mixed'
    else:  # Comportamento oscillatorio
        attractor = 'oscillation'

    oscillation_amplitude = 0.5 * (Z_max - Z_min)

    return attractor, Z_mean, oscillation_amplitude


def build_phase_diagram(theta_NT_range=(0.1, 3.0), lambda_range=(0.0, 1.0),
                        n_theta=20, n_lambda=20, n_samples=2):
    """
    Costruisce il diagramma di fase (θ_NT, λ).

    Args:
        theta_NT_range (tuple): intervallo per θ_NT
        lambda_range (tuple): intervallo per λ
        n_theta (int): numero di punti lungo θ_NT
        n_lambda (int): numero di punti lungo λ
        n_samples (int): numero di campioni per punto (robustezza)

    Returns:
        tuple: (theta_grid, lambda_grid, phase_data, attractor_names)
    """
    theta_NT_vals = np.linspace(theta_NT_range[0], theta_NT_range[1], n_theta)
    lambda_vals = np.linspace(lambda_range[0], lambda_range[1], n_lambda)

    # Mappa degli attrattori: 0=Z≈0, 1=Z≈1, 2=oscillation, 3=mixed
    attractor_map = {'Z≈0': 0, 'Z≈1': 1, 'oscillation': 2, 'mixed': 3}
    phase_data = np.zeros((n_lambda, n_theta))

    print("\n" + "="*70)
    print("Costruzione del Diagramma di Fase")
    print("="*70)
    print(f"{'θ_NT':<10} {'λ':<10} {'Attrattore':<15} {'Z medio':<15} {'Ampiezza osc':<15}")
    print("-" * 65)

    for i, lambda_par in enumerate(lambda_vals):
        for j, theta_NT in enumerate(theta_NT_vals):
            # Campiona multipli punti per robustezza
            attractors = []
            Z_means = []
            amplitudes = []

            for sample in range(n_samples):
                att, Z_mean, amp = identify_attractor(
                    theta_NT, lambda_par, c_abs=0.5, t_max=200.0,
                    Z0=0.5, perturbation=0.05
                )
                if att is not None:
                    attractors.append(att)
                    Z_means.append(Z_mean)
                    amplitudes.append(amp)

            if len(attractors) > 0:
                # Moda (attrattore più frequente)
                from collections import Counter
                most_common_att = Counter(attractors).most_common(1)[0][0]
                phase_data[i, j] = attractor_map[most_common_att]

                avg_Z = np.mean(Z_means)
                avg_amp = np.mean(amplitudes)

                if (j % 5 == 0) and (i % 5 == 0):  # Stampa ogni 5 punti
                    print(f"{theta_NT:<10.2f} {lambda_par:<10.4f} {most_common_att:<15} "
                          f"{avg_Z:<15.4f} {avg_amp:<15.6f}")

    return theta_NT_vals, lambda_vals, phase_data, attractor_map


def plot_phase_diagram(theta_NT_vals, lambda_vals, phase_data, attractor_map):
    """
    Grafica il diagramma di fase con codifica a colori e contorni.

    Args:
        theta_NT_vals (np.ndarray): valori di θ_NT
        lambda_vals (np.ndarray): valori di λ
        phase_data (np.ndarray): mappa dei bacini di attrazione
        attractor_map (dict): mappa nome attrattore -> codice numerico
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Crea meshgrid per immagine
    theta_grid, lambda_grid = np.meshgrid(theta_NT_vals, lambda_vals)

    # Mappa inversa
    inv_map = {v: k for k, v in attractor_map.items()}

    # Colori per attrattori
    colors_dict = {
        'Z≈0': '#1f77b4',          # Blu
        'Z≈1': '#ff7f0e',          # Arancione
        'oscillation': '#2ca02c',  # Verde
        'mixed': '#d62728'         # Rosso
    }

    # Visualizzazione con imshow
    cmap_custom = plt.cm.get_cmap('tab10')
    im = ax.contourf(theta_grid, lambda_grid, phase_data, levels=[0, 1, 2, 3, 4],
                     cmap='RdYlBu_r', alpha=0.8)

    # Contorni per demarcazione
    contours = ax.contour(theta_grid, lambda_grid, phase_data, levels=[0.5, 1.5, 2.5],
                          colors='black', linewidths=1.5, alpha=0.4)

    # Etichette dei contorni
    ax.clabel(contours, inline=True, fontsize=9, fmt='%1.1f')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attractor Type')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Z≈0 (Nulla)', 'Z≈1 (Tutto)', 'Oscillation', 'Mixed'])

    # Punti critici (se rilevabili)
    for i, lambda_par in enumerate(lambda_vals):
        for j, theta_NT in enumerate(theta_NT_vals):
            if j % 5 == 0 and i % 5 == 0:
                att_code = phase_data[i, j]
                att_name = inv_map.get(att_code, 'Unknown')
                # Marker piccolo
                if att_name == 'Z≈0':
                    marker = 'o'
                    color = '#1f77b4'
                elif att_name == 'Z≈1':
                    marker = 's'
                    color = '#ff7f0e'
                elif att_name == 'oscillation':
                    marker = '^'
                    color = '#2ca02c'
                else:
                    marker = 'x'
                    color = '#d62728'

                ax.scatter(theta_NT, lambda_par, marker=marker, s=30, color=color,
                          edgecolors='black', linewidths=0.5, zorder=5, alpha=0.6)

    # Assi e etichette
    ax.set_xlabel(r'Momento Angolare $\theta_{NT}$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Parametro di Transizione $\lambda$', fontsize=13, fontweight='bold')
    ax.set_title(r'Diagramma di Fase D-ND: Bacini di Attrazione $(θ_{NT}, λ)$',
                fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', color='gray')

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Attrattore Z≈0 (Nulla)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Attrattore Z≈1 (Tutto)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Oscillazione'),
        Patch(facecolor='#d62728', edgecolor='black', label='Regime Misto')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    return fig, ax


def main():
    """
    Esegue la costruzione e visualizzazione del diagramma di fase.
    """
    print("="*70)
    print("DIAGRAMMI DI FASE - Modello D-ND (NC.2)")
    print("="*70)

    # Parametri di griglia (ridotti per performance)
    theta_NT_range = (0.1, 3.0)
    lambda_range = (0.0, 1.0)
    n_theta = 20
    n_lambda = 20
    n_samples = 2  # Campioni per robustezza

    # Costruisce il diagramma di fase
    theta_vals, lambda_vals, phase_data, attractor_map = build_phase_diagram(
        theta_NT_range=theta_NT_range,
        lambda_range=lambda_range,
        n_theta=n_theta,
        n_lambda=n_lambda,
        n_samples=n_samples
    )

    print(f"\n{'='*70}")
    print("Diagramma di fase completato!")
    print(f"{'='*70}\n")

    # Grafica
    fig, ax = plot_phase_diagram(theta_vals, lambda_vals, phase_data, attractor_map)

    # Salva figura
    output_file = '/sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/sim_canonical/phase_diagram.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Diagramma di fase salvato: {output_file}\n")

    # Crea figura aggiuntiva con sezioni 1D
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(r'Analisi Unidimensionale dei Bacini di Attrazione',
                  fontsize=14, fontweight='bold')

    # Sezione a λ=0.1 (costante)
    lambda_fixed = 0.1
    lambda_idx = np.argmin(np.abs(lambda_vals - lambda_fixed))
    ax = axes[0, 0]
    ax.bar(theta_vals, phase_data[lambda_idx, :], width=0.08, color='#2E86AB', alpha=0.7)
    ax.set_xlabel(r'Momento Angolare $\theta_{NT}$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tipo di Attrattore', fontsize=11, fontweight='bold')
    ax.set_title(rf'Sezione a $\lambda = {lambda_fixed}$ (costante)', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Z≈0', 'Z≈1', 'Osc', 'Mix'])
    ax.grid(True, alpha=0.3, axis='y')

    # Sezione a λ=0.5 (costante)
    lambda_fixed = 0.5
    lambda_idx = np.argmin(np.abs(lambda_vals - lambda_fixed))
    ax = axes[0, 1]
    ax.bar(theta_vals, phase_data[lambda_idx, :], width=0.08, color='#F18F01', alpha=0.7)
    ax.set_xlabel(r'Momento Angolare $\theta_{NT}$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tipo di Attrattore', fontsize=11, fontweight='bold')
    ax.set_title(rf'Sezione a $\lambda = {lambda_fixed}$ (costante)', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Z≈0', 'Z≈1', 'Osc', 'Mix'])
    ax.grid(True, alpha=0.3, axis='y')

    # Sezione a θ_NT=0.5 (costante)
    theta_fixed = 0.5
    theta_idx = np.argmin(np.abs(theta_vals - theta_fixed))
    ax = axes[1, 0]
    ax.bar(lambda_vals, phase_data[:, theta_idx], width=0.02, color='#06A77D', alpha=0.7)
    ax.set_xlabel(r'Parametro di Transizione $\lambda$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tipo di Attrattore', fontsize=11, fontweight='bold')
    ax.set_title(rf'Sezione a $\theta_{{NT}} = {theta_fixed}$ (costante)', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Z≈0', 'Z≈1', 'Osc', 'Mix'])
    ax.grid(True, alpha=0.3, axis='y')

    # Sezione a θ_NT=2.0 (costante)
    theta_fixed = 2.0
    theta_idx = np.argmin(np.abs(theta_vals - theta_fixed))
    ax = axes[1, 1]
    ax.bar(lambda_vals, phase_data[:, theta_idx], width=0.02, color='#A23B72', alpha=0.7)
    ax.set_xlabel(r'Parametro di Transizione $\lambda$', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tipo di Attrattore', fontsize=11, fontweight='bold')
    ax.set_title(rf'Sezione a $\theta_{{NT}} = {theta_fixed}$ (costante)', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Z≈0', 'Z≈1', 'Osc', 'Mix'])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file2 = '/sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/sim_canonical/phase_diagram_sections.pdf'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Sezioni del diagramma salvate: {output_file2}\n")

    # Statistiche finali
    print("="*70)
    print("Statistiche dei Bacini di Attrazione")
    print("="*70)
    print(f"Totale punti nel diagramma: {n_theta * n_lambda}")
    unique, counts = np.unique(phase_data.flatten(), return_counts=True)
    inv_map = {0: 'Z≈0', 1: 'Z≈1', 2: 'oscillation', 3: 'mixed'}
    for u, c in zip(unique, counts):
        if u in inv_map:
            print(f"  {inv_map[u]:<15}: {c:>4} punti ({100*c/(n_theta*n_lambda):.1f}%)")

    plt.close('all')


if __name__ == '__main__':
    main()
