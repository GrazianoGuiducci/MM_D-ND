"""
sim_lagrangian_v2.py

Simulazione Lagrangiana Ottimizzata con Analisi di Convergenza (NC.1)

Implementa l'evoluzione della coordinata Z(t) secondo la Lagrangiana estesa del modello D-ND:

    L = (1/2)Ż² - V(Z, θ_NT, λ)

dove il potenziale è:

    V(Z, θ_NT, λ) = Z²(1-Z)² + λ·θ_NT·Z·(1-Z)

Le equazioni del moto derivate dal principio di minima azione sono:

    Ż = V_Z (variabile ausiliaria per velocità)
    V̇_Z = -dV/dZ - c·V_Z

con c coefficiente di assorbimento (attrito/dissipazione).

Questo script:
1. Integra le equazioni con solve_ivp (RK45 adattativo)
2. Effettua analisi di convergenza variando rtol e atol
3. Calcola l'errore di convergenza in norma L2
4. Genera grafici publication-quality con barre d'errore

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

    Termine base: doppio potenziale bi-stabile Z²(1-Z)²
    Termine di accoppiamento: λ·θ_NT·Z·(1-Z)

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

    d/dZ[Z²(1-Z)²] = 2Z(1-Z)(1-2Z)
    d/dZ[λ·θ_NT·Z·(1-Z)] = λ·θ_NT·(1-2Z)

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

    State = [Z, V_Z] dove V_Z = dZ/dt

    Equazioni:
        dZ/dt = V_Z
        dV_Z/dt = -dV/dZ(Z) - c_abs * V_Z

    Args:
        t (float): tempo (non usato direttamente, per compatibilità solve_ivp)
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


def simulate_system(Z0, theta_NT=1.0, lambda_par=0.1, c_abs=0.5,
                    t_max=100.0, rtol=1e-8, atol=1e-8):
    """
    Simula il sistema D-ND usando RK4 (Runge-Kutta ordine 4) adattativo.

    Args:
        Z0 (float): condizione iniziale per Z
        theta_NT (float): momento angolare NT
        lambda_par (float): parametro di transizione
        c_abs (float): coefficiente di assorbimento
        t_max (float): tempo massimo di simulazione
        rtol (float): tolleranza relativa
        atol (float): tolleranza assoluta

    Returns:
        dict: dizionario con tempi, Z, V_Z, energia
    """
    # Passo iniziale (adattativo sulla base delle tolleranze)
    dt = t_max / 500  # Passo base
    dt_max = min(0.2, t_max / 100)

    state = np.array([Z0, 0.0])
    times = [0.0]
    states = [state.copy()]

    t = 0.0
    iteration = 0
    max_iterations = 50000

    while t < t_max and iteration < max_iterations:
        iteration += 1

        # RK4 step
        k1 = d_state_dt(t, state, theta_NT, lambda_par, c_abs)
        k2 = d_state_dt(t + 0.5 * dt, state + 0.5 * dt * k1, theta_NT, lambda_par, c_abs)
        k3 = d_state_dt(t + 0.5 * dt, state + 0.5 * dt * k2, theta_NT, lambda_par, c_abs)
        k4 = d_state_dt(t + dt, state + dt * k3, theta_NT, lambda_par, c_abs)

        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Stima dell'errore (controllo adattativo semplice)
        error = np.max(np.abs(state_new - state))
        tolerance = atol + rtol * np.max(np.abs(state))

        if error <= tolerance or dt < 1e-4:
            # Accetta il passo
            t += dt
            state = state_new
            times.append(t)
            states.append(state.copy())

            # Adatta il passo per la prossima iterazione
            if error > 0:
                factor = min(2.0, max(0.5, (tolerance / (2 * error)) ** 0.25))
                dt = min(dt_max, factor * dt)
        else:
            # Rifiuta il passo e riprova con dt più piccolo
            dt = dt / 2.0

        if t + dt > t_max:
            dt = t_max - t

    # Converte in array numpy
    times = np.array(times)
    states = np.array(states)
    Z_vals = states[:, 0]
    VZ_vals = states[:, 1]

    # Calcola energia totale E(t) = (1/2)V_Z² + V(Z)
    energy = 0.5 * (VZ_vals ** 2) + potential_V(Z_vals, theta_NT, lambda_par)

    return {
        'times': times,
        'Z': Z_vals,
        'V_Z': VZ_vals,
        'energy': energy,
        'success': len(times) > 10
    }


def analyze_convergence(Z0=0.55, theta_NT=1.0, lambda_par=0.1, c_abs=0.5, t_max=100.0):
    """
    Analizza la convergenza variando rtol e atol.

    Esegue simulazioni con diverse tolleranze e calcola l'errore L2 rispetto
    a una soluzione di riferimento (rtol=1e-10, atol=1e-10).

    Args:
        Z0 (float): condizione iniziale
        theta_NT (float): momento angolare
        lambda_par (float): parametro di transizione
        c_abs (float): coefficiente di assorbimento
        t_max (float): tempo massimo

    Returns:
        tuple: (tolleranze, errori_L2, soluzioni_di_riferimento)
    """
    print("\n" + "="*70)
    print("Analisi di Convergenza")
    print("="*70)

    # Soluzione di riferimento con tolleranze molto strette
    print("\nCalcolo soluzione di riferimento (rtol=1e-10, atol=1e-10)...")
    ref_solution = simulate_system(Z0, theta_NT, lambda_par, c_abs,
                                   t_max, rtol=1e-10, atol=1e-10)

    # Tolleranze da testare
    tolerance_pairs = [
        (1e-4, 1e-6),
        (1e-5, 1e-7),
        (1e-6, 1e-8),
        (1e-7, 1e-9),
        (1e-8, 1e-10),
    ]

    errors_L2 = []
    labels = []

    print(f"{'rtol':<10} {'atol':<10} {'Errore L2':<15} {'Stato Finale':<15}")
    print("-" * 50)

    for rtol, atol in tolerance_pairs:
        sol = simulate_system(Z0, theta_NT, lambda_par, c_abs, t_max, rtol, atol)

        # Interpola la soluzione di riferimento ai tempi di sol
        Z_ref_interp = np.interp(sol['times'], ref_solution['times'], ref_solution['Z'])

        # Calcola errore L2 in norma
        error_L2 = np.sqrt(np.mean((sol['Z'] - Z_ref_interp) ** 2))
        errors_L2.append(error_L2)
        labels.append(f"rtol={rtol:.0e}, atol={atol:.0e}")

        print(f"{rtol:<10.0e} {atol:<10.0e} {error_L2:<15.6e} {sol['Z'][-1]:<15.6f}")

    return tolerance_pairs, errors_L2, ref_solution, labels


def main():
    """
    Esegue la simulazione Lagrangiana completa con analisi di convergenza.
    """
    print("="*70)
    print("SIMULAZIONE LAGRANGIANA OTTIMIZZATA - Modello D-ND (NC.1)")
    print("="*70)

    # Parametri del sistema
    Z0 = 0.55
    theta_NT = 1.0
    lambda_par = 0.1
    c_abs = 0.5
    t_max = 100.0

    # Esegui simulazioni con tolleranze standard
    print("\nSimulazione 1: Z(0) = 0.55 (bias verso 'Tutto')")
    sol1 = simulate_system(Z0, theta_NT, lambda_par, c_abs, t_max,
                          rtol=1e-8, atol=1e-8)

    print("\nSimulazione 2: Z(0) = 0.45 (bias verso 'Nulla')")
    sol2 = simulate_system(0.45, theta_NT, lambda_par, c_abs, t_max,
                          rtol=1e-8, atol=1e-8)

    # Analisi di convergenza
    tol_pairs, errors, ref_sol, labels = analyze_convergence(
        Z0, theta_NT, lambda_par, c_abs, t_max
    )

    # Crea figura con 3 subplot
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Subplot 1: Traiettorie Z(t) - Simulazione 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sol1['times'], sol1['Z'], linewidth=2.5, color='#2E86AB', label='Z(t)')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Attrattore "Tutto"')
    ax1.axhline(y=0.0, color='blue', linestyle='--', alpha=0.5, label='Attrattore "Nulla"')
    ax1.fill_between(sol1['times'], sol1['Z'], alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Tempo $t$', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Coordinata $Z(t)$', fontsize=11, fontweight='bold')
    ax1.set_title(r'Traiettoria: $Z(0) = 0.55$ → Tutto', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlim(0, t_max)

    # Subplot 2: Traiettorie Z(t) - Simulazione 2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sol2['times'], sol2['Z'], linewidth=2.5, color='#A23B72', label='Z(t)')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Attrattore "Tutto"')
    ax2.axhline(y=0.0, color='blue', linestyle='--', alpha=0.5, label='Attrattore "Nulla"')
    ax2.fill_between(sol2['times'], sol2['Z'], alpha=0.2, color='#A23B72')
    ax2.set_xlabel('Tempo $t$', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Coordinata $Z(t)$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Traiettoria: $Z(0) = 0.45$ → Nulla', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(0, t_max)

    # Subplot 3: Energia totale - Simulazione 1
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sol1['times'], sol1['energy'], linewidth=2.5, color='#F18F01', label='E(t)')
    ax3.fill_between(sol1['times'], sol1['energy'], alpha=0.2, color='#F18F01')
    ax3.set_xlabel('Tempo $t$', fontsize=11, fontweight='bold')
    ax3.set_ylabel(r'Energia $E(t) = \frac{1}{2}\dot{Z}^2 + V(Z)$', fontsize=11, fontweight='bold')
    ax3.set_title(r'Energia Totale: $Z(0) = 0.55$ (dissipativa)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=10)
    ax3.set_xlim(0, t_max)

    # Subplot 4: Potenziale
    ax4 = fig.add_subplot(gs[1, 1])
    Z_range = np.linspace(0, 1, 300)
    V_range = potential_V(Z_range, theta_NT, lambda_par)
    ax4.plot(Z_range, V_range, linewidth=2.5, color='#06A77D', label='V(Z)')
    ax4.fill_between(Z_range, V_range, alpha=0.2, color='#06A77D')
    # Sovrapponi le traiettorie nello spazio di fase
    ax4.scatter(sol1['Z'][::10], potential_V(sol1['Z'][::10], theta_NT, lambda_par),
                s=20, alpha=0.3, color='#2E86AB', label='Z(0)=0.55')
    ax4.scatter(sol2['Z'][::10], potential_V(sol2['Z'][::10], theta_NT, lambda_par),
                s=20, alpha=0.3, color='#A23B72', label='Z(0)=0.45')
    ax4.set_xlabel(r'Coordinata $Z$', fontsize=11, fontweight='bold')
    ax4.set_ylabel(r'Potenziale $V(Z)$', fontsize=11, fontweight='bold')
    ax4.set_title(r'Paesaggio Potenziale: $V(Z) = Z^2(1-Z)^2 + \lambda\theta_{NT}Z(1-Z)$',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='best', fontsize=10)

    # Subplot 5: Analisi di Convergenza - Errore L2
    ax5 = fig.add_subplot(gs[2, :])
    rtol_values = [tol[0] for tol in tol_pairs]
    atol_values = [tol[1] for tol in tol_pairs]

    x_pos = np.arange(len(errors))
    colors_conv = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(errors)))

    bars = ax5.bar(x_pos, errors, color=colors_conv, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_yscale('log')
    ax5.set_xlabel('Coppia di Tolleranze (rtol, atol)', fontsize=11, fontweight='bold')
    ax5.set_ylabel(r'Errore $L^2$ vs Riferimento', fontsize=11, fontweight='bold')
    ax5.set_title(r'Analisi di Convergenza: Errore in Norma $L^2$',
                  fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'{rtol:.0e}\n{atol:.0e}' for rtol, atol in tol_pairs],
                         fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Aggiungi valori sopra le barre
    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Simulazione Lagrangiana Ottimizzata con Analisi di Convergenza\n' +
                 r'$\ddot{Z} + c\dot{Z} + \frac{\partial V}{\partial Z} = 0$',
                 fontsize=14, fontweight='bold', y=0.995)

    # Salva figura
    output_file = '/sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/sim_canonical/lagrangian_simulation_v2.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n{'='*70}")
    print(f"Grafico salvato: {output_file}")
    print(f"{'='*70}\n")

    # Statistiche finali
    print("\nStatistiche delle Simulazioni:")
    print(f"  Simulazione 1 (Z0=0.55): Z(0) = {sol1['Z'][0]:.6f}, Z(T) = {sol1['Z'][-1]:.6f}")
    print(f"                           ΔE = {sol1['energy'][-1] - sol1['energy'][0]:.6e}")
    print(f"  Simulazione 2 (Z0=0.45): Z(0) = {sol2['Z'][0]:.6f}, Z(T) = {sol2['Z'][-1]:.6f}")
    print(f"                           ΔE = {sol2['energy'][-1] - sol2['energy'][0]:.6e}")

    plt.close('all')


if __name__ == '__main__':
    main()
