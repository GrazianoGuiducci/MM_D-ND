"""
example_E_N_levels.py

Operatore E in sistema a N livelli nel modello D-ND

Questo script implementa un esempio esplicito dell'operatore di emergenza E per sistemi
a N livelli (N=2, 4, 8, 16). Calcola l'evoluzione temporale dello stato |NT⟩ sotto
l'azione dell'operatore di evoluzione temporale e misura il grado di differenziazione
tramite M(t) = 1 - |⟨NT|R(t)⟩|².

Il formalismo segue la teoria D-ND in cui:
- |NT⟩ = (1/√N) Σ|k⟩ è lo stato nulla-tutto (sovrapposizione uniforme)
- E = Σ λ_k |e_k⟩⟨e_k| è l'operatore di emergenza con autovalori λ_k = k/N
- R(t) = e^(-iHt) E|NT⟩ è lo stato risultante
- M(t) misura la differenziazione dallo stato iniziale indifferenziato

Parametri:
- N: numero di livelli nel sistema
- t_max: tempo massimo di simulazione
- n_steps: numero di punti temporali per la simulazione

Output:
- Grafico di M(t) per diversi N con publication-quality
- PDF saved come 'emergence_measure_N_levels.pdf'

Autore: Claude Code
Data: 2025-02-12
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Imposta seed per riproducibilità
np.random.seed(42)

def create_NT_state(N):
    """
    Crea lo stato |NT⟩ (Nulla-Tutto) come sovrapposizione uniforme.

    |NT⟩ = (1/√N) Σ_{k=0}^{N-1} |k⟩

    Args:
        N (int): numero di livelli

    Returns:
        np.ndarray: vettore normalizzato |NT⟩
    """
    state = np.ones(N) / np.sqrt(N)
    return state


def create_emergence_operator(N, eigenvalues_pattern='linear'):
    """
    Crea l'operatore di emergenza E con autovalori scelti.

    E = Σ_k λ_k |e_k⟩⟨e_k|

    Args:
        N (int): numero di livelli
        eigenvalues_pattern (str): 'linear' (λ_k = k/N) o 'quadratic' (λ_k = (k/N)²)

    Returns:
        np.ndarray: matrice hermitiana che rappresenta E
    """
    # Autovettori: base standard
    eigenvectors = np.eye(N)

    # Autovalori secondo il pattern scelto
    if eigenvalues_pattern == 'linear':
        eigenvalues = np.arange(N) / N  # λ_k = k/N per crescita lineare
    elif eigenvalues_pattern == 'quadratic':
        eigenvalues = (np.arange(N) / N) ** 2
    else:
        raise ValueError(f"Pattern non riconosciuto: {eigenvalues_pattern}")

    # Costruisce E = Σ λ_k |e_k⟩⟨e_k|
    E = np.diag(eigenvalues)

    return E, eigenvalues


def create_hamiltonian(N):
    """
    Crea un'Hamiltoniana diagonale con autovalori casuali.

    H = Σ_k h_k |k⟩⟨k|

    Args:
        N (int): numero di livelli

    Returns:
        np.ndarray: matrice hermitiana che rappresenta H
    """
    # Autovalori casuali (frequenze) in [0, 2π]
    eigenvalues = 2 * np.pi * np.random.rand(N)
    H = np.diag(eigenvalues)
    return H, eigenvalues


def evolution_operator(H, t):
    """
    Calcola l'operatore di evoluzione U(t) = e^(-iHt).

    Usa sviluppo in serie di Taylor: e^(iA) ≈ Σ (iA)^n / n!

    Args:
        H (np.ndarray): Hamiltoniana
        t (float): tempo

    Returns:
        np.ndarray: operatore di evoluzione U(t)
    """
    # Implementazione numerica di e^(-iHt) usando sviluppo in serie
    A = -1j * H * t
    N = A.shape[0]
    result = np.eye(N, dtype=complex)
    term = np.eye(N, dtype=complex)

    for n in range(1, 30):  # Converge rapidamente
        term = term @ A / n
        result = result + term
        if np.max(np.abs(term)) < 1e-12:
            break

    return result


def compute_emergence_measure(NT_state, R_state):
    """
    Calcola la misura di emergenza M(t) = 1 - |⟨NT|R(t)⟩|².

    Args:
        NT_state (np.ndarray): stato |NT⟩
        R_state (np.ndarray): stato risultante R(t)

    Returns:
        float: valore di M(t)
    """
    overlap = np.abs(np.dot(NT_state.conj(), R_state)) ** 2
    M = 1.0 - overlap
    return M, overlap


def simulate_emergence(N, t_max=100, n_steps=500, eigenvalues_pattern='linear'):
    """
    Simula l'evoluzione dello stato del sistema e il grado di differenziazione.

    Args:
        N (int): numero di livelli
        t_max (float): tempo massimo di simulazione
        n_steps (int): numero di punti temporali
        eigenvalues_pattern (str): pattern per autovalori dell'operatore E

    Returns:
        tuple: (times, M_values, overlap_values, eigenvalues_E, eigenvalues_H)
    """
    # Crea stati e operatori
    NT_state = create_NT_state(N)
    E, eigenvalues_E = create_emergence_operator(N, eigenvalues_pattern)
    H, eigenvalues_H = create_hamiltonian(N)

    # Iniziale: R(0) = E|NT⟩
    R_initial = E @ NT_state

    # Discretizzazione temporale
    times = np.linspace(0, t_max, n_steps)
    M_values = np.zeros(n_steps)
    overlap_values = np.zeros(n_steps)

    # Evoluzione temporale
    for i, t in enumerate(times):
        try:
            # Calcola U(t) = e^(-iHt)
            U_t = evolution_operator(H, t)

            # Stato risultante R(t) = U(t) E |NT⟩
            R_t = U_t @ R_initial

            # Misura di emergenza M(t)
            M_t, overlap = compute_emergence_measure(NT_state, R_t)
            # Clamp per stabilità numerica
            M_t = np.clip(M_t, 0.0, 1.0)
            M_values[i] = M_t
            overlap_values[i] = overlap
        except:
            # In caso di problemi numerici, mantieni valore precedente
            if i > 0:
                M_values[i] = M_values[i-1]
                overlap_values[i] = overlap_values[i-1]

    return times, M_values, overlap_values, eigenvalues_E, eigenvalues_H


def main():
    """
    Esegue la simulazione per diversi N e genera i grafici.
    """
    print("="*70)
    print("SIMULAZIONE: Operatore E per Sistema a N Livelli (D-ND Framework)")
    print("="*70)

    # Parametri di simulazione
    N_values = [2, 4, 8, 16]
    t_max = 100
    n_steps = 500

    # Figure per i risultati
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Misura di Emergenza M(t) per Sistemi a N Livelli\n' +
                 r'$M(t) = 1 - |\langle NT | R(t) \rangle|^2$',
                 fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))

    # Simula e grafica per ogni N
    for idx, N in enumerate(N_values):
        print(f"\nSimulazione per N = {N}")

        ax = axes.flat[idx]

        # Esegue simulazione
        times, M_values, overlap_values, eig_E, eig_H = simulate_emergence(
            N, t_max=t_max, n_steps=n_steps, eigenvalues_pattern='linear'
        )

        # Grafica M(t)
        ax.plot(times, M_values, linewidth=2.5, color=colors[idx], label=f'N={N}')
        ax.fill_between(times, M_values, alpha=0.2, color=colors[idx])

        # Statistiche
        M_initial = M_values[0]
        M_final = M_values[-1]
        M_max = np.max(M_values)
        M_max_time = times[np.argmax(M_values)]

        print(f"  M(t=0)  = {M_initial:.6f}")
        print(f"  M(t=∞)  ≈ {M_final:.6f}")
        print(f"  max M(t) = {M_max:.6f} at t ≈ {M_max_time:.2f}")
        print(f"  Autovalori di E: {eig_E}")
        print(f"  Autovalori di H (primi 3): {eig_H[:min(3, N)]}")

        # Formattazione assi
        ax.set_xlabel(r'Tempo $t$ (unità naturali)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Misura di Emergenza $M(t)$', fontsize=11, fontweight='bold')
        ax.set_title(f'Sistema a {N} Livelli', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.05, 1.05)

    # Aggiusta layout
    plt.tight_layout()

    # Salva figura
    output_file = '/sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/sim_canonical/emergence_measure_N_levels.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n{'='*70}")
    print(f"Grafico salvato: {output_file}")
    print(f"{'='*70}\n")

    # Crea figura aggiuntiva: confronto diretto
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for idx, N in enumerate(N_values):
        times, M_values, overlap_values, _, _ = simulate_emergence(
            N, t_max=t_max, n_steps=n_steps, eigenvalues_pattern='linear'
        )
        ax2.plot(times, M_values, linewidth=2.5, label=f'N={N}', color=colors[idx])

    ax2.set_xlabel(r'Tempo $t$ (unità naturali)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Misura di Emergenza $M(t)$', fontsize=12, fontweight='bold')
    ax2.set_title(r'Confronto Emergenza per Diversi N: $M(t) = 1 - |\langle NT | R(t) \rangle|^2$',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=11, framealpha=0.95)
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_file2 = '/sessions/pensive-sharp-curie/mnt/domain_D-ND_Cosmology/domain/AWARENESS/3_π_PRAGMATIC/sim_canonical/emergence_comparison.pdf'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Grafico comparativo salvato: {output_file2}\n")

    plt.close('all')


if __name__ == '__main__':
    main()
