import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

# ==========================================
# 1. CONFIGURAZIONE & PARAMETRI
# ==========================================

@dataclass
class SimConfig:
    # Dimensioni Spazio-Tempo
    H: int = 96
    W: int = 96
    dt: float = 0.08
    steps: int = 300  # Durata simulazione
    
    # Fisica del Campo (Reaction-Diffusion)
    dx: float = 1.0
    D: float = 0.45          # Diffusione
    decay: float = 0.25      # Decadimento entropico
    nonlin: float = 0.15     # Non-linearità (Tanh)
    alpha: float = 1.0 / 137.035999084 # Fine structure constant (simbolica)
    omega_scale: float = 80.0
    forcing_amp: float = 0.8 # Energia esterna
    forcing_noise: float = 0.05
    
    # Parametri Topologici (Grafo)
    n_nodes: int = 600
    connect_radius: float = 10.0
    base_edge_prob: float = 0.02
    vmod_gain: float = 2.0
    dag_temperature: float = 0.8
    local_window: int = 9
    seed: int = 42

    # --- PLASTICITÀ ENDOGENA (v1.9 UPDATED) ---
    inertia_min: float = 0.1  # Plasticità massima (Caos)
    inertia_max: float = 0.99 # Rigidità massima (quasi totale lock-in)
    
    # Abbassiamo la sensibilità: basta meno coerenza per attivare la memoria.
    inertia_sensitivity: float = 1.5 
    
    # Fattore di polarizzazione: quanto l'inerzia spinge le probabilità a 0 o 1
    crystallization_power: float = 6.0 

# ==========================================
# 2. MOTORE FISICO (FIELD POTENTIAL)
# ==========================================

def laplacian_periodic(u: np.ndarray) -> np.ndarray:
    """Calcola il Laplaciano con condizioni al contorno periodiche (Toro)."""
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4.0 * u)

def step_fp(u: np.ndarray, t: float, rng: np.random.Generator, cfg: SimConfig) -> np.ndarray:
    """Passo di integrazione temporale (Eulero) per il campo scalare."""
    noise = cfg.forcing_noise * rng.normal(size=u.shape)
    
    # Forcing armonico spazialmente modulato
    omega_alpha = cfg.omega_scale * cfg.alpha
    yy, xx = np.mgrid[0:cfg.H, 0:cfg.W]
    phi = 0.07 * xx + 0.11 * yy
    forcing = cfg.forcing_amp * np.sin(omega_alpha * t + phi)
    
    # Equazione RD
    du = (cfg.D * laplacian_periodic(u) - cfg.decay * u + cfg.nonlin * np.tanh(u))
    u_next = u + cfg.dt * du + cfg.dt * forcing + noise
    return u_next

# ==========================================
# 3. METRICHE DI CAMPO & COERENZA
# ==========================================

def spectral_coherence(fp_np: np.ndarray, eps: float = 1e-12) -> float:
    F = np.fft.rfft2(fp_np)
    P = (F.real**2 + F.imag**2)
    P = P / (P.sum() + eps)
    H = -(P * np.log(P + eps)).sum()
    H_max = np.log(P.size + eps) if P.size > 0 else 1.0
    return 1.0 - (H / H_max)

def v_mod(fp: np.ndarray) -> Tuple[float, float, float]:
    sigma = float(fp.std())
    C = spectral_coherence(fp)
    val = np.mean(fp)
    return float(val), sigma, C

# ==========================================
# 4. MOTORE CAUSALE (GRAFO & PLASTICITÀ)
# ==========================================

def sample_nodes(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    ys = rng.integers(0, cfg.H, size=cfg.n_nodes)
    xs = rng.integers(0, cfg.W, size=cfg.n_nodes)
    return np.stack([ys, xs], axis=1)

def apply_synaptic_contrast(probs: np.ndarray, inertia: float, power: float) -> np.ndarray:
    """
    Spinge le probabilità verso 0.0 o 1.0 in proporzione all'Inerzia.
    Se Inertia ~ 1, le probabilità diventano binarie (Cristallizzazione).
    """
    if inertia < 0.2:
        return probs
        
    # Calcola il punto medio dinamico (per evitare di spegnere tutto se probs sono basse)
    midpoint = np.mean(probs) + 1e-6
    
    # Sigmoide centrata su midpoint, resa più ripida dall'inerzia
    steepness = 1.0 + (power * inertia)
    
    # Formula logistica adattata
    # output = 1 / (1 + exp(-k * (x - mid)))
    # Usiamo una versione vettorializzata efficiente approssimata
    centered = (probs - midpoint) * steepness
    
    # Re-mapping lineare con saturazione (più veloce di exp su matrice grande)
    # x_new = 0.5 + centered -> clip
    probs_polarized = np.clip(0.5 + centered / (2.0 * np.std(probs) + 1e-6), 0.0, 1.0)
    
    # Mix: manteniamo un po' del segnale originale se l'inerzia non è totale
    return (1.0 - inertia) * probs + inertia * probs_polarized

def build_gce_dag(fp: np.ndarray, nodes_yx: np.ndarray, cfg: SimConfig, 
                  rng: np.random.Generator, prev_probs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    
    N = nodes_yx.shape[0]
    y, x = nodes_yx[:, 0], nodes_yx[:, 1]
    values = fp[y, x]
    
    # 1. Distanza
    dy = y[:, None] - y[None, :]
    dx = x[:, None] - x[None, :]
    dist2 = dy*dy + dx*dx
    mask = (dist2 > 0) & (dist2 <= cfg.connect_radius**2)
    
    # 2. Plasticità Endogena
    _, _, C_global = v_mod(fp)
    dynamic_inertia = cfg.inertia_min + (cfg.inertia_max - cfg.inertia_min) * (C_global ** cfg.inertia_sensitivity)
    dynamic_inertia = np.clip(dynamic_inertia, 0.0, 0.99)

    # 3. Preferenza Direzionale (Gradient Flow)
    dv = values[:, None] - values[None, :]
    dir_pref = 1.0 / (1.0 + np.exp(-(dv / max(cfg.dag_temperature, 1e-6))))
    
    # 4. Probabilità Istantanea
    p_inst = cfg.base_edge_prob * (1.0 + 0.5 * dir_pref) 
    p_inst = np.clip(p_inst, 0.0, 0.95)
    
    # 5. Integrazione Temporale
    if prev_probs is not None:
        p_raw = (1.0 - dynamic_inertia) * p_inst + dynamic_inertia * prev_probs
    else:
        p_raw = p_inst
        
    # 6. CRISTALLIZZAZIONE (Nuovo Step v1.9)
    # Se il sistema è in alta inerzia, polarizziamo le probabilità
    p_final = apply_synaptic_contrast(p_raw, dynamic_inertia, cfg.crystallization_power)
    
    # Maschera finale per efficienza (azzera prob fuori raggio)
    p_final = p_final * mask
        
    # 7. Campionamento
    rand_matrix = rng.random((N, N))
    A = (rand_matrix < p_final)
    
    # Condition DAG (Gradient Check)
    A = A & (dv > 0)

    np.fill_diagonal(A, False)
    return A.astype(np.uint8), p_final, float(dynamic_inertia)

# ==========================================
# 5. ANALISI & RUNNER
# ==========================================

def gce_metrics(A_prev: np.ndarray, A_now: np.ndarray) -> Dict[str, float]:
    E_prev = A_prev.astype(bool)
    E_now = A_now.astype(bool)
    intersection = np.logical_and(E_prev, E_now).sum()
    union = np.logical_or(E_prev, E_now).sum()
    S_Causal = float(intersection / max(union, 1))
    edge_count = float(E_now.sum())
    return {"S_Causal": S_Causal, "edge_count": edge_count}

def run_simulation(cfg: SimConfig, regime_name: str) -> Dict[str, Any]:
    print(f"\n>>> Avvio Simulazione: Regime {regime_name.upper()}")
    c = SimConfig(**cfg.__dict__)
    
    if regime_name == "Attrattore":
        c.D = 0.60; c.forcing_amp = 0.55; c.forcing_noise = 0.02; c.decay = 0.28
    elif regime_name == "Repulsore":
        c.D = 0.35; c.forcing_amp = 1.10; c.forcing_noise = 0.09; c.decay = 0.18
    elif regime_name == "Indeterminazione":
        c.D = 0.25; c.forcing_amp = 0.75; c.forcing_noise = 0.15; c.decay = 0.10
    
    rng = np.random.default_rng(c.seed)
    nodes = sample_nodes(c, rng)
    u = rng.normal(size=(c.H, c.W)) * 0.2
    
    history = {"C_FP": [], "Inertia": [], "S_Causal": []}
    prev_probs = None
    A_prev = np.zeros((c.n_nodes, c.n_nodes), dtype=np.uint8)
    
    for t_step in range(c.steps):
        t = t_step * c.dt
        u = step_fp(u, t, rng, c)
        
        A_now, probs_now, inertia_val = build_gce_dag(u, nodes, c, rng, prev_probs)
        
        _, _, C_val = v_mod(u)
        m_graph = gce_metrics(A_prev, A_now)
        
        history["C_FP"].append(C_val)
        history["Inertia"].append(inertia_val)
        history["S_Causal"].append(m_graph["S_Causal"])
        
        prev_probs = probs_now
        A_prev = A_now
        
        if t_step % 50 == 0:
            print(f"   Step {t_step}: C={C_val:.3f} -> I={inertia_val:.3f} | Stb={m_graph['S_Causal']:.3f}")

    return history

def print_summary(results: Dict[str, Dict[str, List[float]]]):
    print("\n" + "="*70)
    print(f"{'REGIME':<18} | {'COERENZA':<10} | {'INERZIA':<10} | {'STABILITÀ (Target > 0.5)':<25}")
    print("-" * 70)
    for regime, hist in results.items():
        warmup = 50
        c_mean = np.mean(hist["C_FP"][warmup:])
        i_mean = np.mean(hist["Inertia"][warmup:])
        s_mean = np.mean(hist["S_Causal"][warmup:])
        print(f"{regime:<18} | {c_mean:<10.3f} | {i_mean:<10.3f} | {s_mean:<25.3f}")
    print("="*70)

if __name__ == "__main__":
    base_cfg = SimConfig()
    regimes = ["Attrattore", "Repulsore", "Indeterminazione"]
    all_results = {}
    for r in regimes:
        all_results[r] = run_simulation(base_cfg, r)
    print_summary(all_results)