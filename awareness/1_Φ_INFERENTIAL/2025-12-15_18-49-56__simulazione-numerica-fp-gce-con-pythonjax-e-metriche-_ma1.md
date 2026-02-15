---
title: "Simulazione Numerica FP-GCE con Python/JAX e Metriche Φ_MA.1"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1765739407863_x02iyj9"
model_id: "openai/gpt-5.2"
domain: "D-ND Cosmology - EUFPossibilità + Fondamenti Metafisici"
created_at: "2025-12-14T19:10:07.863Z"
ended_at: "2025-12-15T17:49:56.931Z"
---
# Simulazione Numerica FP-GCE con Python/JAX e Metriche Φ_MA.1

## Ultima risposta assistente

Di seguito trovi un **prototipo numerico FP→GCE** in **Python + JAX** per testare le metriche di **Φ_MA.1**:

- **FP** (campo di Frequenza di Possibilità): simulato come **campo scalare 2D** con dinamica tipo *reaction–diffusion* + forcing **α-modulato**.
- **Metriche FP**:
  - `sigma_FP(x)` (qui: versione globale + locale su finestre)
  - `C_FP_global` (coerenza spettrale globale via entropia spettrale)
  - `Lambda_FP` (tasso di modulazione: norma media della derivata temporale)
- **GCE** (Gerarchie Causali Emergenti): costruite come **DAG** (grafo orientato) a partire dal campo FP tramite un **potenziale modulatorio** `V_mod(FP)` (regime attrattivo/repulsivo).
- **Metriche GCE**:
  - `G_C` (granularità: profondità media dei livelli + branching)
  - `S_Causal` (stabilità: similarità tra grafi consecutivi / persistenza archi)
  - `Gamma_CS` (tasso di generazione complessità: d/dt di G_C)

---

## 1) File unico: `fp_gce_sim_jax.py`

Copia e salva:

```python
# fp_gce_sim_jax.py
# Prototype FP-GCE metrics (Phi_MA.1 inspired) using JAX for field simulation
# Graph construction/metrics are numpy-based for simplicity (still fast for N<=2k)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    raise SystemExit("Install JAX: pip install 'jax[cpu]'") from e

# -----------------------------
# Config
# -----------------------------

@dataclass
class SimConfig:
    # Lattice
    H: int = 128
    W: int = 128
    dx: float = 1.0
    dt: float = 0.08
    steps: int = 300

    # Reaction-diffusion-ish dynamics (stable-ish toy)
    D: float = 0.45          # diffusion strength
    decay: float = 0.25      # damping
    nonlin: float = 0.15     # mild nonlinearity

    # Alpha modulation
    alpha: float = 1.0 / 137.035999084  # fine-structure constant
    omega_scale: float = 80.0           # omega_alpha = omega_scale * alpha
    forcing_amp: float = 0.8            # forcing amplitude
    forcing_noise: float = 0.05         # additive noise amplitude

    # FP->GCE mapping
    n_nodes: int = 800
    connect_radius: float = 10.0        # in lattice coordinates
    base_edge_prob: float = 0.02
    vmod_gain: float = 2.0              # how strongly V_mod modulates edges
    dag_temperature: float = 0.8        # softness of directionality

    # Metrics
    local_window: int = 9               # window size for local sigma
    seed: int = 0

# -----------------------------
# FP field dynamics (JAX)
# -----------------------------

def laplacian_periodic(u: jnp.ndarray) -> jnp.ndarray:
    # 2D periodic laplacian (5-point stencil)
    return (
        jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
        jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) -
        4.0 * u
    )

@jax.jit
def step_fp(u: jnp.ndarray, t: float, key: jax.random.PRNGKey, cfg: SimConfig) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
    """
    FP update:
      u_{t+1} = u + dt*(D*Lap(u) - decay*u + nonlin*tanh(u)) + forcing(alpha,t,x) + noise
    forcing(alpha,t,x) is alpha-modulated: A*sin(omega_alpha*t + phi(x))
    """
    key, sub = jax.random.split(key)
    noise = cfg.forcing_noise * jax.random.normal(sub, shape=u.shape)

    omega_alpha = cfg.omega_scale * cfg.alpha  # alpha-modulated frequency
    # spatial phase pattern to create structured forcing
    yy, xx = jnp.mgrid[0:cfg.H, 0:cfg.W]
    phi = 0.07 * xx + 0.11 * yy
    forcing = cfg.forcing_amp * jnp.sin(omega_alpha * t + phi)

    du = (
        cfg.D * laplacian_periodic(u) -
        cfg.decay * u +
        cfg.nonlin * jnp.tanh(u)
    )
    u_next = u + cfg.dt * du + cfg.dt * forcing + noise
    return u_next, key

# -----------------------------
# FP metrics (mix: JAX -> numpy for FFT convenience)
# -----------------------------

def spectral_coherence(fp_np: np.ndarray, eps: float = 1e-12) -> float:
    """
    C_FP_global: 1 - normalized spectral entropy.
    High if power concentrated in few frequencies (coherent), low if spread (noisy).
    """
    F = np.fft.rfft2(fp_np)
    P = (F.real**2 + F.imag**2)
    P = P / (P.sum() + eps)
    H = -(P * np.log(P + eps)).sum()
    H_norm = H / (np.log(P.size + eps))
    return float(1.0 - H_norm)

def local_sigma(fp_np: np.ndarray, w: int) -> float:
    """
    sigma_FP(x): local std averaged over all window centers (naive, O(HW w^2)).
    For prototype sizes it's OK; optimize later via convolution if needed.
    """
    H, W = fp_np.shape
    r = w // 2
    acc = 0.0
    cnt = 0
    for y in range(r, H - r):
        for x in range(r, W - r):
            patch = fp_np[y-r:y+r+1, x-r:x+r+1]
            acc += patch.std()
            cnt += 1
    return float(acc / max(cnt, 1))

def fp_metrics(fp_prev: np.ndarray, fp_now: np.ndarray, cfg: SimConfig) -> Dict[str, float]:
    sigma_global = float(fp_now.std())
    sigma_local = local_sigma(fp_now, cfg.local_window)
    C_global = spectral_coherence(fp_now)
    Lambda = float(np.mean(np.abs((fp_now - fp_prev) / cfg.dt)))  # avg |d/dt|
    return {
        "sigma_FP_global": sigma_global,
        "sigma_FP_local_mean": sigma_local,
        "C_FP_global": C_global,
        "Lambda_FP": Lambda,
    }

# -----------------------------
# FP -> V_mod(FP) (Phi_MA.1 inspired)
# -----------------------------

def v_mod(fp: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Returns (V_mod, sigma_FP, C_FP_global).
    Using the toy forms from your docs:
      V_attr  ~ - (C^2) * exp(-sigma^2)
      V_rep   ~ + (sigma^4) * exp(-C)
    We blend them by a gating g = sigmoid( a*(sigma - s0) - b*(C - c0) ).
    """
    sigma = float(fp.std())
    C = spectral_coherence(fp)
    V_attr = - (C**2) * math.exp(-(sigma**2))
    V_rep = + ((sigma**2)**2) * math.exp(-(C))
    # gate: when sigma high and C low -> repulsive
    s0, c0 = 0.8, 0.5
    a, b = 3.0, 3.0
    g = 1.0 / (1.0 + math.exp(-(a*(sigma - s0) - b*(C - c0))))
    V = (1.0 - g) * V_attr + g * V_rep
    return float(V), sigma, C

# -----------------------------
# GCE construction: sample nodes on lattice, build DAG by FP-guided probability
# -----------------------------

def sample_nodes(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    ys = rng.integers(0, cfg.H, size=cfg.n_nodes)
    xs = rng.integers(0, cfg.W, size=cfg.n_nodes)
    return np.stack([ys, xs], axis=1)  # (N,2)

def build_gce_dag(fp: np.ndarray, nodes_yx: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Build adjacency matrix A (N,N) for a DAG:
    - edge candidate if distance <= connect_radius
    - base probability modulated by V_mod(fp) and local FP gradient/ordering
    - direction: from higher FP to lower FP (softened by dag_temperature)
    """
    N = nodes_yx.shape[0]
    y = nodes_yx[:, 0]
    x = nodes_yx[:, 1]
    values = fp[y, x]

    # pairwise distances (numpy, O(N^2) ok for N<=2k)
    dy = y[:, None] - y[None, :]
    dx = x[:, None] - x[None, :]
    dist2 = dy*dy + dx*dx
    mask = (dist2 > 0) & (dist2 <= cfg.connect_radius**2)

    V, sigma, C = v_mod(fp)
    # If V is negative (attr), boost; if positive (rep), damp
    # map V in [-1, +1] roughly: use tanh
    v = math.tanh(cfg.vmod_gain * V)

    # directionality: prefer from high->low (causal "downhill"), but allow softness
    dv = values[:, None] - values[None, :]  # dv>0 means i higher than j
    dir_pref = 1.0 / (1.0 + np.exp(-(dv / max(cfg.dag_temperature, 1e-6))))

    # probability field
    p0 = cfg.base_edge_prob
    p = p0 * (1.0 + 0.8 * dir_pref) * (1.0 - 0.5 * (v > 0)) * (1.0 + 0.7 * (v < 0))
    p = np.clip(p, 0.0, 0.7)

    # sample edges
    U = rng.random((N, N))
    A = (U < p) & mask

    # enforce DAG by removing edges that go "uphill" too strongly:
    # keep only edges where dv is positive w.r.t. threshold sampled by dir_pref
    A = A & (rng.random((N, N)) < dir_pref)
    np.fill_diagonal(A, False)
    return A.astyp
…

## Note

- modalità: last-assistant
- model: openai/gpt-5.2
- domain: D-ND Cosmology - EUFPossibilità + Fondamenti Metafisici
