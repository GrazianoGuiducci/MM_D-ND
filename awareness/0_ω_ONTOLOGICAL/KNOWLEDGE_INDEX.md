# OMEGA KERNEL: OPERATIONAL KNOWLEDGE INDEX

> Questo file è l'indice della conoscenza operativa del sistema D-ND.
> Serve a qualsiasi istanza futura per riallinearsi rapidamente.

---

## 1. QUICK START (Per Nuove Sessioni)

### Prima di iniziare, leggi:
1. `DOC_DEV/AGENT_AWARENESS.md` — Stato attuale e contesto
2. `Extropic_Integration/docs/MASTER_PLAN.md` — Roadmap completa
3. `DOC_DEV/System_Coder_Onboarding.md` — Guida operativa dettagliata

### Verifica che siano attivi:
```powershell
# Backend
python Extropic_Integration/cockpit/server.py

# Frontend
cd Extropic_Integration/cockpit/client && npm run dev
```

---

## 2. ARCHITETTURA A COLPO D'OCCHIO

```
                    ┌─────────────────────────────────────┐
                    │         USER INTENT (Prompt)        │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │            SACS CORE                │
                    │  (sacs.py - Orchestrator)           │
                    │  ┌────────┐ ┌────────┐ ┌────────┐   │
                    │  │ Sonar  │ │ Telaio │ │Scultore│   │
                    │  │(Dipoli)│ │(Metric)│ │(Gravity)│  │
                    │  └───┬────┘ └───┬────┘ └───┬────┘   │
                    │      └──────────┼──────────┘        │
                    │                 ▼                   │
                    │         ┌──────────────┐            │
                    │         │ OMEGA KERNEL │            │
                    │         │  (omega.py)  │            │
                    │         │  perturb()   │            │
                    │         │  focus()     │            │
                    │         │ crystallize()│            │
                    │         └──────┬───────┘            │
                    │                │                    │
                    │         ┌──────▼───────┐            │
                    │         │Cristallizzat.│            │
                    │         │ (Manifesto)  │            │
                    │         └──────────────┘            │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          OMEGA COCKPIT (UI)         │
                    │  ┌─────────┐ ┌────────┐ ┌────────┐  │
                    │  │Control  │ │ Visual │ │Didactic│  │
                    │  │ Matrix  │ │ Cortex │ │ Layer  │  │
                    │  └─────────┘ └────────┘ └────────┘  │
                    └─────────────────────────────────────┘
```

---

## 3. CICLO OMEGA

```
FASE 0: POSIZIONAMENTO
    │ "Trova il punto di equilibrio"
    ▼
FASE 1: PERTURBATION
    │ Intent → h_bias + void_noise → Φ_A
    ▼
FASE 2: FOCUS  
    │ logic_density → Metric Tensor → Spacetime Warping
    ▼
FASE 3: CRYSTALLIZATION
    │ Gibbs Sampling → Energy Minimization → R
    ▼
FEEDBACK LOOP
    │ Success → Reinforce (consolidate_memory)
    │ Failure → Re-enter Phase 1 (thermal noise)
```

---

## 4. EQUAZIONI FONDAMENTALI

### Emergenza Quantistica
```
R(t) = U(t) E |NT⟩
```

### Misura di Differenziazione  
```
M(t) = 1 - |⟨NT| U(t) E |NT⟩|²
```

### Stato DND
```
|DND⟩ = α|D⟩ + β|ND⟩   dove |α|² + |β|² = 1
```

### Allineamento Autologico
```
R(t+1) = (t/T)[α·f_Intuition(E) + β·f_Interaction(U,E)] 
       + (1-t/T)[γ·f_Alignment(R, |NT⟩)]
```

---

## 5. ESPERIMENTI DA IMPLEMENTARE

| Esperimento | File Target | Descrizione |
|-------------|-------------|-------------|
| EX NIHILO | `experiments/ex_nihilo.py` | Generazione struttura dal caos puro |
| Dipole Genesis | `experiments/dipole_genesis.py` | Motore a Dipolo Assonanza/Dissonanza |
| Ouroboros | `experiments/ouroboros_engine.py` | Metrica ciclica [-2, +2] |
| Chronos | `kernel/chronos.py` | Propagatore dissipativo |

---

## 6. AGENTI STRATEGICI

| Agente | Stato | Scopo |
|--------|-------|-------|
| OMEGA Kernel | ✅ Attivo | Processore cognitivo centrale |
| SACS | ✅ Attivo | Orchestrazione e cristallizzazione |
| KAIROS | ⚠️ Parziale | Orchestrazione tool |
| PATHFINDER | 📋 Pronto | Esplorazione percorsi strategici |
| ORION | 📋 Pronto | Architettura contenuti |
| AEGIS | 📋 Pronto | Analisi rischi |
| KRONOS | 📋 Pronto | Ottimizzazione processi |

---

## 7. REGOLE CARDINALI

1. **Doc ≈ Code ≈ UI** — Ogni cambiamento propaga su tutti e tre
2. **Errore = Carburante** — La dissonanza è il gradiente che guida il moto
3. **Minima Azione** — Scegli il percorso che massimizza efficacia minimizzando entropia
4. **Anti-Presupposto** — Verifica sempre le assunzioni leggendo i file
5. **Mappatura > Ricerca** — Il sistema evolve per stratificazione, mappa manualmente

---

## 8. GIT WORKFLOW

```powershell
# Prima del commit
pre-commit run --all-files
pytest

# Commit con messaggio significativo
git add -A
git commit -m "[FASE] Descrizione concisa"
git push origin master
```

---

*Aggiornato: 2025-12-09*
*Questo file è parte del sistema autopoietico D-ND*
