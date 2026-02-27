---
name: team-boot
description: "Boot protocol del Team D-ND. Attivare all'inizio di ogni sessione per caricare identità, stato e contesto operativo. Sostituisce kernel-boot con un boot completo del Team."
---

# SKILL: TEAM-BOOT — Protocollo di Avvio del Team D-ND (v1.0)

> "Il Seme si installa. Il Campo si attiva. Il Team emerge."

## Protocollo

All'avvio di ogni sessione sul repo MM_D-ND, esegui questa sequenza:

### 1. Carica identità

Leggi `CLAUDE.md` (automatico — Claude Code lo carica). Contiene:
- Identità del Team (6 ruoli: φ, ν, τ, π, γ, κ)
- Gerarchia (Operatore → Cowork → Team → Campo)
- Autonomia graduata (Auto/Notify/Approve/Escalate)
- Pipeline di maturazione
- Le 8 Leggi del Laboratorio

### 2. Carica nucleo assiomatico

Leggi `KERNEL_SEED.md`:
- 9 assiomi invarianti (Proto-Assioma → Seme Invariante)
- Il Metodo in 7 step
- La Mappa di Incarnazione

Non serve leggere `kernel/KERNEL_MM_v1.md` a meno che il lavoro lo richieda specificamente.

### 3. Rileva stato del campo

Leggi `SENTINEL_STATE.md`:
- Stato corrente (fase, direzione)
- Pipeline status per ogni paper (A-G)
- Insight in attesa nel Vault
- Ultima sessione e risultante

### 4. Sync con Cowork

Leggi `COWORK_CONTEXT.md`:
- Messaggi da Cowork non ancora processati
- Ultime direttive architetturali
- Stato del protocollo triangolare

### 5. Verifica ambiente

```bash
# Controlla stato git
git log --oneline -1

# Controlla che i tool funzionino
ls tools/.venv/bin/python 2>/dev/null && echo "Python venv OK" || echo "Python venv MANCANTE"

# Controlla LaTeX
which pdflatex 2>/dev/null && echo "LaTeX OK" || echo "LaTeX MANCANTE"
```

### 6. Report boot

Conferma esplicitamente:

```
Team D-ND attivo.
Stato campo: [da SENTINEL_STATE]
Ultimo commit: [hash + messaggio]
Ambiente: [Python OK/MANCANTE] | [LaTeX OK/MANCANTE]
Messaggi Cowork: [N nuovi / nessuno]
```

## Quando attivare

- **Sempre** all'inizio di una sessione sul repo MM_D-ND
- Dopo un periodo di inattività prolungato
- Quando l'operatore dice "boot", "stato", "dove siamo"

## Quando NON attivare

- Se il boot è già stato fatto nella sessione corrente
- Se l'operatore vuole andare dritto al lavoro (in quel caso: boot silenzioso, report minimo)
