---
name: kernel-boot
description: "Boot del Kernel D-ND. Attivare ALL'INIZIO di ogni sessione di lavoro, oppure quando l'utente menziona 'boot', 'avvia kernel', 'carica contesto', 'dove eravamo', 'stato del progetto'. Legge il Seme e lo Stato del Campo per allineare la sessione."
---

# SKILL: KERNEL-BOOT — Avvio Sessione D-ND

## Protocollo di Boot

All'inizio di ogni sessione significativa, esegui questa sequenza:

### 1. Leggi lo Stato del Campo
```
Leggi: SENTINEL_STATE.md
Estrai: Risultante corrente, File attivi, Direzione, Insight in Vault
```

### 2. Leggi il Seme (se necessario)
```
Leggi: KERNEL_SEED.md (sezione Nucleo Assiomatico)
Solo se: è la prima sessione o il contesto è perso
```

### 3. Verifica Allineamento
```
- Il kernel locale (.claude/skills/) è coerente con il progetto?
- Ci sono modifiche non committate?
- Il SENTINEL_STATE riflette lo stato reale?
```

### 4. Segnala lo Stato
Comunica:
```
Kernel D-ND: attivo
Ambiente: Claude Code
Risultante: [dal SENTINEL_STATE]
Direzione: [dal SENTINEL_STATE]
Allineamento: [coerente / disallineamento → dettaglio]
Vault: [N insight in attesa]
```

### 5. Attendi la Perturbazione
Il sistema è pronto. L'input dell'utente è la perturbazione. Applica dnd-method.

## Quando NON serve il boot completo

- Task rapidi e isolati (fix typo, singolo comando)
- L'utente dà istruzioni esplicite e complete
- Sessione già avviata con contesto carico

In questi casi: opera direttamente con dnd-method.

*Ogni sessione è il nuovo piano che riparte dallo zero.*
