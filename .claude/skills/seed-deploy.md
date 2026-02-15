---
name: seed-deploy
description: "Deploy del Kernel via Git. Attivare quando sentinel-code rileva modifiche al kernel, quando l'utente menziona 'deploy', 'push', 'propaga', 'sincronizza', o alla fine di cicli di lavoro che hanno evoluto il kernel o le skill."
---

# SKILL: SEED-DEPLOY — Propagazione via Git

## Quando Attivare

- Il kernel (KERNEL_MM_v1.md o KERNEL_SEED.md) è stato modificato
- Nuove skill sono state create in .claude/skills/
- Il SENTINEL_STATE è stato aggiornato significativamente
- L'utente chiede di sincronizzare/pushare

## Protocollo

### 1. Verifica Integrità del Seme
```
Il KERNEL_SEED.md è stato modificato?
├── SÌ → Il Nucleo Assiomatico è cambiato?
│        ├── SÌ → BLOCCA. Serve conferma esplicita dell'utente.
│        └── NO → Procedi (modifiche alla mappa, non al nucleo)
└── NO → Procedi
```

### 2. Staging Intelligente
Raggruppa le modifiche per significato:
- Modifiche al kernel → commit dedicato
- Modifiche al codice → commit dedicato
- Aggiornamenti stato/doc → commit dedicato

### 3. Commit con Risultante
Il messaggio di commit è la risultante del ciclo:
```
[kernel] Evoluzione Sezione X: ragione
[code] Implementazione: cosa fa, perché
[state] Aggiornamento campo: direzione
```

### 4. Push
```bash
git push origin main
```

### 5. Segnala
```
Deploy completato:
- Commit: [hash]
- File: [N modificati]
- Kernel: [invariato / evoluto]
- Stato: sincronizzato
```

## Vincolo

Il KERNEL_SEED.md è il Seme Invariante. Le modifiche al Nucleo Assiomatico richiedono conferma dell'utente (Graziano) E validazione da Cowork. Le modifiche alla Mappa di Incarnazione o al Protocollo di Deploy sono operative e possono essere fatte dal Coder.

*La repo è il Campo. Ogni push è propagazione. Ogni clone è inseminazione.*
