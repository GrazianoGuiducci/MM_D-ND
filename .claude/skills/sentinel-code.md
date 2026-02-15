---
name: sentinel-code
description: "Sentinella del Codice. Attivare alla fine di ogni task significativo, prima di commit, quando l'utente chiede 'consolida', 'punto della situazione', 'cosa abbiamo fatto', 'commit', o quando emerge un insight durante il coding."
---

# SKILL: SENTINEL-CODE — Consolidamento e Insight nel Codice

## Alla Fine di Ogni Task

### 1. Mappa la Risultante
- Quali file sono stati creati/modificati/rimossi?
- Il requisito è soddisfatto?
- I test passano?

### 2. Verifica Coerenza
- Il codice è coerente con il codebase esistente?
- Ci sono dipendenze non necessarie?
- La Lagrangiana è rispettata? (minimo codice, massima coerenza)

### 3. Rileva Insight (Attenzione Periferica)
Durante il coding, hai notato:
- Pattern ripetuti → potenziale refactoring?
- Bug latenti in codice adiacente?
- Architettura migliore che emerge?
- Connessioni con altri componenti del progetto D-ND?

Classifica ogni insight:
- **Cristallizza** → agisci ora (refactoring piccolo, fix immediato)
- **Vault** → registra in SENTINEL_STATE (troppo grande per ora, serve direzione)
- **Decadimento** → rilascia (rumore, non assonante)

### 4. Aggiorna SENTINEL_STATE.md
Se il task ha modificato il progetto in modo significativo:
- Aggiorna la Risultante Corrente
- Aggiorna i File Attivi
- Aggiorna la Direzione
- Aggiungi insight al Vault se presenti

### 5. Prepara il Commit
- Messaggio di commit coerente con la risultante (cosa È stato fatto, non cosa è stato toccato)
- Raggruppa le modifiche logicamente
- Se il kernel è evoluto → attiva seed-deploy

### 6. Libera Attenzione
Ciò che è consolidato esce dal campo attivo. Il prossimo task parte dal nuovo zero.

## Regola dell'Eccezione nel Commit

Prima di committare, una pausa:
- La soluzione è buona. C'è la soluzione migliore?
- Se sì e il costo è basso → implementa ora
- Se sì e il costo è alto → Vault, commit la buona soluzione

*Il commit è il piano completato. Zero non banale. Base del prossimo piano.*
