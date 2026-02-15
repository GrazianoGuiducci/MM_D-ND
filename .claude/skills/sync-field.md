---
name: sync-field
description: "Protocollo di Sincronizzazione del Campo. Attivare AUTOMATICAMENTE alla fine di ogni ciclo di lavoro, quando aggiorni il COWORK_CONTEXT.md, quando l'utente chiede 'aggiorna stato', 'sincronizza', 'cosa abbiamo fatto', 'report', o quando passi il testimone a Cowork."
---

# SKILL: SYNC-FIELD — Protocollo di Sincronizzazione

## Identità

Sync-Field è il protocollo che formalizza il flusso informativo nel triangolo Graziano ↔ Claude Code ↔ Cowork. Il COWORK_CONTEXT.md è il veicolo. Questa skill è il formato.

## Quando Attivare

- Fine di ogni ciclo di lavoro significativo
- Passaggio di stato a Cowork ("ho finito, come procediamo?")
- Richiesta esplicita di report o sincronizzazione
- Prima di un commit che modifica il kernel o le skill

## Formato del Report

Quando aggiorni il COWORK_CONTEXT.md, usa questa struttura:

```markdown
## REPORT [Claude Code → Cowork] — [data]

### Risultante
Cosa è stato fatto. File creati/modificati/rimossi. In forma essenziale.

### Stato
Dove siamo adesso. Cosa funziona. Cosa manca.

### Insight (Attenzione Periferica)
Cosa ho notato durante l'esecuzione che potrebbe avere valore:
- Pattern emersi
- Possibili refactoring
- Connessioni con altri componenti
- Domande architetturali per Cowork

### Vault
Insight che hanno valore ma non sono azionabili ora.
Formato: [insight] — [perché non ora] — [quando potrebbe servire]

### Direzione
Proposta per il prossimo passo. Oppure: "Attendo direzione da Cowork/Graziano."
```

## Protocollo

1. **Completa il task** (con dnd-method)
2. **Consolida** (con sentinel-code)
3. **Sincronizza** (con sync-field — questo protocollo):
   - Aggiorna la sezione REPORT nel COWORK_CONTEXT.md
   - Aggiorna SENTINEL_STATE.md se il campo è cambiato
4. **Committa** (con seed-deploy se serve)
5. **Segnala** all'utente: "Report scritto nel COWORK_CONTEXT. Pronto per Cowork."

## Vincolo

Il report è denso. La Lagrangiana: massima informazione, minime parole. Ogni riga del report ha una ragione. Se non aggiunge informazione → taglia.

*Il COWORK_CONTEXT è il campo di sincronizzazione. Sync-Field è il suo protocollo.*
