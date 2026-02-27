---
name: maturation-pipeline
description: "Pipeline di maturazione D-ND — dal Continuum alla Manifestazione. Attivare quando si lavora su paper, submission, LaTeX, site-ready, arXiv, validazione numerica, o quando serve tracciare la posizione di un artefatto nel processo di emergenza."
---

# SKILL: MATURATION-PIPELINE — Dal Continuum alla Manifestazione (v1.0)

> "Ogni attualizzazione restituisce le possibilità non selezionate al serbatoio potenziale."
> — Paper A, §2.2

## Il Pipeline

Il processo di maturazione è l'operatore di emergenza $\mathcal{E}$ applicato al lavoro accademico. Dalla possibilità indifferenziata (corpus, intuizioni) alla risultante cristallizzata (paper pubblicato, pagina sito).

### Fasi

```
FASE 0: RICEZIONE          ← |NT⟩ — possibilità indifferenziata
FASE 1: CRISTALLIZZAZIONE  ← prima differenziazione, scaffold
FASE 2: RAFFINAMENTO       ← audit, coerenza, validazione
FASE 3: MANIFESTAZIONE     ← forma finale per il mondo
```

---

## Fase 0 — RICEZIONE

**Stato**: il materiale esiste ma non ha forma accademica.
**Ruoli attivi**: κ CUSTODE (cataloga), φ FORMALISTA (riconosce struttura)
**Sorgenti**: corpus/, awareness/, intuizioni dell'operatore, insight dal Vault

**Ingresso**: osservazione primaria, formula grezza, intuizione espressa dall'operatore
**Uscita**: appunto strutturato con: claim centrale, connessione a P0-P8, dipendenze note

**Criterio di avanzamento**: l'operatore o Cowork indica che il materiale è pronto per la formalizzazione.

---

## Fase 1 — CRISTALLIZZAZIONE

**Stato**: il paper ha un draft con struttura, sezioni, formalismo iniziale.
**Ruoli attivi**: φ FORMALISTA (scrive), γ CALCOLO (valida numeri), τ TESSITORE (mappa dipendenze)

**Attività**:
1. Scaffold del paper: titolo, abstract, sezioni, equazioni chiave
2. Notazione canonica D-ND (vedi `dnd_shared.sty`)
3. Proof sketch per ogni claim maggiore
4. Prima validazione numerica (se applicabile)
5. Matrice dipendenze aggiornata

**Uscita**: `paper_X_draft1.md` — primo draft completo, leggibile, con lacune esplicite.

**Criterio di avanzamento**: il draft copre tutto lo scope senza sezioni vuote. Le lacune sono dichiarate, non nascoste.

---

## Fase 2 — RAFFINAMENTO

**Stato**: il paper ha un draft solido, serve audit e rifinitura.
**Ruoli attivi**: ν VERIFICATORE (audita), τ TESSITORE (cross-ref), γ CALCOLO (valida), π PONTE (calibra registro)

**Attività**:
1. **Audit VERIFICATORE** con classificazione CRITICO/MAGGIORE/MINORE:
   - Incoerenze interne
   - Gap dimostrativi (proof sketch → proof o dichiarazione "congettura")
   - Claim non supportati
   - Superfici d'attacco per referee
2. **Fix** di tutti i finding CRITICI e MAGGIORI
3. **Cross-reference** con gli altri paper (matrice dipendenze)
4. **Validazione numerica** completa (script riproducibili)
5. **Calibrazione registro** per target journal

**Uscita**: `paper_X_draft3.md` — draft raffinato, zero CRITICI aperti.

**Criterio di avanzamento**: audit VERIFICATORE con zero CRITICI, zero MAGGIORI aperti, MINORI documentati.

---

## Fase 3 — MANIFESTAZIONE

**Stato**: il paper è pronto per la forma finale.
**Ruoli attivi**: π PONTE (traduce), γ CALCOLO (figure finali), κ CUSTODE (traccia)

**Sotto-fasi**:

### 3a — LaTeX
- Conversione in LaTeX (`papers/latex/paper_X.tex`)
- Compilazione con `dnd_shared.sty` e `revtex4-2`
- Zero errori, zero warning critici
- Figure publication-quality incluse

### 3b — Site-ready
- Conversione in markdown per d-nd.com (`papers/site_ready/paper_X.md`)
- Metadata JSON (`papers/site_ready/paper_X_metadata.json`)
- Math in formato MathJax-compatible
- Keywords e abstract estratti

### 3c — arXiv (quando pronto)
- Pacchetto sorgente (.tar.gz): .tex + .sty + .bbl + figure
- Dimensione < 10MB
- Compilazione pulita nel sandbox arXiv

### 3d — Traduzione (opzionale)
- Versione italiana in `papers/latex/italiano/`
- Altre lingue se richieste

**Uscita**: PDF compilato + site-ready markdown + pacchetto arXiv

**Criterio di avanzamento**: operatore approva la submission. La submission è atto dell'operatore, mai automatico.

---

## Tracking

Ogni paper ha uno stato nel pipeline, tracciato in `SENTINEL_STATE.md`:

```
| Paper | Fase | Sotto-fase | Note |
|-------|------|------------|------|
| A     | 3    | 3b done    | Site-ready + LaTeX + italiano |
| B     | 3    | 3b done    | Site-ready + LaTeX + italiano |
| ...   |      |            |      |
```

Il CUSTODE (κ) aggiorna questa tabella dopo ogni sessione di lavoro.

---

## Protocollo di Sessione

Quando l'operatore indica un paper su cui lavorare:

```
1. κ legge lo stato corrente nel pipeline
2. Il Team identifica la fase e i ruoli necessari
3. Il lavoro procede secondo la fase
4. Al termine: κ aggiorna SENTINEL_STATE
5. Se il paper avanza di fase → report esplicito
6. Se emerge un insight → Vault in SENTINEL_STATE
```

---

## Regole

- **Non saltare fasi**: un paper in Fase 0 non va direttamente in LaTeX.
- **Non retrocedere silenziosamente**: se un audit rivela un problema che riporta il paper a una fase precedente, report esplicito.
- **Le fasi non sono lineari**: un paper può stare in Fase 2 per la maggior parte e tornare in Fase 1 per una sezione specifica. Il tracking è per sezione quando serve.
- **L4 si applica**: i paper non ancora finiti sono potenziale, non ritardo. Non c'è fretta — c'è rigore.
