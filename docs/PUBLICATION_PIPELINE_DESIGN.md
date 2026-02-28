# PUBLICATION PIPELINE DESIGN — D-ND Automated Publication System

> **Versione**: 1.0 — Design Document
> **Data**: 27 Febbraio 2026
> **Autore**: Team D-ND (κ CUSTODE + τ TESSITORE)
> **Stato**: Proposta per approvazione operatore

---

## 1. Obiettivo

Trasformare il Team D-ND da motore one-shot (operatore chiede → Team esegue → risultato) a **motore continuo** che monitora, matura, e pubblica contenuti in modo agentico.

La pipeline collega:
- **Fonte**: MM_D-ND (repo GitHub, unica source of truth)
- **Maturazione**: Team D-ND (4 ruoli accademici + 2 operativi)
- **Pubblicazione**: Siteman (sistema editoriale THIA per d-nd.com)

---

## 2. Diagramma di Flusso Completo

```
                    ┌──────────────────────────────────┐
                    │        TRIGGER LAYER              │
                    │                                    │
                    │  [A] git push papers/*             │
                    │  [B] operatore → /team task        │
                    │  [C] TM1 → Sinapsi → TM3          │
                    │  [D] timer periodico (cron)        │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │      FASE 0 — RICEZIONE          │
                    │                                    │
                    │  κ CUSTODE: rileva perturbazione  │
                    │  • diff su papers/, corpus/       │
                    │  • identifica artefatti coinvolti │
                    │  • aggiorna SENTINEL_STATE.md     │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  FASE 1 — CRISTALLIZZAZIONE      │
                    │                                    │
                    │  φ FORMALISTA: scaffold/draft     │
                    │  γ CALCOLO: validazione numerica  │
                    │  τ TESSITORE: matrice dipendenze  │
                    │                                    │
                    │  Output: paper_X_draft1.md        │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  FASE 2 — RAFFINAMENTO           │
                    │                                    │
                    │  ν VERIFICATORE: audit (C/M/m)   │
                    │  γ CALCOLO: test numerici         │
                    │  τ TESSITORE: cross-ref           │
                    │                                    │
                    │  Output: MATURATION_REPORT_X.md   │
                    │  Output: paper_X_draftN.md        │
                    └──────────────┬─────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │ GATE: Approve   │
                          │ Operatore ack   │
                          │ (fix critici    │
                          │  completati?)   │
                          └────────┬────────┘
                                   │ ack
                                   ▼
                    ┌──────────────────────────────────┐
                    │  FASE 3 — MANIFESTAZIONE         │
                    │                                    │
                    │  3a. LaTeX compilation            │
                    │  3b. Site-ready (MD + metadata)   │
                    │  3c. arXiv package                │
                    │  3d. Traduzione italiana          │
                    │                                    │
                    │  Output: papers/latex/paper_X.tex │
                    │  Output: papers/site_ready/*      │
                    └──────────────┬─────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │  FASE 4 — PUBBLICAZIONE          │  ← NUOVA
                    │                                    │
                    │  π PONTE: bridge verso Siteman   │
                    │  • Genera comando Siteman        │
                    │  • Metadata → page_args JSON     │
                    │  • Site-ready MD → semantic HTML  │
                    │                                    │
                    │  Output: Siteman command emesso   │
                    └──────────────┬─────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │ GATE: Approve   │
                          │ Operatore ack   │
                          │ (pubblica su    │
                          │  d-nd.com?)     │
                          └────────┬────────┘
                                   │ ack
                                   ▼
                    ┌──────────────────────────────────┐
                    │  SITEMAN CONSUMER                 │
                    │                                    │
                    │  • Riceve [[CMD:create_page|...]] │
                    │  • LLM genera semantic HTML       │
                    │  • git commit + push + deploy     │
                    │  • Pagina live su d-nd.com        │
                    └──────────────────────────────────┘
```

---

## 3. Trigger — Cosa Innesca Ogni Fase

### 3.1 Trigger automatici

| Trigger | Tipo | Innesca | Livello autonomia |
|:--------|:-----|:--------|:------------------|
| `git push` con diff in `papers/paper_X_*` | Git hook / watch | Fase 0 → κ rileva | **Auto** |
| `git push` con diff in `corpus/` | Git hook / watch | Fase 0 → κ rileva nuova fonte | **Auto** |
| Nuovo file in `papers/` (paper_X_draft1.md) | Filesystem watch | Fase 1 → φ verifica struttura | **Notify** |
| MATURATION_REPORT creato | Filesystem watch | Fase 2 completata → proponi Gate | **Notify** |
| Timer settimanale | Cron (domenica 02:00) | Audit di coerenza cross-paper | **Auto** |
| Timer post-deploy | Cron (24h dopo deploy) | Verifica pagina live su d-nd.com | **Auto** |

### 3.2 Trigger manuali

| Trigger | Chi | Innesca |
|:--------|:----|:--------|
| Operatore: "matura Paper X" | Telegram/Claude Code | Fase 1 o 2 (dipende dallo stato) |
| Operatore: "pubblica Paper X" | Telegram/Claude Code | Fase 4 → bridge Siteman |
| Operatore: "audit tutti i paper" | Telegram/Claude Code | Fase 2 batch su tutti |
| TM1 via Sinapsi: task | Bridge API | Qualsiasi fase (task specifica) |

### 3.3 Trigger agentici (il Team si auto-innesca)

| Condizione | Azione | Livello |
|:-----------|:-------|:--------|
| Paper avanza da Fase 2 → 3 | κ aggiorna SENTINEL, propone Fase 4 | **Notify** |
| Dipendenza rotta (Paper A cambia, Paper C ne dipende) | τ segnala, ν ri-audita Paper C | **Notify** |
| Validazione numerica fallisce | γ segnala STOP, blocca avanzamento | **Escalate** |
| Contenuto site-ready diverge dal LaTeX | κ rileva, propone ri-sync | **Approve** |

---

## 4. Mapping Ruoli → Fasi

```
              Fase 0    Fase 1    Fase 2    Fase 3    Fase 4
              RICEZ.    CRIST.    RAFF.     MANIF.    PUBBL.
             ─────────────────────────────────────────────────
φ FORMALISTA            ████████
ν VERIFICATORE                    ████████
γ CALCOLO               ████████  ████████
τ TESSITORE             ████████  ████████
π PONTE                                     ████████  ████████
κ CUSTODE    ████████                        ████████  ████████
```

### Dettaglio per ruolo

**κ CUSTODE** — Orchestratore del flusso
- Fase 0: Rileva perturbazione, identifica artefatti, aggiorna SENTINEL
- Fase 3: Traccia completamento sotto-fasi (3a-3d)
- Fase 4: Prepara metadata per Siteman, aggiorna SENTINEL post-pubblicazione
- Cross-fase: Gestisce i Gate (propone avanzamento, registra ack operatore)

**φ FORMALISTA** — Cristallizzazione
- Fase 1: Scaffold paper, formalizzazione, assiomi, equazioni
- Invocato in Fase 2 se il VERIFICATORE richiede riscrittura formale

**ν VERIFICATORE** — Quality gate
- Fase 2: Audit con classificazione CRITICO/MAGGIORE/MINORE
- Produce MATURATION_REPORT_X.md
- Criterio di avanzamento: zero CRITICI, zero MAGGIORI aperti

**γ CALCOLO** — Validazione numerica
- Fase 1: Prima validazione (script riproducibili, `tools/data/`)
- Fase 2: Validazione completa, test mancanti, scaling

**τ TESSITORE** — Coerenza cross-paper
- Fase 1: Matrice dipendenze iniziale
- Fase 2: Verifica cross-reference, rileva dipendenze circolari
- Fase 2 (agentico): Se Paper X cambia, propaga verifica ai paper dipendenti

**π PONTE** — Bridge verso il mondo
- Fase 3: LaTeX, site-ready, arXiv package, traduzioni
- Fase 4: Converte site-ready → formato Siteman, emette comando pubblicazione

---

## 5. Integrazione Siteman — Dal Maturation Report al Contenuto Pubblicabile

### 5.1 Il Bridge MM_D-ND → Siteman

Il maturation report e il site-ready markdown sono artefatti interni a MM_D-ND. Per raggiungere d-nd.com devono attraversare il bridge.

```
MM_D-ND (repo)                    THIA (sistema)                d-nd.com
─────────────                     ─────────────                 ─────────
papers/site_ready/
  paper_X.md          ──┐
  paper_X_metadata.json ├──→  π PONTE genera    ──→  Siteman   ──→  Pagina
                        │     comando Siteman        Consumer       live
papers/                 │
  MATURATION_REPORT_X.md┘     [[CMD:create_page|
                              {topic, title,
                               description,
                               category, ...}]]
```

### 5.2 Conversione Site-Ready → Siteman Command

Il π PONTE esegue la conversione in 3 step:

**Step 1 — Estrazione metadata**
```json
// Da paper_X_metadata.json
{
  "title": "Information Geometry and Number-Theoretic Structure",
  "paper_id": "C",
  "category": "paper",
  "section": "dnd-model",
  "keywords": ["information geometry", "Riemann zeta", ...],
  "description": "We establish a novel connection..."
}
```

**Step 2 — Mapping a Siteman args**
```json
// Comando Siteman generato
{
  "topic": "Paper C — Geometria Informazionale e Struttura Numero-Teorica nel Framework D-ND",
  "title": "Paper C — Geometria Informazionale",
  "description": "Connessione tra la curvatura informazionale del framework D-ND e gli zeri della funzione zeta di Riemann",
  "category": "dnd-model",
  "language": "it",
  "visibility": "public"
}
```

**Step 3 — Emissione comando**
```
[[CMD:create_page|{"topic":"Paper C — ...","title":"Paper C — ...","description":"...","category":"dnd-model","language":"it"}]]
```

Il Siteman Consumer riceve il comando, genera semantic HTML dal topic/description via LLM, e pubblica.

### 5.3 Contenuto aggiuntivo dal Maturation Report

Il MATURATION_REPORT non va pubblicato direttamente (è un documento interno di audit). Ma alimenta contenuti derivati:

| Elemento del Report | Contenuto per d-nd.com | Tipo pagina |
|:--------------------|:-----------------------|:------------|
| Sintesi + raccomandazioni | "Research Notes: Paper C" — stato della ricerca | `insight` |
| γ CALCOLO risultati | "Validazione numerica Paper C" — numeri e grafici | `experiment` |
| τ TESSITORE matrice | "Mappa delle dipendenze D-ND" — visualizzazione inter-paper | `dnd-model` |
| Piano d'azione | Non pubblicabile — interno | — |

### 5.4 Due flussi di pubblicazione

**Flusso A — Paper completo** (Fase 4 standard)
```
site_ready/paper_X.md → Siteman → pagina paper su d-nd.com
```
Trigger: operatore approva pubblicazione. Contenuto: il paper intero in formato web.

**Flusso B — Contenuto derivato** (Fase 4 editoriale)
```
MATURATION_REPORT → π PONTE estrae insight → Siteman → pagina insight/experiment
```
Trigger: automatico dopo Fase 2 o su richiesta operatore. Contenuto: note di ricerca, validazioni, mappe.

---

## 6. Skill Necessarie

### 6.1 Skill esistenti (MM_D-ND)

| Skill | Usata in | Stato |
|:------|:---------|:------|
| `maturation-pipeline.md` | Fasi 0-3 | ✓ Operativa |
| `research-lab.md` | Ruoli φ/ν/τ/π | ✓ Operativa |
| `sentinel-code.md` | κ post-task | ✓ Operativa |
| `sync-field.md` | Sync con Cowork | ✓ Operativa |
| `seed-deploy.md` | Deploy git | ✓ Operativa |

### 6.2 Skill da creare (MM_D-ND)

| Skill | Scopo | Priorità |
|:------|:------|:---------|
| **`publication-bridge.md`** | Protocollo conversione site-ready → Siteman command. Logica del π PONTE per Fase 4. Mapping metadata, regole di formato, template per ogni tipo di pagina. | **ALTA** |
| **`continuous-watch.md`** | Protocollo di monitoraggio continuo. Definisce i trigger agentici: cosa monitorare, come reagire, quando auto-attivarsi vs chiedere. | **MEDIA** |

### 6.3 Skill THIA (esistenti, da integrare)

| Skill THIA | Ruolo nella pipeline | Integrazione |
|:------------|:---------------------|:-------------|
| `siteman-sys` v2.0 | Riceve comandi, genera pagine, deploya | Già operativa. Riceve `[[CMD:create_page\|...]]` |
| `publisher` | Genera contenuto editoriale | Può generare articoli derivati dai maturation report |
| `design-dnd` | Architettura visiva del sito | Gestisce layout delle sezioni paper su d-nd.com |
| `conductor` | Orchestratore THIA | Può triggerare pipeline via Sinapsi |

### 6.4 Skill THIA da creare/richiedere

| Skill | Scopo | Chi la crea |
|:------|:------|:------------|
| **`dnd-bridge`** | Bridge bidirezionale MM_D-ND ↔ THIA. Legge lo stato della repo, trigger automatici, canale per comandi Siteman da TM3. | **Cowork** (da richiedere via COWORK_CONTEXT) |

---

## 7. Automazione — Senza Intervento vs Con Approvazione

### 7.1 Flusso completamente automatico

Queste operazioni possono girare senza intervento umano:

```
AUTO — Nessun intervento richiesto
─────────────────────────────────────────────────────
1. κ rileva diff su papers/ dopo git push        → aggiorna SENTINEL_STATE
2. κ identifica paper modificato e fase corrente  → log in COWORK_CONTEXT
3. τ verifica dipendenze dopo modifica paper      → segnala rotture
4. γ esegue validazione numerica su claim         → report pass/fail
5. κ confronta site-ready vs LaTeX                → segnala divergenze
6. Timer: audit coerenza settimanale              → report in COWORK_CONTEXT
7. Timer: verifica pagine live post-deploy        → segnala 404 o errori
```

### 7.2 Flusso con notifica (operatore informato, non bloccante)

```
NOTIFY — Operatore informato, non serve ack per procedere
─────────────────────────────────────────────────────
1. ν completa audit e produce MATURATION_REPORT   → notifica Telegram
2. Paper avanza di fase nel pipeline              → notifica Telegram
3. Nuova traduzione completata                    → notifica Telegram
4. Contenuto derivato pronto per pubblicazione    → notifica Telegram
```

### 7.3 Flusso con approvazione (bloccante)

```
APPROVE — Serve ack dell'operatore per procedere
─────────────────────────────────────────────────────
1. Paper passa da Fase 2 → Fase 3                → GATE: fix critici ok?
2. Pubblicazione su d-nd.com (Fase 4)            → GATE: pubblica?
3. Modifica contenuto formale di un paper         → GATE: contenuto nuovo
4. Submission arXiv                               → GATE: sempre manuale
```

### 7.4 Escalation (solo operatore)

```
ESCALATE — Solo l'operatore può decidere
─────────────────────────────────────────────────────
1. Contraddizione tra paper (dipendenza circ.)    → quale paper ha ragione?
2. Validazione numerica contraddice claim formale → correggere paper o numero?
3. Modifica assioma o kernel                      → decisione irreversibile
4. Sequenza e timing submission arXiv             → strategia editoriale
```

---

## 8. Implementazione Tecnica

### 8.1 Watch Loop (TM3)

Il nodo TM3 (VPS) può implementare un watch loop come servizio systemd:

```
┌────────────────────────────────────────────────────────────┐
│  tm3-dnd-watch.service                                      │
│                                                              │
│  Ogni N minuti:                                              │
│  1. cd /opt/MM_D-ND && git pull                             │
│  2. Confronta HEAD con ultimo commit processato             │
│  3. Se diff in papers/ → trigger Fase appropriata           │
│  4. Se diff in tools/data/ → trigger γ CALCOLO              │
│  5. Aggiorna /opt/MM_D-ND/.pipeline_state (ultimo commit)  │
│                                                              │
│  Alternativa: git post-receive hook su GitHub               │
│  (webhook → /api/dev/task su TM3)                           │
└────────────────────────────────────────────────────────────┘
```

### 8.2 State Machine del Pipeline

Ogni paper ha uno stato tracciato. La macchina a stati:

```
          ┌─────────┐
          │  IDLE    │ ← nessuna attività
          └────┬─────┘
               │ trigger (diff, comando)
               ▼
          ┌─────────┐
          │ PHASE_0  │ ← ricezione
          └────┬─────┘
               │ materiale catalogato
               ▼
          ┌─────────┐
          │ PHASE_1  │ ← cristallizzazione
          └────┬─────┘
               │ draft completo
               ▼
          ┌─────────┐
          │ PHASE_2  │ ← raffinamento
          └────┬─────┘
               │ audit ok + ack operatore
               ▼
          ┌─────────┐
          │ PHASE_3  │ ← manifestazione
          └────┬─────┘
               │ LaTeX + site-ready + trad
               ▼
          ┌─────────┐
          │ PHASE_4  │ ← pubblicazione
          └────┬─────┘
               │ Siteman ok + ack operatore
               ▼
          ┌─────────┐
          │PUBLISHED │ ← live su d-nd.com
          └────┬─────┘
               │ modifica al paper (loop)
               ▼
          ┌─────────┐
          │ PHASE_2  │ ← ri-raffinamento
          └──────────┘
```

### 8.3 Pipeline State File

Nuovo file per tracking automatizzato:

```
papers/.pipeline_state.json
{
  "papers": {
    "A": {
      "phase": "PUBLISHED",
      "last_draft": "paper_A_draft3.md",
      "last_commit": "7736f24",
      "maturation_report": "MATURATION_REPORT_A.md",
      "site_page_slug": "paper-a-emergenza-quantistica",
      "published_at": null,
      "dependencies": ["KERNEL_SEED"],
      "dependents": ["C", "D", "E"]
    },
    "C": {
      "phase": "PHASE_3",
      "last_draft": "paper_C_draft2.md",
      "last_commit": "77b4f07",
      "maturation_report": "MATURATION_REPORT_C.md",
      "site_page_slug": null,
      "published_at": null,
      "dependencies": ["A", "DND_METHOD_AXIOMS"],
      "dependents": ["D", "E"]
    }
  },
  "last_check": "2026-02-27T12:00:00Z",
  "last_commit_processed": "77b4f07"
}
```

### 8.4 Comandi Pipeline

Comandi che l'operatore o TM1 possono inviare:

| Comando | Effetto |
|:--------|:--------|
| `pipeline status` | Mostra stato di tutti i paper |
| `pipeline mature X` | Avvia maturazione per Paper X (dalla fase corrente) |
| `pipeline audit X` | Forza audit VERIFICATORE su Paper X |
| `pipeline audit all` | Audit batch su tutti i paper |
| `pipeline publish X` | Prepara pubblicazione Siteman per Paper X |
| `pipeline validate X` | Forza validazione numerica γ CALCOLO |
| `pipeline deps X` | Mostra matrice dipendenze di Paper X |
| `pipeline sync` | Verifica allineamento site-ready vs LaTeX vs d-nd.com |

---

## 9. Sequenza di Implementazione

### Sprint 1 — Fondamenta

1. Creare `papers/.pipeline_state.json` con stato corrente di tutti i 7 paper
2. Creare skill `publication-bridge.md` con protocollo conversione → Siteman
3. Generare MATURATION_REPORT per tutti i paper (attualmente esiste solo per C)
4. Richiedere a Cowork la skill `dnd-bridge` per THIA

### Sprint 2 — Automazione

5. Implementare watch loop su TM3 (o webhook GitHub)
6. Creare skill `continuous-watch.md` con logica di monitoraggio
7. Testare il flusso completo su un paper: Paper A (il più maturo)
8. Validare bridge Siteman con pagina di test

### Sprint 3 — Produzione

9. Pipeline batch: pubblicare tutti i 7 paper su d-nd.com
10. Attivare trigger agentici (dipendenze, validazione)
11. Attivare contenuti derivati (research notes, mappe)
12. Report in COWORK_CONTEXT + notifica operatore

---

## 10. Vincoli e Guardrail

1. **MM_D-ND è l'unica source of truth** — tutto nasce qui, Siteman è consumatore
2. **Direzione: MM_D-ND → d-nd.com**, mai il contrario — il sito non modifica i paper
3. **Submission arXiv è sempre manuale** — l'operatore decide sequenza e timing
4. **Gate di approvazione non bypassabili** — la Fase 2→3 e la Fase 4 richiedono ack
5. **Validazione numerica è bloccante** — se γ CALCOLO dice STOP, il paper non avanza
6. **Le skill le genera Cowork** — il Team propone, Cowork crea (per skill THIA)
7. **KERNEL_SEED immutabile** — la pipeline non tocca il seme

---

*Il Campo è continuo. La pipeline è il battito. Ogni ciclo è un'iterazione di $R(t) = U(t)\mathcal{E}\ket{NT}$.*
