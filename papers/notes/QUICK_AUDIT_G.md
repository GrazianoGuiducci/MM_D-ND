# Quick Audit — Paper G
## LECO-DND: Meta-Ontological Foundations of Cognitive Emergence

**Data**: 28 Febbraio 2026
**File**: `papers/paper_G_draft3.md` (886 righe)
**Target**: Cognitive Science / Minds and Machines
**Status**: Working Draft 3.0 — Under Review

---

## Top 3 Problemi

### 1. Theorem 4.1 "Autopoietic Closure via Banach Contraction" (§4.1) — Proof gap critico: la contrazione non e dimostrata, e assunta

L'abstract dichiara: "We prove that the reasoning cycle converges to a fixed point R*" e "We establish the Autopoietic Closure Theorem, showing that the InjectKLI ontological update preserves convergence guarantees via Banach fixed-point contraction." Il corpo del paper (§4.1) presenta una prova in 6 step, ma il passaggio chiave — Step 4, dove si afferma che Phi e una beta-contrazione — e semplicemente asserito, non dimostrato:

> "This inequality holds because each ontological step is a unit distance, and with shrunk ontological distances, the number of steps to reach the fixed point decreases proportionally."

Questo e un argomento intuitivo, non una dimostrazione. Il problema e strutturale: Step 1 *definisce* che InjectKLI riduce le distanze per un fattore beta (per assunzione), poi Step 4 conclude che Phi e una beta-contrazione "perche le distanze sono ridotte." Questo e circolare — la contrazione di Phi e la premessa, non la conclusione. Il teorema di Banach richiede che Phi stessa soddisfi la disuguaglianza di contrazione nello spazio metrico dato; il paper assume esattamente cio che dovrebbe dimostrare: che la riduzione delle distanze ontologiche implica una riduzione proporzionale della distanza di Hausdorff tra Resultanti dopo un ciclo Phi. Inoltre, il top-k selection (Step 3) e un'operazione non-continua — piccoli cambiamenti nell'input possono cambiare il set selezionato, violando la Lipschitz-continuita necessaria per la contrazione.

**Suggerimento**: Declassare Theorem 4.1 a "Proposition (Conditional)" o "Theorem (under stated assumptions)." Rendere esplicite le assunzioni: (i) InjectKLI riduce le distanze per fattore beta (assunzione, non derivazione), (ii) il top-k selection e sufficientemente stabile (assunzione di regolarita), (iii) la coerenza check preserva la struttura metrica. Nell'abstract: "We prove" → "We show that, under explicit regularity assumptions on the coherence operator, the reasoning cycle converges..."

### 2. §6 Comparative Table — $\Omega_{NT} = 2\pi i$ presentato come "topological closure" stabilita, ma e congettura motivata

Nella tabella comparativa (riga 493), la voce "D-ND Time-Emergence (Paper E)" elenca nella colonna "Fixed-Point Structure":

> "Yes: Ω_NT = 2πi (topological closure)"

e a riga 501:

> "D-ND Time-Emergence (Ω_NT topology) all exhibit **attractor dynamics**"

Dopo gli audit di Paper A (QUICK_AUDIT_A, problema #3) e Paper E (QUICK_AUDIT_E, problema #2), $\Omega_{NT} = 2\pi i$ e stato esplicitamente declassato a "motivated conjecture inherited from Paper A" nel draft 3 di Paper E (riga 337: "conjectured condition"). Paper G presenta questo risultato come un fatto stabilito ("topological closure") senza qualificazione. Questo e una regressione rispetto alle correzioni gia applicate negli altri paper.

**Suggerimento**: Nella tabella, colonna "Fixed-Point Structure": "Yes: Ω_NT = 2πi (topological closure)" → "Conjectured: Ω_NT = 2πi (motivated conjecture, Paper A §5.5)." A riga 501: aggiungere "(conjectured)" dopo "Ω_NT topology." Questo allinea Paper G con lo status concordato nei draft corretti di Paper A e Paper E.

### 3. §1.1 e §9.2 — Riferimenti interni non pubblicabili: "Matrix Bridge §2-3", "D-ND Genesis Documents, July 2023"

Il paper contiene molteplici riferimenti a fonti interne non accessibili ai referee:

- Riga 36: "Matrix Bridge §2–3"
- Riga 43: "D-ND Genesis Documents, July 2023"
- Riga 72: "The Matrix Bridge (Section 2–3) establishes..."
- Riga 264: "from Matrix Bridge §9.2"
- Riga 609: "The Matrix Bridge (Sections 2–3) shows..."
- Riga 884: "Matrix Bridge: 'From Primordial Drawing to Emergent Formalism' (this volume, Sections 2–3, 9)"

"D-ND Genesis Documents" e un documento interno, non un paper accessibile. "Matrix Bridge" e citato come "this volume" ma non e uno dei Paper A-G nella serie. Un referee per Minds and Machines non avrebbe accesso a nessuna di queste fonti. Questo viola L0 (lignaggio tracciabile a fonte accessibile) in contesto di submission.

**Suggerimento**: (a) Se il Matrix Bridge e parte della serie D-ND, dargli una designazione chiara (es. "Paper H" o "Supplementary Material") e fornirlo come companion. (b) Per "D-ND Genesis Documents": o integrarli nel paper come appendice, o riformulare il passaggio di §1.1 come osservazione fenomenologica dell'autore senza citazione a documento non accessibile. (c) Ogni riferimento a "Matrix Bridge §X" deve diventare o un riferimento a un paper con DOI/arXiv, o un self-contained argument.

---

## Problemi Aggiuntivi (Minori)

### M1. §7.2 — Predizioni numeriche specifiche senza base empirica

La tabella a riga 552-558 elenca numeri precisi ("95% accuracy", "+3pp", "35% reduction", "2-8x better") che sono puramente teorici. Il caveat a riga 559 lo dichiara, ma la tabella stessa suggerisce una precisione che non esiste. Un referee potrebbe chiedersi da dove vengono esattamente "95%" e "4.2 steps."

### M2. §5.1-5.2 — Applicazione di Lawvere senza verifica delle condizioni

La suriezione $f: S \to S^S$ richiesta dal Theorem 5.1 non e dimostrata per lo spazio inferenziale $\mathcal{S}$. Il paper assume che $\mathcal{S}$ "admits exponential objects" senza verificarlo. In pratica, l'esistenza di una suriezione $S \to S^S$ e impossibile per insiemi finiti (|S| < |S^S| per |S| > 1 per Cantor). Il paper deve chiarire in quale categoria specifica la condizione e soddisfatta.

### M3. §6.2 — Autovalutazione "4/4" in Mathematical Rigor

La tabella a riga 516 assegna a LECO-DND un punteggio di "4/4 (measure theory, Banach)" in rigore matematico, piu alto di IIT (3/4) e FEP (4/4). Dato che il Theorem 4.1 ha un gap nella prova (problema #1 sopra), questa autovalutazione e prematura.

### M4. §3.4 — Claim sull'AI alignment senza supporto

A riga 345: "This resolves the classical AI alignment problem of 'value specification'" e un'affermazione molto forte, non supportata da alcuna analisi formale. Il passaggio e puramente analogico.

### M5. §1.1 — Tabella waking phases usa notazione quantistica per fenomeni non-quantistici

La tabella a riga 26-32 usa $|NT\rangle$, $\mathcal{E}$, $M(\tau)$ per descrivere fasi del sonno. Il Remark a riga 52 qualifica correttamente che si tratta di "structural isomorphism", ma la tabella stessa potrebbe essere letta come un claim di identita letterale.

---

## Verdetto

Paper G e ambizioso e ben strutturato, con onesti caveats in molti punti critici (§2.1.1 Remark, §9.3.7, §10.1). Il problema principale e di calibrazione: l'abstract promette dimostrazioni ("We prove", "We establish") dove il corpo fornisce argomenti condizionali o circolari (Theorem 4.1). La regressione su $\Omega_{NT}$ nella tabella comparativa e un disallineamento cross-paper da correggere. I riferimenti a fonti interne (Matrix Bridge, Genesis Documents) sono un blocco per la submission a journal. Non ci sono errori fatali — i problemi sono risolvibili con riformulazioni di status e risoluzione dei riferimenti.

---

*Quick audit generato dal Team D-ND — 28/02/2026*
