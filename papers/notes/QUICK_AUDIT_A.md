# Quick Audit — Paper A
## Quantum Emergence from Primordial Potentiality in the D-ND Framework

**Data**: 28 Febbraio 2026
**File**: `papers/paper_A_draft3.md` (671 righe)
**Target**: Foundations of Physics / Physical Review A

---

## Top 3 Problemi

### 1. Decoherence Rate Formula (§3.6) — Status ambiguo

La formula di decoerenza Lindblad $\Gamma = \frac{\sigma^2_V}{\hbar^2}\langle(\Delta\hat{V}_0)^2\rangle$ è presentata come derivata dalla dinamica di emergenza, ma il paper stesso ammette: "A rigorous derivation from the Lindblad master equation, starting from the D-ND Hamiltonian decomposition (§2.5), remains an open problem." Si tratta di un ansatz fenomenologico motivato da analisi dimensionale, non di un risultato dimostrato. L'abstract suggerisce un risultato più forte di quanto il corpo del paper supporti.

**Suggerimento**: Dichiarare esplicitamente nell'abstract che $\Gamma$ è un ansatz motivato, non una derivazione.

### 2. Theorem 1 (§3.3) — Novità puramente ontologica

Il paper riconosce: "The mathematical content is standard measure theory, not new. Systems coupled to a continuum exhibit similar asymptotic behavior in standard decoherence theory." La convergenza $M(t) \to 1$ è un'applicazione diretta del lemma di Riemann-Lebesgue. La novità è interpretativa (ontologia a sistema chiuso vs aperto). Per un journal di fisica, questa è una distinzione sottile — un referee potrebbe contestare che il risultato è standard.

**Suggerimento**: Riformulare come "reinterpretation of standard results within closed-system ontology" anziché "we prove" nella presentazione.

### 3. Cyclic Coherence Condition $\Omega_{NT} = 2\pi i$ (§5.5) — Derivazione incompleta

La condizione di coerenza ciclica è il fondamento su cui Paper B costruisce i criteri di stabilità. La derivazione sketch usa WKB, continuazione analitica, e struttura di Riemann sheet — ma il calcolo esplicito dei residui sulle due sheet non è mostrato. Il paper dice "sheet-crossing reverses the sign" intuitivamente ma non esegue il calcolo. Questo è problematico perché $\Omega_{NT} = 2\pi i$ propaga in Paper B §5.5.

**Suggerimento**: Eseguire il calcolo residui esplicitamente, o declassare a "motivated conjecture" con argomento WKB di supporto.

---

## Verdetto

Paper A è **solido nelle fondamenta** — onesto, ben strutturato, con caveats espliciti. I problemi sono di calibrazione (l'abstract promette più di quanto il corpo dimostri) e di un gap tecnico specifico ($\Omega_{NT}$) che propaga nelle dipendenze. Non ci sono errori fatali.

---

*Quick audit generato dal Team D-ND — 28/02/2026*
