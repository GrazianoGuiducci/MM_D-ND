# Quick Audit — Paper E
## Cosmological Extension of the D-ND Framework

**Data**: 28 Febbraio 2026
**File**: `papers/paper_E_draft3.md` (1127 righe)
**Target**: Classical and Quantum Gravity / Foundations of Physics

---

## Top 3 Problemi

### 1. Modified Einstein Equations (S7) — Auto-contraddizione abstract vs §7.2

L'abstract dichiara: "equation (S7) is not a phenomenological ansatz but a structural necessity derived from Axiom P4". Ma §7.2 (Limitations, riga 979) dice l'opposto: "The modified Einstein equations (S7) are phenomenological ansatze rather than precise geometric consequences." E riga 985: "a fully independent derivation from quantum gravity first principles remains an open problem. The specific functional form retains some freedom."

Il paper argomenta contro se stesso. Lo status reale è: l'*esistenza* di un accoppiamento informazionale è motivata dagli assiomi, ma la *forma funzionale specifica* di $T_{\mu\nu}^{\text{info}}$ è un ansatz con gradi di libertà. La posizione onesta è già in §7.2 — va propagata all'abstract e a §2.2.

**Suggerimento**: Riscrivere abstract e §2.2: "the *existence* of an informational coupling is an axiomatic consequence of P4, while the *specific functional form* is a motivated ansatz constrained by — but not uniquely determined by — the axioms." Rimuovere "not a phenomenological ansatz" dall'abstract.

### 2. $\Omega_{NT} = 2\pi i$ (§5.1) — Usato come risultato stabilito, ma Paper A lo ha declassato a congettura

$\Omega_{NT} = 2\pi i$ è boxed come (S8) a riga 339, presentato come "established." Ma QUICK_AUDIT_A problema #3 (ora fixato nel draft 3.1 di Paper A) lo classifica come "motivated conjecture with strong WKB support, not a theorem." Paper E costruisce un'intera cosmologia ciclica (§5.1-5.3) sopra, con predizioni (CMB, preservazione informazione tra eoni).

Il passaggio dalla fase reale $2\pi$ alla condizione complessa $2\pi i$ (§5.1 riga 346-350) è asserito in una frase senza derivazione.

**Suggerimento**: Etichettare $\Omega_{NT} = 2\pi i$ come "motivated conjecture inherited from Paper A" in tutto il paper. Nell'abstract: "we establish" → "we impose" o "building on the conjectured cyclic coherence condition." Le sezioni §5.1-5.3 devono dichiarare esplicitamente che sono condizionate alla congettura.

### 3. §6.4.5 Test 2 — "Corpus extraction" + claim Riemann zeros nel power spectrum galattico

Test 2 (riga 674): "From the corpus extraction, the Riemann zeta function constraint on the stress-energy tensor eigenvalue spectrum implies anomalous clustering at scales corresponding to Riemann zeros." Due problemi:

(a) "Corpus extraction/mining" è linguaggio interno (come flaggato in QUICK_AUDIT_B problema #1). Non ha significato per un referee.

(b) Il claim che il power spectrum galattico mostri picchi ai numeri d'onda corrispondenti agli zeri di Riemann è straordinario e presentato senza derivazione. Se vero, sarebbe una delle connessioni più notevoli della fisica matematica. Presentarlo come "test" Tier 1 senza derivazione non è credibile.

**Suggerimento**: (a) Sostituire "corpus extraction" con derivazione esplicita o "exploratory analysis (to be published separately)". (b) Declassare Test 2 da "decisive falsification test" a "speculative prediction requiring formal derivation."

---

## Nota cross-reference

Paper E eredita Ω_NT da Paper A (ora congettura motivata) e usa "corpus mining" (flaggato in Paper B). Entrambi i problemi upstream propagano qui e vanno gestiti onestamente.

## Verdetto

Paper E è il **più ambizioso** dopo Paper B (1127 righe: Einstein modificato, inflazione, dark energy, antigravità, tempo emergente, cosmologia ciclica, predizioni DESI). La struttura è ben organizzata e §7.2 è onesto sui limiti. Il problema strutturale è che il **fronte del paper (abstract, §2.2) fa claim che il retro (§7.2) esplicitamente ridimensiona**. I fix sono di calibrazione — il paper non ha bisogno di riscrittura, ha bisogno che l'onestà di §7.2 si propaghi all'abstract.

---

*Quick audit generato dal Team D-ND — 28/02/2026*
