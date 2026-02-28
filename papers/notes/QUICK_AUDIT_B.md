# Quick Audit — Paper B
## Phase Transitions and Lagrangian Dynamics in the D-ND Continuum

**Data**: 28 Febbraio 2026
**File**: `papers/paper_B_draft3.md` (1380 righe)
**Target**: Physical Review E / Journal of Statistical Mechanics

---

## Top 3 Problemi

### 1. Master Equation B1 (§5.3-5.5) — Semi-derivata da fonti esterne

L'equazione master (boxed, §5.5) è introdotta come derivata da "corpus mining results and unified formula synthesis", ma:
- "Corpus mining" non è definito nel paper — è un riferimento a `UNIFIED_FORMULA_SYNTHESIS.md`, un documento interno non peer-reviewed
- Il passaggio dalla Lagrangiana (§2-3) all'equazione master via "Euler-Forward Integration" (§5.3.1) salta alla forma esponenziale $e^{\pm\lambda Z(t)}$ con giustificazione "emergence of exponential coupling" senza derivazione esplicita
- I termini integrali $\int [\vec{D}_{\text{primary}} \cdot \vec{P}_{\text{possibilistic}} - \nabla \cdot \vec{L}_{\text{latency}}] dt'$ non sono mappati a termini specifici della Lagrangiana

**Suggerimento**: O derivare l'equazione master esplicitamente dalla Lagrangiana di §2, o dichiarare che è un ansatz motivato dalla struttura Lagrangiana (non derivato da essa).

### 2. Esponenti critici (§4.2.2) — Universalità sovrastimata

I valori $\beta=1/2, \gamma=1, \delta=3, \nu=1/2$ sono derivati e presentati come se fossero universali per il framework D-ND. Solo dopo (§4.2.2) il paper aggiunge caveats: "exact only under specific conditions" — in particolare, richiedono interazioni a raggio infinito (mean-field). Ma §1.2 e Paper D estendono a osservatori multipli con interazioni locali, dove gli esponenti ricevono correzioni logaritmiche. Il paper riconosce il problema ma la struttura (prima derivazione "universale", poi caveats) è fuorviante.

**Suggerimento**: Invertire la struttura — presentare le condizioni di validità *prima* della derivazione, poi derivare come caso specifico. Oppure rinominare: "Mean-field critical exponents" anziché "Critical exponents of the D-ND transition."

### 3. Stability Criterion (§5.5) — Formalmente mal definito

Il criterio di stabilità usa:
- $\rho_{NT}(t)$: "coherence density in NT continuum" — mai definita matematicamente (probabilità? stato quantistico? unità dimensionali?)
- $dV$: integrazione su "NT" che non è una varietà — è un'etichetta notazionale
- $\|\nabla P(t)\| / \rho_{NT}(t)$: il gradiente presuppone che $P(t)$ sia una funzione su uno spazio — ma su quale spazio?

Il risultato è un **criterio qualitativo presentato come quantitativo**. La validazione numerica (§6) testa l'integrazione Runge-Kutta di $\ddot{Z} + c\dot{Z} + \partial V/\partial Z = 0$, non il criterio di stabilità.

**Suggerimento**: O definire rigorosamente $\rho_{NT}$, $NT$ come spazio, e le unità — o spostare il criterio in una sezione "Qualitative stability analysis" senza pretese di quantitatività.

---

## Nota cross-reference

Paper B dipende criticamente da Paper A §5.5 ($\Omega_{NT} = 2\pi i$) per il criterio di stabilità e l'analisi delle biforcazioni. Se la derivazione in Paper A è incompleta (vedi QUICK_AUDIT_A problema #3), la catena argomentativa in Paper B si indebolisce. I due paper vanno solidificati insieme.

## Verdetto

Paper B è il **più ambizioso** del corpus (1380 righe, Lagrangiana + transizioni di fase + dinamica non lineare). La struttura è impressionante ma ci sono tre punti dove la presentazione è più forte del contenuto effettivo. I fix sono di calibrazione, non strutturali — il paper non ha bisogno di riscrittura, ha bisogno di onestà nella presentazione dei limiti.

---

*Quick audit generato dal Team D-ND — 28/02/2026*
