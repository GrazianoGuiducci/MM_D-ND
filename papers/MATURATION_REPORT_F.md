# Report di Maturazione — Paper F
## D-ND Quantum Information Engine: Modified Quantum Gates and Computational Framework

**Data**: 28 Febbraio 2026
**Ruoli attivi**: φ FORMALISTA, ν VERIFICATORE, γ CALCOLO, τ TESSITORE
**Source**: `papers/paper_F_draft2.md` (1518 righe)
**Target Journal**: Quantum Computing and Quantum Information Theory / Physical Review A (QI)
**Fase attuale**: 2 (Raffinamento) — non pronto per Fase 3

---

## φ FORMALISTA — Coerenza assiomatica

### Tracciamento ai 9 Assiomi (P0–P8)

| Assioma | Connessione nel Paper F | Forza |
|:--------|:------------------------|:------|
| P0 (Zero) | Lo stato |NT⟩ come superposizione iniziale indifferenziata — punto di partenza del circuito | ★★★ |
| P1 (Dipolo) | SpinNode {-1, +1} ↔ dipolo D-ND in §6.5.1. Esplicita e ben motivata | ★★★★ |
| P2 (Assonanza) | La corrispondenza gate D-ND ↔ primitive THRML (§6.5.4) come assonanza formale tra computing e termodinamica | ★★★ |
| P3 (Risultante) | La convergenza IFS al punto fisso della densità possibilistica — risultante computazionale. Non esplicitata come P3 | ★★ |
| P4 (Potenzialità) | La densità possibilistica ρ_DND (§2.1) con M_proto come misura di proto-attualizzazione. Molto forte | ★★★★★ |
| P5 (Autopoiesi) | Il circuito D-ND come sistema autopoietico che genera il proprio stato — non dichiarato | ★ |
| P6 (Movimento) | L'evoluzione temporale U(t) e il ciclo di emergenza M(t) → 0 → 1 | ★★★ |
| P7 (Limite) | Buono: le limitazioni della universalità sono dichiarate (§3.5 Open Problem). Il regime di validità del simulatore IFS è esplicito (§5.1 Remark) | ★★★★ |
| P8 (Seme) | Il fattore di compressione topologica χ (Betti number) come invariante strutturale | ★★★ |

**Lacune**:
- P3 (Risultante): la convergenza IFS È la risultante ma non è connessa esplicitamente all'assioma. Il punto fisso della simulazione incarna R+1=R ma la connessione è muta.
- P5 (Autopoiesi): il paper parla di "Quantum Information Engine" — un motore che genera la propria struttura computazionale. Ma l'autopoiesi non è mai menzionata. Ironia: Paper F è assegnato a P5 nel curriculum, ma P5 è il più debole.

### Rigore formale

**Positivo**:
- Le definizioni dei gate sono precise (Def. 3.1-3.4) con normalizzazione esplicita.
- La prova di universalità (Prop. 3.5) è ben strutturata: riduzione al caso standard + estensione perturbativa.
- Il claim di universalità è onestamente limitato al regime perturbativo (δV ≪ 1), con l'Open Problem per il caso generale.
- Le Appendici A e B contengono prove complete per Prop. 2.2 e Prop. 4.3.
- I bound di errore nella simulazione lineare sono quantitativi con tabella numerica (§5.4).
- Il BBBV lower bound è correttamente richiamato (§6.1 Remark) per evitare overclaiming.

**Negativo**:
- L'error model (§4.2) è heuristico: la soppressione lineare ε(t) = ε₀(1-M(t)) è assunta dalla struttura di L_k^DND, non derivata da primi principi.
- Le metriche di simulazione (§6.6) sono definite ma mai validate contro simulazioni reali.
- L'intero bridge THRML/Omega-Kernel (§6.5) è un'analogia formale, non una dimostrazione. La corrispondenza è strutturale ma i "=" sono tutti "↔".
- Il paper contiene ~300 righe di pseudocodice Python (§5.3 + §6.5.5 + §6.6.5) — eccessivo per un paper accademico. Codice ≠ prova.

---

## ν VERIFICATORE — Audit di profondità

### CRITICI (3)

**C1 — L'error model L_k^DND(t) = L_k · (1-M(t)) non è giustificato (§4.2)**
La Proposizione 4.3 assume che gli operatori di Lindblad siano modificati linearmente dall'emergenza: L_k^DND(t) = L_k · (1-M(t)). Questa è un'assunzione ad hoc, non una derivazione. Perché la soppressione dovrebbe essere lineare in M(t) e non quadratica, esponenziale, o di altra forma? Il paper dice "the emergence measure M(t) modifies the Lindblad dissipation operators" come se fosse un fatto, ma è un postulato. Per un paper che pretende di dimostrare "emergence-assisted error suppression", il meccanismo di soppressione deve essere derivato, non assunto.

**Suggerimento**: Derivare la forma funzionale di L_k^DND dalla struttura di ℰ e dal Hamiltoniano del sistema, oppure dichiarare esplicitamente che è un modello fenomenologico e testarne la consistenza con una famiglia di ansatz (lineare, quadratica, esponenziale).

**C2 — La prova di universalità (Prop. 3.5) non dimostra universalità per nessun δV > 0**
La prova perturbativa dice: per δV piccolo, i gate D-ND sono perturbazioni lisce dei gate standard, quindi universalità è "preservata per continuità". Ma la continuità della mappa gate→unitario NON implica densità del sottogruppo generato. Il teorema di Solovay-Kitaev richiede che il gate set generi un sottogruppo DENSO di SU(2^n). Perturbare i generatori non garantisce che il sottogruppo generato resti denso — potrebbe collassare su un sottogruppo proprio. Serve un argomento aggiuntivo (ad esempio: mostrare che i gate D-ND generano ancora un libero-prodotto denso in SU(2^n), o usare il criterio di Naimark-Segal per gruppi topologici).

**Suggerimento**: O rafforzare la prova con un argomento algebrico (mostrare che la chiusura del sottogruppo generato contiene i generatori standard come limite), oppure declassare "Proposition 3.5" a "Conjecture 3.5" con evidenza perturbativa e numerica.

**C3 — La complessità O(n³·T) per il simulatore IFS non è dimostrata (§5.3)**
Il paper afferma che nella regione λ < 0.3 la complessità è O(n³·T), ma il pseudocodice include: (1) simulazione quantistica standard P(t) che è O(n²·2^n), (2) tracking proto-branches che cresce esponenzialmente. Il paper dice "most branches pruned" ma non dimostra che il numero di branch significativi sia polinomiale. Il claim di simulazione classica efficiente per circuiti quantistici, anche in regime limitato, richiede una prova rigorosa — altrimenti è un'affermazione potenzialmente in contraddizione con risultati di complessità noti (BQP vs BPP).

**Suggerimento**: Formalizzare le condizioni di pruning. Dimostrare che sotto le ipotesi (M(t) < 0.5, circuito di profondità < 100), il numero di branch significativi scala come poly(n). Oppure rimuovere il claim di complessità polinomiale e presentare il simulatore come euristica con evidenza empirica.

### MAGGIORI (5)

**M1 — Sezione THRML/Omega-Kernel troppo lunga e speculativa (§6.5, ~350 righe)**
L'intera sezione §6.5 con sottosezioni 6.5.1–6.5.6 e §6.6 con 6.6.1–6.6.6 occupa circa 650 righe — quasi metà del paper. Il contenuto è un'analogia strutturale tra D-ND e THRML, con pseudocodice Python che non è né testato né verificabile. Per un paper di quantum information theory, questa sezione dilata il focus e indebolisce il nucleo (gate + universalità + error suppression). Un referee la segnalerà come "non necessaria".

**Suggerimento**: Condensare §6.5 in una singola sottosezione (§6.4) di ~100 righe che stabilisce la corrispondenza strutturale. Spostare il codice e le metriche in materiale supplementare. Le sottosezioni 6.6.1–6.6.6 possono diventare una tabella.

**M2 — La Proposizione 2.3a è vaga e contradditoria (§2.3)**
La proposizione dice M_dist = Shannon entropy e M_ent = negativity, poi in Prop. 2.3a usa definizioni diverse ("Shannon entropy of actualized mode distribution" e "nonlocal coherence preserved across actualized subsystems") senza formule. Peggio: il constraint M_dist + M_ent = M(t) implica che Shannon entropy + negativity = emergence measure, il che non ha giustificazione generale. Sono quantità definite in modi diversi — la loro somma non ha ragione di uguagliare M(t).

**Suggerimento**: Scegliere: o le definizioni operative (Shannon entropy, negativity) o le definizioni emergentiste (Prop. 2.3a). Non entrambe. Se si sceglie la seconda, fornire definizioni formali. Se la prima, dimostrare che la somma corrisponde a M(t).

**M3 — La formulazione dei gate mescola formalismo quantistico con grafi (§3.1-3.3)**
I gate D-ND sono definiti in termini di grafi di emergenza (vertici, pesi, vicinato) ma agiscono su stati quantistici (ket, unitari, traccia). Il passaggio dall'uno all'altro non è mai formalizzato. Come si costruisce il grafo di emergenza da uno stato quantistico dato? La sezione §3.2 "Emergence Graph Construction" fornisce una risposta (eigenvalues di ℰ come vertici, transizioni come edge), ma è un remark, non una definizione formale integrata nella costruzione dei gate.

**Suggerimento**: Aggiungere una Definition 3.0 (Emergence Graph Construction) prima dei gate, che formalizzi la corrispondenza stato quantistico → grafo. I gate poi diventano operatori sul grafo che inducono unitari sullo spazio di Hilbert.

**M4 — Le "Corpus Simulations" citate in §5.4 non sono identificabili**
La tabella di errore numerico (§5.4) cita "(from Corpus Simulations)" ma non c'è riferimento a quale corpus, quale codice, o quali parametri. Un referee non può verificare i numeri. Se sono simulazioni dell'autore, devono essere riproducibili e citate. Se sono del corpus D-ND, il riferimento deve essere esplicito.

**Suggerimento**: Citare esplicitamente la sorgente dei dati numerici. Idealmente, fornire un link al codice o almeno i parametri della simulazione.

**M5 — La Conjecture 6.1 (Grover speedup) è formulata in modo confuso**
La congettura dice "D-ND quantum search may achieve a constant-factor improvement over standard Grover" con fattore α ≥ 1. Ma poi §6.4 riformula come "open problem" con un'analisi diversa (fattore √(λn)). Le due formulazioni sono inconsistenti: la prima è un fattore costante, la seconda dipende da n. Il lettore non sa quale claim sta facendo il paper.

**Suggerimento**: Unificare §6.1 e §6.4 in una sola sezione. Scegliere una formulazione (fattore costante O sclamento con n) e giustificarla. Rimuovere la formulazione debole.

### MINORI (4)

**m1** — Le reference ai "Formula C1", "Formula C2", etc. (§3.1-3.3) sono un residuo interno. Per un paper stand-alone, le formule dovrebbero avere numerazione standard (Eq. 1, Eq. 2, etc.) o referenze locali.

**m2** — La sezione §1.1 "Notation Clarification" è troppo lunga per un'introduzione. Questa chiarificazione andrebbe in una nota a piè pagina o in un'appendice notazionale.

**m3** — Il Shortcut_DND (§3.4) non è un gate ma una strategia di compilazione — il paper lo dice ma lo chiama comunque "gate". Questo crea confusione quando si parla di "four fundamental gates" nel testo.

**m4** — La sezione §6.3 definisce una classe di complessità BQP_DND ma non la caratterizza formalmente. La definizione "problems solvable by D-ND circuits with polynomial emergence overhead" è informale. O formalizzare (definire la macchina di Turing astratta) o rimuovere la definizione e usare linguaggio informale.

---

## γ CALCOLO — Validazione numerica

### Dati esistenti

| Test | Claim del paper | Dato disponibile | Verdetto |
|:-----|:----------------|:-----------------|:---------|
| Error table (§5.4) | λ=0.1→0.3%, λ=0.5→5.8%, etc. | "Corpus Simulations" — **sorgente non identificata** | ⚠ NON VERIFICABILE |
| Universalità numerica (§3.5) | "Numerical evidence on small systems (n ≤ 5)" | Non fornita nel paper | ⚠ CLAIM NON SUPPORTATO |
| Overhead reduction (§6.2) | "30-50% reduction" | Non supportato da calcoli | ⚠ CLAIM NON SUPPORTATO |
| Grover speedup (§6.4) | fattore √(λn) | Analisi preliminare, non validata | ⚠ INCOMPLETO |

### Validazioni necessarie

1. **Test 1 — Tabella errori (§5.4)**: Riprodurre la tabella con simulazione esplicita. Parametri: sistema a n=4 qubit, circuito di 10 gate, λ variabile da 0.1 a 0.9. Confrontare errore R_exact vs R_linear.
2. **Test 2 — Universalità numerica (§3.5)**: Per n=2,3,4, generare 1000 unitari random in SU(2^n), decomporli in gate D-ND con δV ∈ {0.01, 0.1, 0.3}, misurare l'errore di approssimazione.
3. **Test 3 — Compressione topologica (§3.4)**: Per un ensemble di grafi di emergenza (random, small-world, lattice), calcolare χ e misurare la riduzione effettiva del gate count.
4. **Test 4 — IFS convergenza (§5.1)**: Verificare che il simulatore IFS converge al risultato esatto per λ < 0.3 su circuiti di profondità 10-100.

### Stato CALCOLO
**Nessun dato numerico è verificabile.** Tutti i claim quantitativi rimandano a "corpus simulations" o "numerical evidence" non fornita. Questo è il problema più grave del paper: fa affermazioni quantitative senza dati accessibili.

---

## τ TESSITORE — Matrice dipendenze

### Paper F dipende da:

| Da | Cosa usa | Verificato |
|:---|:---------|:-----------|
| Paper A | M(t) emergence measure, |NT⟩, U(t), ℰ, M_proto = 1-M(t) | ✓ Coerente (§2.3, Prop. 2.3) |
| Paper A | Assiomi A₁-A₆ (impliciti) | ✓ Referenziati in §1 |
| Paper A | Eigenvalori λ_k di ℰ (per costruzione grafo, §3.2 Remark) | ✓ Usati correttamente |
| Paper C | K_gen curvatura informazionale (citata in §5.2 "Formula C7") | ⚠ Referenza interna "Formula C1-C7" — va verificata |
| Paper B | λ_DND coupling constant (citata in §1.1) | ✓ Solo notazione, nessuna dipendenza contenutistica |
| Paper D | λ_auto convergence rate (citata in §1.1) | ✓ Solo notazione |
| Paper E | λ_cosmo emergence coupling (citata in §1.1) | ✓ Solo notazione |

### Paper F introduce (usato da altri):

| Concetto | Introdotto in | Usato da |
|:---------|:-------------|:---------|
| ρ_DND (densità possibilistica) | §2.1, Def. 2.1 | Paper G (applicazioni cognitive) |
| Gate D-ND {H_DND, CNOT_DND, Phase_DND, Shortcut_DND} | §3.1-3.4 | Paper G (modello computazionale) |
| Error suppression formula ε(t) = ε₀(1-M(t)) | §4.2, Prop. 4.3 | Nessuno (nuovo) |
| Linear simulation framework R_linear | §5.2 | Nessuno (nuovo) |
| Bridge THRML/Omega-Kernel | §6.5 | Nessuno (nuovo, speculative) |

### Cross-reference da verificare
- Le "Formula C1-C7" nel testo sono etichette interne non connesse a Paper C. Sono formule F1-F7 del Paper F stesso. La nomenclatura è confusa: "C" suggerisce Paper C ma si riferisce al modello "Circuit". Da rinominare.
- §2.3 usa M(t) da Paper A — la definizione è coerente.
- La catena Paper A → Paper F → Paper G è solida: A definisce ℰ e M(t), F costruisce i gate e il simulatore, G applica il tutto.

### Dipendenze circolari: **NESSUNA RILEVATA**

---

## Sintesi e Raccomandazioni

### Stato di solidità

```
ROBUSTO:
  ├── Definizione ρ_DND (§2.1) — ben definita con 3 misure indipendenti operazionali
  ├── Struttura dei gate (§3.1-3.4) — formalmente precisa con normalizzazione
  ├── Connessione a Paper A (§2.3) — M(t) correttamente integrato
  ├── Onestà sui limiti (§3.5 Open Problem, §5.1 Remark, §6.1 BBBV bound)
  ├── Appendici A e B — prove complete per Prop. 2.2 e Prop. 4.3
  └── Regime di validità esplicito per simulazione lineare (λ < 0.3-0.5)

FRAGILE:
  ├── Error model L_k^DND = L_k·(1-M(t)) — ad hoc, non derivato
  ├── Universalità perturbativa — argomento di continuità non sufficiente
  ├── Complessità O(n³·T) del simulatore — non dimostrata
  ├── Proposizione 2.3a — vaga e contradditoria con Def. 2.1
  └── Numerazione "Formula C1-C7" confonde con Paper C

MANCANTE:
  ├── TUTTI i dati numerici — non verificabili, sorgente non identificata
  ├── Test di universalità per δV > 0
  ├── Connessione esplicita a P5 (Autopoiesi) — il paper è Paper F, assegnato a P5!
  ├── Definizione formale del grafo di emergenza prima dei gate
  └── Formulazione unitaria di Grover speedup (§6.1 vs §6.4 inconsistenti)
```

### Piano d'azione proposto

**Priorità ALTA** (necessari per draft 3):
1. **C1**: Giustificare o dichiarare l'error model come fenomenologico. Testare contro ansatz alternativi.
2. **C2**: Declassare Prop. 3.5 a Congettura con evidenza perturbativa + numerica, o rafforzare con argomento algebrico.
3. **C3**: Rimuovere il claim di complessità polinomiale O(n³·T) o dimostrarlo rigorosamente.
4. **M1**: Condensare §6.5+§6.6 da ~650 righe a ~150 righe. Pseudocodice in supplementare.
5. **M2**: Risolvere la contraddizione tra Def. 2.1 e Prop. 2.3a.
6. **m1**: Rinominare "Formula C1-C7" in "Eq. (F.1)-(F.7)" o numerazione standard.

**Priorità MEDIA** (rafforzano significativamente):
7. **M3**: Aggiungere Definition 3.0 (Emergence Graph Construction) formale.
8. **M4**: Identificare e citare la sorgente dei "Corpus Simulations".
9. **M5**: Unificare §6.1 e §6.4 sulla questione Grover speedup.
10. Connettere esplicitamente a P5 (Autopoiesi): il circuito che genera il proprio stato computazionale.
11. **m3**: Distinguere chiaramente Shortcut_DND come strategia di compilazione, non gate.

**Priorità BASSA** (completezza):
12. **m2**: Spostare Notation Clarification (§1.1) in appendice.
13. **m4**: O formalizzare BQP_DND o rimuoverlo.
14. Eseguire le 4 validazioni numeriche (Test 1-4 del CALCOLO).
15. Connettere esplicitamente a P3 (Risultante) nel contesto IFS.

---

*Report generato dal Team D-ND — sessione 28/02/2026*
*Ruoli: φ FORMALISTA, ν VERIFICATORE, γ CALCOLO, τ TESSITORE*
*Esecutore: TM3 (VPS host)*
*Prossimo step: fix priorità ALTA → draft 3.0*
