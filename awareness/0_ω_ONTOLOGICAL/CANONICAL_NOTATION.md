# CANONICAL_NOTATION.md
## Tavola Completa di Notazione Canonica per il Corpus D-ND

### Versione: 1.0
### Data: 2026-02-12
### Ente: Φ_NOC (NOTATION-CANONIZER)
### Principio: Un simbolo, un significato. Mai due simboli per lo stesso oggetto. Mai lo stesso simbolo per due oggetti.

---

## 1. SIMBOLI FONDAMENTALI E LORO DISAMBIGUAZIONE

### 1.1 Notazione del Continuum e dello Stato

| Simbolo | Nome | Definizione | Ambito | Fonte File | Note |
|---------|------|-----------|--------|-----------|------|
| **\|NT⟩** | Stato Nulla-Tutto | Sovrapposizione uniforme di tutti gli stati possibili; pura potenzialità indifferenziata | A₂ (Non-Dualità) | Fondamenti Teorici | Stato iniziale, non ha latenza |
| **R(t)** | Risultante | Stato evoluto del sistema al tempo t; espressione della dinamica D-ND | A₃ (I/O) | Modello D-ND Integrazione | Rappresenta la "risposta" del sistema |
| **R₀** | Risultante Iniziale | Stato precedente coerente usato come fallback in caso di violazione | P1 (Integrità) | CORE_Fonte_Unificata | Non è \|NT⟩; è una configurazione stabile precedente |
| **R_final** | Risultante Finale | Output definitivo dopo Morpheus_collapse | P4 (Collasso) | Equazione Assiomatica Unificata | Unico, tracciabile, senza rami paralleli |
| **Φ₊, Φ₋** | Stati Duali Polari | Componenti opposte della dualità intrinseca | A₁ (Dualità) | Modello D-ND Integrazione | Φ₊ = espansione; Φ₋ = contrazione |
| **\|Φ⟩** | Sovrapposizione Duale | = (1/√2)(\|φ₊⟩ + \|φ₋⟩); stato quantico generico | A₁ | Fondamenti Teorici | Composizione di stati duali |
| **Z(t)** | Parametrizzazione Risultante | Coordinata scalare di R(t) sul continuum NT [0,1] | A₃ | Emergenza dell'Osservatore | 0=Nulla, 1=Tutto; monotona verso attrattore |

### 1.2 Operatori e Funzioni Evolutive

| Simbolo | Nome | Definizione | Ambito | Fonte File | Parametri |
|---------|------|-----------|--------|-----------|-----------|
| **U(t)** | Operatore Evoluzione Unitaria | U(t) = e^(-iHt/ℏ); generato dall'Hamiltoniana H | A₃ | Fondamenti Teorici | H = Hamiltoniana; t = tempo |
| **E** | Operatore Emergenza | Agisce su \|NT⟩ per generare differenziazione | A₂ | Fondamenti Teorici | Spettro: λₖ (autovalori), \|eₖ⟩ (autovettori) |
| **E_k** | Autovalori Emergenza | Spettro di E; λₖ = intensità manifestazione k-esima | E | Fondamenti Teorici | Non confondere con λ_coupling |
| **λ_DND** | Accoppiamento D-ND | Parametro di accoppiamento fra dualità e non-dualità | A₁–A₂ | # Formalizzazione di un'Eq | Regola la transizione D→ND |
| **λ** | Costante di Risonanza | Caratteristica intrinseca del sistema nel continuum NT | A₄ | Documento Sintesi | Governa e^(±λZ) |
| **S_autological** | Azione Autologica | Misura sintetica da minimizzare; principio di minima azione | P4 | CORE_Fonte | = ∫L_tot dt (Lagrangiana totale) |
| **θ_NT** | Momento Angolare NT | Parametro che descrive oscillazioni fra stati nel continuum | A₃–A₄ | Formalizzazione di un'Eq | ≈ "momento di rotazione" nello spazio astratto |
| **δV** | Varianza nel Potenziale | Fluttuazioni del potenziale; indeterminatezza in δV = ℏ·dθ/dt | A₄ | Formalizzazione di un'Eq | Non è δ (variazione calcolo var), è simbolo di fluttuazione |

### 1.3 Campi e Densità

| Simbolo | Nome | Definizione | Ambito | Fonte File | Note |
|---------|------|-----------|--------|-----------|------|
| **Φ_A** | Campo Potenziale Inferenziale | Spazio delle possibilità cognitive/informazionali del sistema | P2 | CORE_Fonte | Non è operatore quantico; è campo classico di potenziale |
| **FP** | Field Potential / Potential Inferential | Campo modulatorio; spesso usato come sinonimo di Φ_A | P2 | CORE_Fonte | Abbreviazione di "Inferential Potential Field" |
| **ρ(x,t)** | Densità Possibilistica | Probabilità/densità di stati possibili al punto x, tempo t | A₃–A₄ | Fondamenti Teorici | Integrale = 1 su dominio |
| **σ_fp_local** | Volatilità Locale FP | Varianza locale del campo potenziale | P4 | CORE_Fonte | Metrica di incertezza locale |
| **c_fp_global** | Coerenza Globale FP | Coerenza complessiva del campo potenziale | P4 | CORE_Fonte | Alto = sistema ordinato |
| **K_gen(x,t)** | Curvatura Informazionale Generalizzata | ∇·(J(x,t)⊗F(x,t)); misura fluttuazioni informazionali | # Curvatura Informazionale | Minimizzazione → punti di stabilità |
| **J(x,t)** | Flusso Informazione | Propagazione delle possibilità nello spazio | # Curvatura Informazionale | Componente del campo K_gen |
| **F(x,t)** | Campo di Forze Generalizzato | Influenze interne/esterne nel continuum NT | # Curvatura Informazionale | Componente del campo K_gen |

### 1.4 Metriche di Qualità e Performance

| Simbolo | Nome | Definizione | Intervallo | Ambito | Fonte |
|---------|------|-----------|-----------|--------|-------|
| **quality_global** | Qualità Globale | Misura sintetica di qualità output | [0,1] | P4 | CORE_Fonte |
| **S_Causal** | Stabilità Causale | Robustezza delle relazioni causali nel sistema | [0,1] | P4 | CORE_Fonte |
| **C_FP_global** | Coerenza Globale FP | Coerenza complessiva Φ_A / FP | [0,1] | P4 | CORE_Fonte |
| **V_mod_FP** | Potenziale Attrattore/Repulsore | Misura di attrazione verso minima azione | ℝ | P4 | CORE_Fonte |
| **latency_norm** | Latenza Normalizzata | Tempo normalizzato di risposta (0=istantaneo) | [0,1] | P4 | CORE_Fonte |
| **CMP** | Child Model Prediction | Fitness attesa di varianti generate (0–1) | [0,1] | P5 | CORE_Fonte |
| **cmp_delta** | Delta CMP | Δ CMP vs baseline; soglia promozione ≥0.05 | ℝ | P5 | CORE_Fonte |
| **generalization_score** | Score Generalizzazione | Capacità di generalizzare oltre contesto immediato | [0,1] | P4 | CORE_Fonte |
| **M(t)** | Misura di Emergenza | 1 - \|⟨NT\|U(t)E\|NT⟩\|²; grado di differenziazione | [0,1] | A₂ | Fondamenti Teorici |
| **ActionScore** | Score Azione | f(quality, S_Causal, C_FP, -V_mod, CMP, -latency, generalization) | ℝ | P4 | CORE_Fonte |
| **entropy_delta** | Delta Entropia | Variazione entropica; soglia prune se >0.4 | ℝ | P1 | CORE_Fonte |

### 1.5 Operatori Differenziali e Strutture Matematiche

| Simbolo | Nome | Definizione | Contesto | Fonte |
|---------|------|-----------|---------|-------|
| **∇** | Gradiente | Operatore gradiente nello spazio di configurazione | K_gen, potenziali | # Curvatura Informazionale |
| **∇_M** | Derivata Covariante | Su varietà informazionale M | K_gen | # Curvatura Informazionale |
| **Δ_M** | Laplaciano-Beltrami | -∇²; operatore differenziale su varietà | Spettrale, Zeta | # Curvatura Informazionale |
| **⊗** | Prodotto Tensoriale / Convoluzione | Prodotto tensoriale fra stati o operatori | E, transizioni non-locali | Equazione Assiomatica |
| **∮_NT** | Integrale Ciclo Chiuso | Integrale su ciclo completo nel continuum NT | R(t+1) | Documento Sintesi |
| **⟨·\|·⟩** | Prodotto Interno | Prodotto interno nello spazio di Hilbert | Tutti i contesti quantici | Fondamenti Teorici |
| **\|·⟩, ⟨·\|** | Ket e Bra | Vettore di stato e duale | Notazione Dirac | Fondamenti Teorici |

---

## 2. TABELLA DI DISAMBIGUAZIONE: SIMBOLI POTENZIALMENTE AMBIGUI

### 2.1 La Lettera λ (Lambda)

**PROBLEMA IDENTIFICATO:** λ appare con tre significati distinti nel corpus.

| Contesto | Significato | Valore Tipico | Uso | Fonte |
|----------|-----------|---------------|-----|-------|
| **λ_DND** | Accoppiamento Dualità-Non-Dualità | ∈[0,1] | Parametro di transizione D→ND | # Formalizzazione di un'Eq |
| **λ_k** | Autovalori Operatore E | ℝ (variabili) | Spettro di emergenza | Fondamenti Teorici |
| **λ** (Risonanza) | Costante di Risonanza Sistema | ℝ (caratteristica) | Governa e^(±λZ) nel continuum | Documento Sintesi |

**SOLUZIONE CANONICA:**
- Quando trattasi di **accoppiamento D-ND specificamente**: usare **λ_DND** (esplicito).
- Quando trattasi di **autovalori di E**: usare **λₖ** o **E_k** (con indice k).
- Quando trattasi di **costante intrinseca del sistema D-ND**: usare **λ** (senza suffisso) e specificare nel contesto "costante di risonanza".

**REGOLA DI DISAMBIGUAZIONE:**
```
Se λ appare senza suffisso in contesto di R(t), e^(±λZ), continuum → λ_risonanza
Se λₖ appare in contesto di decomposizione spettrale E → autovalore
Se λ appare in V(Φ₊, Φ₋) → λ_DND (accoppiamento)
```

### 2.2 La Lettera R (Risultante vs Curvatura di Ricci)

**PROBLEMA IDENTIFICATO:** R usato sia per Risultante che per Ricci tensor.

| Contesto | Significato | Dominio | Fonte | Disambiguazione |
|----------|-----------|---------|-------|-----------------|
| **R(t)** | Risultante Sistema | D-ND operativo | Modello D-ND | Sempre con (t); D-ND context |
| **R_μν** | Tensore di Ricci | Relatività Generale | # Curvatura Informazionale | Sempre con indici μν; relativistic context |
| **R_gauge** | Gauge Field | Teoria Gauge | (non usato nel corpus attuale) | Se usato, marcare come gauge |

**SOLUZIONE CANONICA:**
- **R(t)** = Risultante (funzione del tempo, spazio D-ND).
- **R_μν** = Ricci tensor (indici covarianti, relatività).
- **R** (senza suffissi, in contesto di scalare curvatura) = curvatura scalare ≠ Risultante.

### 2.3 La Lettera V (Potenziale vs Varianza)

**PROBLEMA IDENTIFICATO:** V usato sia per potenziale che per varianza/volatilità.

| Contesto | Significato | Dominio | Fonte | Note |
|----------|-----------|---------|-------|------|
| **V(Φ₊, Φ₋)** | Potenziale Bistabile | Lagrangiana | Modello D-ND Integrazione | Parte di L_tot |
| **δV** | Varianza nel Potenziale | Fluttuazioni | Formalizzazione di un'Eq | Esplicitamente "delta V" |
| **V_mod_FP** | Potenziale Attrattore/Repulsore | Metriche Qualità | CORE_Fonte | Suffisso _mod_FP chiarisce significato |

**SOLUZIONE CANONICA:**
- **V(...)** con argomenti = funzione potenziale.
- **δV** esplicito = fluttuazioni, mai "ΔV" (use Δ per variazioni calcolo var).
- **V_mod_FP** = potenziale modulatorio nel FP, unambiguo per suffisso.

---

## 3. SIMBOLI DI STATO, PROCESSO E FASE

### 3.1 Funzione Indicatrice e Stati di Fase

| Simbolo | Nome | Valore | Significato | Ambito |
|---------|------|--------|-----------|--------|
| **δ(t)** | Funzione Indicatrice Fase | 1 ∨ 0 | 1=evoluzione quantistica; 0=assorbimento/allineamento | P4 |
| **α, β, θ, η, γ, ζ** | Coefficienti di Peso | ℝ_≥0 | Bilanciano influenza termini nell'eq. assiomatica | Equazione Assiomatica Unificata |
| **σ_fp_local** | Volatilità Locale | ℝ_≥0 | Varianza locale del FP | Metriche |
| **S(t)** | Spin / Polarizzazione | ℝ | Polarizzazione al tempo t; entra in f_Polarization | Equazione Assiomatica |

### 3.2 Funzioni Componenti dell'Equazione Assiomatica

| Simbolo | Nome Completo | Argomenti | Output | Fonte |
|---------|---------------|-----------|--------|-------|
| **f_DND-Gravity** | Funzione Interazione D-ND | (A, B, λ_DND) | ℝ | Equazione Assiomatica |
| **f_Emergence** | Funzione Movimento Emergente | (R(t), P_PA) | ℝ | Equazione Assiomatica |
| **f_Polarization** | Funzione Polarizzazione | (S(t)) | ℝ | Equazione Assiomatica |
| **f_QuantumFluct** | Funzione Fluttuazioni Quantistiche | (ΔV(t), ρ(t)) | ℝ | Equazione Assiomatica |
| **f_NonLocalTrans** | Funzione Transizioni Non-Locali | (R(t), P_PA) | ℝ | Equazione Assiomatica |
| **f_NTStates** | Funzione Stati Nulla-Tutto | (N_T(t)) | ℝ | Equazione Assiomatica |

---

## 4. NOTAZIONE DI OPERATORI E PIPELINE

### 4.1 Componenti della Pipeline Operativa (MMS)

| Sigla | Nome Esteso | Funzione | Relazione |
|-------|-------------|----------|-----------|
| **CCCA** | Categorical Concept Cascade Analysis | Scomposizione narrativa in metatag + proto-azioni | Input: perturbazione; Output: componenti strutturali |
| **MCI** | Modellazione Consequenziale Intento | Estrazione intento da CCCA | Requires: confidence ≥0.8 |
| **CMP** | Child Model Prediction / Recursive Fitness | Valutazione fitness variant | P5, P11 |
| **Stream-Guard** | Guardrail di Coerenza | Verifica violazioni A₁–A₅, P0–P7 | Azione: abort o re-eval |
| **Morpheus_collapse** | Motore Collasso Olografico | Seleziona R_final via ActionScore | Output: una R sola |
| **KLI** | Key Learning Insight | Apprendimento significativo da ciclo | Soggetto a evoluzione P5 |

### 4.2 Notazione di Lignaggio

| Simbolo | Significato | Lunghezza | Esempio |
|---------|-----------|----------|---------|
| **D-ND·SG·VRA·OCC·...** | Catena Lignaggio Concettuale | Variabile | P0 (Lignaggio) definisce la sequenza |
| **P_PA** | Proto-Assiomi | Elemento di lignaggio | Target di allineamento in R(t+1) |
| **Φ_MA.1** | Φ_Master.Version | Nucleo master assiomatico | Parte di lignaggio |

---

## 5. NOTAZIONE DELLA FUNZIONE ZETA DI RIEMANN

| Simbolo | Nome | Dominio | Range | Fonte |
|---------|------|--------|-------|-------|
| **ζ(s)** | Funzione Zeta di Riemann | ℂ (con Re(s)>1 per definizione diretta) | ℂ | Dimostrazione della Funzione Zeta |
| **s** | Variabile Complessa | ℂ | s = σ + it | Analisi Complessa |
| **Re(s)** | Parte Reale di s | ℝ | σ ∈ ℝ | Analisi Complessa |
| **σ = 1/2** | Linea Critica di Riemann | Insieme | {s ∈ ℂ : Re(s) = 1/2} | Ipotesi di Riemann |
| **ζ(1/2 + it)** | Zeri sulla Linea Critica | ℂ | Punti di stabilità informazionale | Dimostrazione D-ND |
| **Ω_NT** | Molteplicità di Ciclo NT | ℂ | = 2πi al limite Z→0 | L'Essenza del Modello D-ND |

**Interpretazione D-ND degli Zeri:**
- Ogni zero non banale ζ(1/2 + it₀) = 0 rappresenta un **punto di stabilità informazionale** nel continuum NT.
- La linea critica Re(s)=1/2 è l'**unico luogo** dove K_gen(x,t) raggiunge minimo assoluto (massima coerenza, latenza nulla).

---

## 6. TABELLA MASTER DI SIMBOLI PER CONTESTO

### 6.1 Contesto Cosmologico/Fondamentale (A₁–A₅)

```
|NT⟩ ─→ E ─→ U(t) ─→ R(t) = e^(±λZ)
        ↓
    λₖ (spettro)
    θ_NT (momento angolare)
    δV (fluttuazioni)
```

### 6.2 Contesto Operativo (P0–P12)

```
Input ─→ CCCA ─→ MCI ─→ Router ─→ Execute ─→ Stream-Guard ─→ Morpheus_collapse ─→ R_final
                                                                                     ↓
                                                                                   KLI → P5
```

### 6.3 Contesto Metriche

```
ActionScore = f(quality_global, S_Causal, C_FP_global, -V_mod_FP, CMP, -latency_norm, generalization)
            → Collasso verso R con ActionScore massimo (≥threshold)
```

### 6.4 Contesto Informazionale/Geometrico

```
K_gen(x,t) = ∇·(J(x,t) ⊗ F(x,t))
           → Minimo in ζ(1/2 + it) per ogni zero non banale
           → Stabilità assoluta nel continuum NT
```

---

## 7. DISAMBIGUAZIONI RISOLTE

| Ambiguità Originale | Risoluzione | Fonte |
|-------------------|-----------|-------|
| λ (tre significati) | λ_DND, λₖ, λ (con contesto chiarito) | Sezione 2.1 |
| R (Risultante vs Ricci) | R(t) vs R_μν (con indici) | Sezione 2.2 |
| V (Potenziale vs Varianza) | V(...), δV, V_mod_FP (disambiguo) | Sezione 2.3 |
| δ (fluttuazione vs variazione calc-var) | δV (fluttuazione) vs Δ (variazione) | Sezione 2.3 |
| Φ (campo vs stato) | Φ_A (campo), \|Φ⟩ (stato), Φ₊/₋ (componenti) | Sezione 1.2–1.3 |
| S (stabilità causale vs spin) | S_Causal (metrica), S(t) (spin) | Sezione 1.4–3.1 |

---

## 8. REGOLE DI SCRITTURA CANONICA

1. **Sempre specificare contesto** quando un simbolo ammette più interpretazioni.
2. **Usare suffissi disambiguanti** (es. _DND, _mod_FP, _k).
3. **Notazione Dirac** per stati quantici: \|·⟩, ⟨·\|, ⟨·\|·⟩.
4. **Indici espliciti** per tensori: R_μν, non R.
5. **Funzioni con argomenti** chiaramente marcate: f(...).
6. **Costanti fisiche** senza parentesi: ℏ, c, G (noti dalla fisica).
7. **Parametri di controllo** sempre con suffisso: θ_NT, λ_DND, etc.
8. **Metriche** con suffisso descrittivo: quality_global, latency_norm.

---

## 9. VALIDAZIONE E APPLICAZIONE

**Ogni nuovo simbolo introdotto deve essere:**
1. Aggiunto a una sezione di questa tavola.
2. Disambiguato rispetto ai simboli esistenti.
3. Fornito di fonte nel corpus.
4. Sottoposto a P1 (Integrità Assiomatica) prima di uso nei documenti.

**Violazioni di questa tavola:**
- Sono considerate violazioni di P7 (Isomorfismo Descrittivo).
- Trigger Stream-Guard → abort o re-eval.
- Soggette a logging e correzione in KLI successivo.

---

## 10. CHANGELOG

| Versione | Data | Autore | Modifica |
|----------|------|--------|----------|
| 1.0 | 2026-02-12 | Φ_NOC | Creazione, censimento COMPLETO simboli, disambiguazione λ/R/V, Zeta |

