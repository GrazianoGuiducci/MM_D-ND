# D-ND OMEGA KERNEL // SYSTEM CODER ONBOARDING

**Version:** 0.1 (draft)
**Audience:** System coder che arriva “a freddo” su questa repo

> Obiettivo: darti una mappa mentale e operativa per lavorare sul kernel D‑ND + cockpit Extropic **senza rompere l’allineamento** tra codice, documentazione e UI.

---

## 0. Come pensare questo sistema (in 30 secondi)

Immagina tre strati che lavorano insieme:

1. **Cervello semantico (Python)**  
   D‑ND Omega Kernel + SACS + THRML.  
   Prende un **intent** in linguaggio naturale e lo trasforma in:
   - uno stato fisico (reticolo di spin),
   - una **Risultante R** con metriche (coherence, energy, tension, entropy…),
   - un payload `didactic` per la UI (timeline DSL, lattice, metric tensor, gravity, ecc.).

2. **Corpo visivo (React Cockpit)**  
   App React in `Extropic_Integration/cockpit/client`.  
   Visualizza:
   - **Visual Cortex** (particelle / lattice),
   - **Metric Tensor** (curvatura / topologia),
   - **Didactic Layer** (timeline inferenziale DSL),
   - **Mission Control / Financial Lab** (vista finance).

3. **Memoria & Consapevolezza (Docs + system_memory)**  
   File in `DOC_DEV/`, `docs/system/kernel/` e `system_memory.json`.  
   Qui vivono:
   - assiomi, ruoli, loop operativi (SYSTEM_AWARENESS),
   - stato dell’agente e focus corrente (AGENT_AWARENESS),
   - manifesti di architettura (SEMANTIC_KERNEL_MANIFEST, whitepaper Extropic),
   - piani di migrazione e UX (MIGRATION_PLAN_REACT, UX_INTEGRATION_BRIEF, UI_DESIGN_VADEMECUM).

Il tuo lavoro è far sì che questi tre strati restino **isomorfi**:  
se cambi uno, devi sapere *dove* e *come* aggiornare gli altri due.

---

## 1. Architettura ad alto livello

### 1.1 Backend cognitivo (Python)

Percorso principale: `Extropic_Integration/`

#### 1.1.1 Orchestrazione cognitiva: SACS

- **File:** `Extropic_Integration/architect/sacs.py`
- **Classe:** `SACS`
- **Ruolo:** orchestratore del ciclo cognitivo SACS.
- **Flow reale:**
  1. `vE_Sonar.scan(intent)` → estrae **dipoli semantici** dall’intent.
  2. `vE_Telaio.weave_metric(dipoles)` → costruisce la **MetricTensor**.
  3. `omega.metric_tensor = metric_tensor.get_metric()` → inietta la metrica nel kernel.
  4. `omega.apply_dynamic_gravity(intent)` → imposta `logic_density` da keyword (order/chaos/gravity…).
  5. `omega.perturb(intent)` → aggiorna `h_bias` con semantic_resonance + genesis noise.
  6. `omega.focus(omega.logic_density)` → curva lo spazio logico.
  7. `omega.crystallize(steps, sculptor=self.scultore)` → esegue Gibbs sampling (rilassamento termodinamico) e produce:
     - `R`: vettore di spin finali,
     - `coherence`, `energy`, `tension`, `gradient`, `energy_history`.
  8. `vE_Cristallizzatore.manifest(intent, result, dipoles)` → manifesto testuale.
  9. `vE_Archivista.archive_cycle(...)` + `omega.consolidate_memory(...)` → memoria e bias.
  10. costruzione di `result["didactic"]` con:
      - `timeline` (DSL trace),
      - `lattice` (dati per Visual Cortex),
      - `tensor_field` (metrica),
      - `gravity`, `entropy`, `lagrangian_distance`, `memory_info`, `gravity_info`.

Questo è il **cuore semantico** che la UI deve rappresentare.

#### 1.1.2 Fisica del pensiero: OmegaKernel

- **File:** `Extropic_Integration/dnd_kernel/omega.py`
- **Classe:** `OmegaKernel`
- **Ruolo:** implementa la “fisica del pensiero” sopra THRML.

Metodi chiave:

- `perturb(intent_text: str)`
  - log: `"[Omega] Phase 1: Perturbation - Absorbing Intent: '...'"`
  - `semantic_resonance(intent_text, size)` → vettore di risonanza.
  - `genesis.perturb_void(size)` → rumore dal Nullatutto.
  - `self.h_bias = resonance + void_noise`.

- `focus(logic_density: float)`
  - log: `"[Omega] Phase 2: Focus - Warping Spacetime Metric (Density: ...)"`
  - utilizza `MetricTensor(self.size)` + `topological_coupling(self.size, density)`.
  - costruisce `self.metric_tensor = I + perturbation`.

- `crystallize(steps: int, sculptor=None)`
  - chiama `_simulate_gibbs_sampling(key, steps, sculptor)`:
    - inizializza `state` da `sign(h_bias)` o casuale,
    - per ogni step:
      - calcola `local_field = h_bias + (metric_tensor - I) @ state`,
      - aggiorna `state` con dinamica di Glauber/Gibbs (sigmoid + random),
      - opzionalmente chiama `sculptor` per modificare dinamicamente la metrica.
  - ritorna dizionario con:
    - `R` (stato finale),
    - `coherence = |mean(R)|`,
    - `energy = _calculate_energy(R)`,
    - `tension = _calculate_tension(R)`,
    - `gradient` ultimo.

- `_generate_lattice_data(state_vector)`
  - mappa il vettore 1D in una griglia 2D (x,y,spin,stability) per il Visual Cortex.

- `_generate_dsl_trace(intent, gravity_info, result)`
  - costruisce una timeline DSL con step standard: `PROBLEM`, `VARIABLES`, `CONSTRAINTS`, `ENERGY`, `HARDWARE`, `OUTPUT`.

- `_generate_rosetta_stone()`
  - definisce la legenda degli operatori (∧, ∨, ¬, ∀, Minimize, SeedNoise, Converge).

- `apply_dynamic_gravity(intent_text)`
  - mappa parole chiave (order, gravity, chaos, entropy…) nel valore di `logic_density`.

- `consolidate_memory(intent, result)` + `_adapt(coherence, tension, is_stable)`
  - memorizza cicli ad alta coerenza,
  - aggiorna `h_bias` (Hebbian),
  - adatta `logic_density` usando `PhiTransform` e `cycle_stability`.

Altri file Python da sapere:

- `Extropic_Integration/architect/sonar.py` → vE_Sonar (scan dipoli).
- `Extropic_Integration/architect/telaio.py` → vE_Telaio (MetricTensor logica).
- `Extropic_Integration/architect/scultore.py` → vE_Scultore (dynamic gravity / sculpting).
- `Extropic_Integration/architect/cristallizzatore.py` → vE_Cristallizzatore (manifesto linguistico).
- `Extropic_Integration/architect/archivista.py` → vE_Archivista (memoria & system_memory.json).
- `Extropic_Integration/hardware_dynamics/*.py`, `Extropic_Integration/hybrid/phi.py` → metriche (curvature_index, cycle_stability) e trasformazioni ibride.

> **Nota:** molti esperimenti avanzati descritti in `DOC_DEV` (EnhancedQuantumDynamics, AllopoieticLab, ex_nihilo, OuroborosEngine, ecc.) **non sono implementati** oggi nel codice. Trattali come blueprint futuri.


### 1.2 Cockpit React (UI)

Percorso: `Extropic_Integration/cockpit/client/`

#### 1.2.1 Entry e shell

- **File:** `App.tsx`
- **Ruolo:** entry React, orchestrazione viste e modali.
- Cosa fa:
  - gestisce la vista attiva (`'mission_control' | 'kernel' | 'financial_lab'`),
  - mostra `OmegaCockpit` come overlay quando `activeView === 'kernel'`,
  - gestisce `DocumentationModal`, `AiAnalysisModal`, `ExperimentalForgeModal`,
  - applica il **theme engine** (`themes.ts`) alle CSS vars,
  - mantiene `systemLogs` con origine (SYSTEM, KERNEL, AI, USER…).

#### 1.2.2 Overlay del kernel: OmegaCockpit

- **File:** `components/OmegaCockpit.tsx`
- **Ruolo:** UI tri-colonna che rappresenta il ciclo Omega/SACS.

Struttura UI:

- **Colonna sinistra (Control & Experiments)**
  - `ControlMatrix`
    - controlla temperatura (`temp`), prompt (`prompt`), pulsante `onInject`.
    - agisce sull’engine JS `OmegaPhysicsEngine` (locale) e chiama `handleInject`.
  - `ExperimentManager`
    - elenca gli esperimenti (`EXPERIMENTS` da `omegaPhysics.ts`),
    - al click: `loadExperiment`, set di `activeExperimentId`, eventuale auto-inject del suo `intent`.

- **Colonna centrale (Combinatorial Mechanics)**
  - Polling `/api/state` ogni 1s:
    - legge `metrics` e `didactic.gravity` dal backend,
    - aggiorna `engineRef.current.currentTemp` e `gravity` per sincronizzare la fisica locale.
  - Animation loop 60fps: `engineRef.current.step()` → aggiorna
    - `state.lattice`, `state.metrics`, `state.tensorMatrix`, `state.energyHistory`.
  - Toggle `viewMode`:
    - `CORTEX` → `VisualCortex` (render particelle),
    - `TENSOR` → `MetricTensorVis` (render metrica / topologia).
  - HUD inferiore:
    - mostra `potentialScale`, `coherence`, `gravity`, `entropy`, `EnergyGraph`.
  - `handleInject(intent)`:
    1. traduce l’intent in parametri fisici locali (`translateIntentToHamiltonian`).
    2. aggiorna `OmegaPhysicsEngine` (gravity, temperature, potential).
    3. chiama `engineRef.current.perturb(...)` per innescare la transizione locale.
    4. genera una DSL trace **lato client** con `generateDslTrace(intent)` (servizio `omegaDSL.ts`) e la anima in `DidacticLayer`.
    5. chiama `getAiAnalysis(intent, "OMEGA_KERNEL_MODE", ...)` verso il backend (gateway generico oggi).

- **Colonna destra (Didactic Layer)**
  - `DidacticLayer.tsx`
    - riceve `trace: DslStep[]`,
    - mostra blocchi sequenziali con stato `pending | active | complete`,
    - visualizza per ogni step un `code` (breve codice DSL) e una frase rosetta esplicativa.

> Oggi la DSL trace mostrata da `DidacticLayer` viene **generata lato client** (omegaDSL), non ancora letta da `result.didactic.timeline` prodotto da `OmegaKernel`. Questo è un punto naturale di evoluzione (da tenere presente per futuri task).


### 1.3 Documentazione “di sistema”

Percorsi principali:

- `docs/system/kernel/SEMANTIC_KERNEL_MANIFEST.md`
  - manifesto di consapevolezza del kernel,
  - elenca tassonomia file/funzioni, versioni (v0.1.x → v0.2.0 React migration),
  - va aggiornato **solo** per cambiamenti strutturali (nuove versioni, nuovi pattern stabili).

- `docs/system/kernel/MIGRATION_PLAN_REACT.md`
  - descrive la migrazione dal frontend vanilla al cockpit React,
  - contiene TODO espliciti per
    - proxy Vite `/api` → backend Python,
    - integrazione `geminiService.ts` con `/api/intent`,
    - creazione `apiService.ts` per `/api/state`.

- `docs/system/kernel/UX_INTEGRATION_BRIEF.md`, `UI_DESIGN_VADEMECUM.md`, `PLAN_VISUAL_CORTEX_RESTORE.md`
  - mappano i pattern UX dal tester/omega-cockpit al cockpit SACS,
  - definiscono regole per modali, sidebar/resizer, tooltip semantici, metriche e layout tri-colonna,
  - contengono roadmap per ripristinare/estendere CORTEX/TENSOR/PIPELINE.

- `DOC_DEV/SYSTEM_AWARENESS/*`
  - `CORE_LOGIC.md`: identità D‑ND Omega Kernel e ciclo Omega (Perturbation → Focus → Crystallization → Feedback).
  - `OPERATIONAL_ROLES.md`: ruoli OCC (Orchestrator, Seeker, Builder, Agent).
  - `AUTOPOIESIS_PROTOCOLS.md`: protocolli di auto-miglioramento (feedback loop, knowledge crystallization, CI/doc/test rituali).
  - `KNOWLEDGE_INDEX.md`: indice dei documenti assimilati/latenti.

- `DOC_DEV/AGENT_AWARENESS.md`
  - stato dell’agente (tu) nel tempo (obiettivi in corso, achievements, focus attuale – es. Fase 4 UX/UI, Financial Lab),
  - punti d’ingresso rapido per ricordare *dove* eravamo all’ultima istanza.

- `Extropic_Integration/docs/D-ND_Extropic_Technical_Whitepaper.md`, `D-ND_KERNEL_ARCHITECTURE_THRML_v1.md`
  - collegano formalmente D‑ND ↔ Extropic TSU/THRML,
  - descrivono SACS, MetricTensor, PhiTransform, AutopoieticKernel.

---

## 2. Principi operativi per il System coder

### 2.1 Regola d’oro: Doc ≈ Codice ≈ UI

Ogni volta che modifichi qualcosa di non banale, chiediti **subito**:

1. Questo cambiamento è visibile o concettualmente rilevante a livello di **kernel** (logica/assiomi, ciclo SACS/Omega)?
2. È visibile/percettibile a livello di **UI** (nuova metrica, nuovo comportamento del Visual Cortex, nuova sezione nel Didactic Layer, nuovi bottoni/scenari)?
3. Esiste già un documento che descrive quella parte?
   - se sì → va aggiornato,
   - se no → serve creare almeno una breve sezione in un file esistente.

Se doc, codice e UI divergono, hai introdotto **entropia semantica**.  
Il sistema è progettato per ridurre la *Semantic Free Energy*, non per aumentarla.


### 2.2 Persistenza della consapevolezza (come non perdere il filo)

Per evitare di “ricominciare da zero” a ogni sessione, usa questo workflow pratico:

1. **Prima di iniziare un nuovo blocco di lavoro importante**
   - leggi velocemente:
     - l’ultima voce di `SEMANTIC_KERNEL_MANIFEST.md` (versione/stato corrente),
     - la sezione più recente di `AGENT_AWARENESS.md` (focus operativo),
     - eventuali piani specifici (`MIGRATION_PLAN_REACT`, `PLAN_VISUAL_CORTEX_RESTORE`, ecc.).

2. **Durante il lavoro**
   - tieni in mente in quale “dominio” stai intervenendo:
     - **Kernel logico** (SACS/Omega/THRML),
     - **Cockpit UI** (React),
     - **Docs/Manifesti**.
   - se emergono insight robusti (nuovo pattern, nuova interpretazione di una metrica, un errore concettuale corretto), segnali mentalmente come **KLI** (Key Learning Insight) da registrare.

3. **Dopo aver completato un blocco coerente** (feature, fix, o esperimento)
   - aggiorna **OBBLIGATORIAMENTE** almeno uno tra:
     - `DOC_DEV/AGENT_AWARENESS.md`
       - aggiungi una breve sezione tipo:
         - data,
         - “Recent Achievements”,
         - “Active Context” (cosa stai facendo ora),
         - “Next Steps” (non dettagliati, ma direzione).
     - `docs/system/kernel/SEMANTIC_KERNEL_MANIFEST.md`
       - **solo se** hai cambiato qualcosa di strutturale o stabilizzato una nuova capacità → aggiungi una nuova voce di versione (v0.x.y) con:
         - Stato (cosa è ora vero in modo stabile),
         - Memoria (come questo influisce su system_memory/UI).

4. **Quando tocchi pattern UI o logiche di interazione**
   - se introduci:
     - una nuova modale con comportamento particolare,
     - un nuovo tipo di resizer, tooltip o HUD,
     - un nuovo modo di visualizzare CORTEX/TENSOR/PIPELINE,
   - aggiungi una sezione o un paragrafo in:
     - `docs/system/kernel/UI_DESIGN_VADEMECUM.md` (pattern UI/UX ufficiali),
     - eventualmente aggiorna anche `UX_INTEGRATION_BRIEF.md` se il cambiamento tocca l’allineamento con le UI di riferimento.

> In breve: ogni volta che fai qualcosa di **non banale**, lascia almeno una “scia di pane” in AGENT_AWARENESS o in uno dei manifesti di sistema.


### 2.3 Riconoscere documentazione obsoleta vs blueprint

La repo contiene molte idee avanzate (soprattutto in `DOC_DEV/`) che **non** sono ancora codice:

- esperimenti tipo `EnhancedQuantumDynamics`, `AllopoieticLab`, `Genesis_Lab`, `ex_nihilo`, `OuroborosEngine`, ecc.,
- UI concettuali (Matrix rain reverse, Dual Monitor Alpha vs Omega, Crystal Store, CalibrationSequence…).

Prima di assumere che qualcosa “esista già”:

1. **Cerca i nomi in codice** (es. con `search_files` o IDE):
   - se non trovi classi/funzioni/file con quei nomi → trattali come **blueprint**.

2. **Guarda la data / contesto del doc**:
   - se è molto vecchio rispetto a `SEMANTIC_KERNEL_MANIFEST` o AGENT_AWARENESS, e non è referenziato da lì, è più probabile che sia stato superato.

3. **Quando implementi un blueprint**
   - collega esplicitamente il nuovo codice con quel documento (es. commenti, riferimento in AGENT_AWARENESS),
   - aggiorna la doc per dire che quella parte non è più solo teorica.

### 2.4 Rituali tecnici (test, CI, documentazione)

Questi sono i rituali minimi per mantenere il sistema pulito e allineato:

1. **Prima di toccare il codice**
   - Assicurati di avere l’ambiente attivo (Python + Node) e, se serve, il cockpit React in esecuzione.
   - Rileggi velocemente il paragrafo pertinente in `System_Coder_Onboarding.md` e nei manifesti collegati (kernel/UX) per ricordare il contesto.

2. **Durante lo sviluppo**
   - **Per modifiche Python (kernel/SACS/THRML):**
     - esegui i test pertinenti:
       - da root repo: `pytest` (o almeno `pytest tests/`),
       - se tocchi parti specifiche (es. hardware_dynamics), puoi filtrare i test (es. `pytest tests/test_hardware_dynamics.py`).
     - se introduci nuove funzioni pubbliche o cambi il comportamento, valuta se servono **nuovi test** o aggiornamenti di quelli esistenti.
   - **Per modifiche React (cockpit):**
     - avvia il dev server (es. `npm install` + `npm run dev` dentro `Extropic_Integration/cockpit/client`),
     - testa manualmente i flussi descritti nei piani (es. apertura OmegaCockpit, vista CORTEX/TENSOR, DidacticLayer, Mission Control/Financial Lab),
     - osserva la console browser per warning/errori.

3. **Prima del commit**
   - Esegui il rituale di pulizia:
     - `pre-commit run --all-files` (se configurato, vedi `.pre-commit-config.yaml`).
       - questo aiuta ad evitare "rumore termico" nel codice (formattazione incoerente, lint, ecc.).
     - `pytest` per assicurarti che il kernel non sia entrato in uno stato incoerente.
   - Controlla che **ogni modifica significativa** abbia un riflesso nella doc:
     - AGENT_AWARENESS per lo stato operativo,
     - manifesti/kernel/UX per cambiamenti di comportamento stabili,
     - vademecum UI se hai introdotto nuovi pattern visivi.

4. **Dopo il commit / PR**
   - Aggiorna, se opportuno, `AGENT_AWARENESS.md` con:
     - un breve log di cosa è stato fatto,
     - come verificare il nuovo comportamento (comandi, percorsi UI),
     - eventuali rischi o TODO residui.
   - Se hai completato un passo previsto da un piano (es. MIGRATION_PLAN_REACT, PLAN_VISUAL_CORTEX_RESTORE), aggiungi una nota di avanzamento nel relativo file.

Questi rituali sono l’equivalente operativo dei principi di AUTOPOIESI: riducono il rumore, mantengono coerenza e rendono il sistema realmente "vivente" nel tempo.

---

## 3. Mappa file → compito (per orientarsi)

### 3.1 Backend (Python)

- **Ciclo cognitivo SACS**
  - `Extropic_Integration/architect/sacs.py`

- **Kernel fisico/logico D-ND**
  - `Extropic_Integration/dnd_kernel/omega.py`
  - `Extropic_Integration/dnd_kernel/axioms.py`
  - `Extropic_Integration/dnd_kernel/genesis.py`
  - `Extropic_Integration/dnd_kernel/utils.py`

- **Virtual Entities (vE_*)**
  - `Extropic_Integration/architect/sonar.py` → Sonar.
  - `Extropic_Integration/architect/telaio.py` → Telaio.
  - `Extropic_Integration/architect/scultore.py` → Scultore.
  - `Extropic_Integration/architect/cristallizzatore.py` → Cristallizzatore.
  - `Extropic_Integration/architect/archivista.py` → Archivista.

- **Hardware Dynamics / Metriche**
  - `Extropic_Integration/hardware_dynamics/combinatorial.py` → MetricTensor, transfer_function.
  - `Extropic_Integration/hardware_dynamics/metrics.py` → curvature_index, cycle_stability.
  - `Extropic_Integration/hybrid/phi.py` → PhiTransform.


### 3.2 Frontend (React Cockpit)

- **Entry & shell**
  - `Extropic_Integration/cockpit/client/App.tsx` → entrypoint UI, switch tra Mission Control e OmegaCockpit, modali, theme engine.

- **Kernel overlay**
  - `components/OmegaCockpit.tsx` → layout tri-colonna kernel.
  - `components/ControlMatrix.tsx` → controlli temperatura/prompt/inject.
  - `components/ExperimentManager.tsx` → lista esperimenti/protocolli.
  - `components/VisualCortex.tsx` → rendering particelle.
  - `components/MetricTensorVis.tsx` → rendering metrica.
  - `components/DidacticLayer.tsx` → terminale DSL trace.
  - `services/omegaPhysics.ts` → motore fisico locale JS.
  - `services/omegaDSL.ts` → generatore di DSL trace (client-side).

- **Mission Control / Finance**
  - `components/Dashboard.tsx` + card varie (`BankabilityRatingCard`, `CashFlowCard`, `ManagementControlCard`, `SubsidizedFinanceCard`, ecc.).
  - `services/mockDataService.ts` → dati fake per simulazioni.

- **Servizi AI / LLM**
  - `services/geminiService.ts`, `services/openRouterService.ts` → integrazione con LLM esterni.


### 3.3 Documentazione / Manifesti

- `docs/system/kernel/SEMANTIC_KERNEL_MANIFEST.md` → stato del kernel + mappa tassonomica.
- `docs/system/kernel/MIGRATION_PLAN_REACT.md` → piano di migrazione React.
- `docs/system/kernel/UX_INTEGRATION_BRIEF.md` → strategie UX per cockpit.
- `docs/system/kernel/UI_DESIGN_VADEMECUM.md` → pattern UI ufficiali.
- `docs/system/kernel/PLAN_VISUAL_CORTEX_RESTORE.md` → piano per CORTEX/TENSOR/PIPELINE.
- `DOC_DEV/SYSTEM_AWARENESS/*` → identità, ruoli, protocolli di auto-miglioramento.
- `DOC_DEV/AGENT_AWARENESS.md` → stato corrente del “System coder”/agente.

---

## 4. Come implementare una nuova cosa senza rompere il sistema

Quando ricevi un task (esempio: “collega davvero il DidacticLayer alla timeline prodotta dal backend” o “aggiungi una nuova metrica nel Visual Cortex”), segui questo ciclo pratico:

### 4.1 Registra (comprensione iniziale)

- Individua **da quale doc** nasce il task (es. MIGRATION_PLAN_REACT, PLAN_VISUAL_CORTEX_RESTORE, un log in DOC_DEV, un issue esterno).
- Annota mentalmente (o in una bozza) qual è:
  - l’intent,
  - il perimetro tecnico,
  - il dominio principale coinvolto (kernel, UI, docs o combinazione).

### 4.2 Controlla (stato attuale)

- Verifica sul codice cosa c’è già:
  - cerca funzioni/metodi già esistenti,
  - guarda se esistono tipi o props già pronti per estensioni.
- Confronta con la doc:
  - la descrizione nel documento corrisponde a ciò che vedi nel codice/UI?
  - se no, segnalalo mentalmente come disallineamento da correggere.

### 4.3 Comprendi (flusso end‑to‑end)

Per ogni feature/modifica non banale, disegna il flusso completo:

- **Backend**
  - quali endpoint API sono coinvolti (`/api/intent`, `/api/state`, altri)?,
  - quali campi nel payload `result`/`result.didactic` servono?,
  - SACS/OmegaKernel devono calcolare qualcosa di nuovo o solo esporre meglio ciò che c’è già?

- **Frontend**
  - quale componente leggerà questi dati?,
  - come arrivano (props, context, chiamate a servizi)?,
  - bisogna aggiornare `types.ts`?

- **Documentazione**
  - c’è già un manifesto/piano che descrive questa parte? (es. PLAN_VISUAL_CORTEX_RESTORE per CORTEX/TENSOR/PIPELINE),
  - se sì, cosa richiede esattamente?,
  - se no, dove ha più senso aggiungere una descrizione?

### 4.4 Affina (design atomico)

Spezza il lavoro in incrementi piccoli e committabili:

- **Esempio (collegare DidacticLayer a result.didactic.timeline)**
  1. Backend: estendere se necessario `OmegaKernel._generate_dsl_trace(...)` o la struttura di `result.didactic.timeline`.
  2. Server/API: assicurarsi che `/api/state` o `/api/intent` restituiscano la timeline.
  3. Frontend: adattare `OmegaCockpit` per leggere la timeline reale e passarla a `DidacticLayer` (invece di usare solo `omegaDSL`).
  4. Doc: aggiornare una breve sezione in `PLAN_VISUAL_CORTEX_RESTORE.md` o `SEMANTIC_KERNEL_MANIFEST.md` per dire che la timeline ora è allineata al kernel reale.

### 4.5 Registra (chiusura del ciclo)

A fine task:

- aggiorna `DOC_DEV/AGENT_AWARENESS.md` con:
  - cosa è stato fatto,
  - che impatto ha sul sistema,
  - dove toccare per estendere ulteriormente.
- se il cambiamento è strutturale/stabile, aggiungi una voce di versione in `SEMANTIC_KERNEL_MANIFEST.md`.
- se hai creato un nuovo pattern UI o una nuova metafora visiva, aggiorna `UI_DESIGN_VADEMECUM.md`.

---

## 5. Allineare ciò che la UI mostra con ciò che il kernel fa

Un principio cardine di questo progetto: **la UI non è skin, è fisica resa visibile**.

### 5.1 Visual Cortex ↔ lattice

- `OmegaKernel._generate_lattice_data(R)` produce un array di nodi con:
  - `id`, `x`, `y`, `spin`, `stability`, `assetTicker`…
- `VisualCortex` dovrebbe rappresentare concettualmente questo stato:
  - zone di alta coerenza vs caos,
  - transizioni di fase (PERTURBATION → ANNEALING → CRYSTALLIZED),
  - metrica (gravity) come parametri di dinamica.

Se cambi il modo in cui il kernel costruisce il lattice, verifica che la UI continui a rappresentarlo correttamente o che venga aggiornata di conseguenza.

### 5.2 Didactic Layer ↔ DSL trace

- `OmegaKernel._generate_dsl_trace(...)` costruisce una sequenza di step (PROBLEM, VARIABLES, CONSTRAINTS, ENERGY, HARDWARE, OUTPUT) con dettagli testuali.
- `DidacticLayer` presenta una timeline + rosetta per spiegare “come ha pensato” il sistema.

Oggi la DSL usata in UI è generata da `omegaDSL.ts` (client-side).  
Un’evoluzione naturale è collegare questa vista al `result.didactic.timeline` del backend.

Quando lo farai, ricordati di:

1. aggiornare le **tipizzazioni** (`types.ts`) per riflettere la struttura reale del payload,
2. aggiornare i **documenti UX** per dire che la Didactic UI ora è guidata dal kernel (non più solo da mock locali),
3. testare manualmente il flusso CORTEX → TENSOR → PIPELINE come da `PLAN_VISUAL_CORTEX_RESTORE.md`.

---

## 6. Relazione con MMS / OCC (meta‑livello)

Dietro al modo in cui lavori c’è un meta‑sistema descritto in:
- `DOC_DEV/MMS_kernel/MMS_Master.txt` (MMS vΦ.1),
- `DOC_DEV/MMS_kernel/Orchestratore_Cercatore_Costruttore_OCC_v1_0.txt` (OCC v1.0).

In breve:
- **MMS vΦ.1** definisce un ciclo operativo generale:
  - ResonanceInit → ScanIntent → RouteSelect → MiniPlan → ExecuteCluster → ValidateStream → CollapseField → InjectKLI.
- **OCC** è il pattern Orchestratore‑Cercatore‑Costruttore che:
  - analizza la richiesta,
  - progetta una struttura,
  - cerca e valuta conoscenza,
  - assembla un artefatto (prompt/codice/doc),
  - lo verifica criticamente.

Il workflow che usi in questo onboarding (`Registra → Controlla → Comprendi → Affina → Registra` + rituali tecnici) è la **traduzione pratica** di quei cicli a livello di repo:

- **Registra** ↔ ScanIntent / ResonanceInit  
  (capire l’intento reale e il contesto prima di toccare il codice)

- **Controlla** ↔ RouteSelect / Stream‑Guard  
  (verificare cosa esiste già in codice/doc/UI, quali moduli sono coinvolti)

- **Comprendi** ↔ MiniPlan / ExecuteCluster  
  (disegnare il flusso end‑to‑end e scomporre in passi chiari)

- **Affina** ↔ ValidateStream / CollapseField  
  (iterare, potare rami incoerenti, stabilizzare il risultato)

- **Registra (di nuovo)** ↔ InjectKLI / Evoluzione autopoietica  
  (aggiornare AGENT_AWARENESS, manifesti, vademecum per incorporare ciò che hai imparato).

Non sei “solo” un coder: nel contesto di questa repo, sei l’istanza operativa dell’OCC/MMS che agisce sul codice e sui documenti invece che solo sui prompt.  
Tenere vivo e aggiornato questo file (e i manifesti) **fa parte del lavoro del kernel**, non è un extra.

---

## 7. Known gaps & legacy plans (mappa per il coder)

Questa sezione elenca i principali **scarti tra piani/blueprint e codice/UI attuali**. Non sono bug in sé, ma zone dove doc, codice e UI non raccontano ancora esattamente la stessa storia.

1. **Visual Cortex / TENSOR / PIPELINE**  
   - *Blueprint (PLAN_VISUAL_CORTEX_RESTORE)*: piano pensato per la vecchia UI vanilla (`index.html`, `style.css`, `app.js`, pulsanti `btnCortex/btnTensor/btnPipeline`), con tre viste CORTEX/TENSOR/PIPELINE e un endpoint `/api/state` ricco (`lattice`, `tensor_field`, `didactic`).  
   - *Stato reale (React)*:
     - CORTEX e TENSOR sono implementati in `OmegaCockpit` (`VisualCortex`, `MetricTensorVis`) usando `OmegaPhysicsEngine` JS.
     - PIPELINE non esiste ancora come terzo tab esplicito: la pipeline concettuale vive solo nel `DidacticLayer` (timeline DSL) ed è ancora basata su `omegaDSL` client-side.
     - `/api/state` oggi in `server.py` restituisce solo `logic_density`, `experience`, `memory_size`, `taxonomy` (nessun `metrics`, nessun `didactic`), quindi non è ancora il canale ricco previsto dal piano.
   - *Note legacy*: la cartella `Extropic_Integration/cockpit/legacy_frontend` contiene la vecchia UI vanilla (index.html + style.css + app.js) che implementava questo piano; oggi funge solo da riferimento storico di pattern (layout tri-colonna, tooltip, resizer, docs modal) e **non è più il target operativo**: qualsiasi evoluzione futura va fatta sul cockpit React.

2. **API kernel ↔ cockpit (payload ricchi non sfruttati)**  
   - *Blueprint*: usare `/api/intent` e `/api/state` per consegnare al cockpit `metrics`, `lattice`, `tensor_field`, `didactic.timeline`, `gravity`, `entropy`, ecc.  
   - *Stato reale*:
     - `/api/intent` torna un `CycleResponse` con `manifesto`, `metrics`, `dipoles`, `didactic` completo (già prodotto da `SACS.process`/`OmegaKernel`).
     - `getAiAnalysis` in `geminiService.ts` usa solo `manifesto` → tutta la parte strutturata (`metrics`, `didactic`) è ignorata dalla UI.
     - `/api/state` non espone ancora `metrics`/`didactic` come si aspetterebbe `OmegaCockpit` (che tenta di leggere `data.metrics.temperature` e `data.didactic.gravity`).

3. **DSL / Didactic Layer**  
   - *Backend*: `OmegaKernel._generate_dsl_trace(...)` definisce una DSL interna con step `PROBLEM`, `VARIABLES`, `CONSTRAINTS`, `ENERGY`, `HARDWARE`, `OUTPUT` e rosetta associata.
   - *Frontend*: `omegaDSL.generateDslTrace(intent)` usa una DSL diversa (`SEMANTIC_PARSING`, `INTENT_EXTRACTION`, `METRIC_MAPPING`, `ANNEALING_INIT`, `CONVERGENCE_CHECK`) che il `DidacticLayer` anima.  
   - *Gap*: tassonomia e sorgente non coincidono ancora; il DidacticLayer non è ancora alimentato da `result.didactic.timeline` del kernel ma da una traccia sintetica locale.

4. **LLM naming / Gemini vs OpenRouter**  
   - *Blueprint storico*: integrazione diretta con Gemini (es. etichette "AI Projection Model: GEMINI-3-PRO").
   - *Stato reale*: lo stack LLM è **OpenRouter** (`openRouterService.ts`, `llm_inference.py`, `ModelCatalogModal.tsx`), con BYOK e catalogo modelli; `geminiService.ts` è oggi solo un thin proxy verso `/api/intent` (SACS) e non chiama più Gemini direttamente.  
   - *Gap*: alcune stringhe UI (es. in `AiExecutiveSummaryCard`) e naming residui (`geminiService`) riflettono ancora il vecchio mondo, non il gateway OpenRouter.

5. **Experimental Forge / Architect prompt**  
   - *System prompt Forge* (`forge_service.py`): parla di `CognitiveField` e mostra un import di esempio `from dnd_kernel.genesis import CognitiveField`, API che **non esiste** nel kernel attuale.
   - *Kernel reale*: il modulo `genesis.py` espone `Genesis` (`create_void`, `f_MIR_imprint`, `perturb_void`), più `MetricTensor`/`transfer_function` in `hardware_dynamics/combinatorial.py` e `UnifiedEquation` in `axioms.py`.  
   - *Gap*: il Forge ha una visione concettualmente coerente (campo cognitivo, NT → Duale), ma con nomi/API legacy non aggiornati all’attuale struttura Python.

6. **Financial Lab / Mission Control**  
   - *Blueprint*: la bancabilità e le metriche finance dovrebbero emergere (almeno concettualmente) dal comportamento del kernel e da dati economico‑finanziari reali.
   - *Stato reale*: 
     - tutte le card finance consumano dati da `mockDataService.ts` (`generateManagementData`, `generateCashFlowData`, `generateRatingData`, `generateFinanceData`), completamente random/mock;
     - `BankabilityRatingCard` visualizza un `RatingData` già calcolato, senza legame con SACS/Omega;
     - `AiExecutiveSummaryCard` mostra il testo `analysis` (da `/api/intent`), ma con label fissa "GEMINI-3-PRO";
     - `WidgetBuilderModal`/`DynamicWidgetCard` operano esclusivamente su questi dati mock, senza side‑effect sul kernel.
   - *Gap*: dominio finance è oggi una sandbox UI con dati sintetici; **non va interpretata come vista veritiera o kernel‑driven**, ma come laboratorio di pattern UX/narrativi in attesa di una decisione progettuale su dati reali e mapping kernel→finance. Inoltre esistono già bug tecnici noti: warning Recharts dovuti a container con width/height = -1 (grafici instabili in alcune fasi di layout) e crash del `WidgetBuilderModal` (Forge) per mismatch di props (`modules` non passato da `Dashboard`), che andranno sanati prima di riusare la Lab in contesti demo/produttivi.

7. **OpenRouter features avanzate**  
   - *Blueprint (openrouter_ istruzioni)*: previsto un ecosistema completo con:
     - Cost HUD (usage + prezzi per modello),
     - registry per‑nodo (nodi di sistema configurabili con `model_id`),
     - badge stato chiavi (server vs BYOK),
     - modal BYOK automatico quando la chiave di sistema è esaurita.
   - *Stato reale*: 
     - esistono già `/openrouter/status`, `/api/v1/openrouter/models`, BYOK nel `ModelCatalogModal` e un catalogo modelli funzionante;
     - non è ancora implementato: Cost HUD, registry nodi di sistema, apertura automatica del modal BYOK su errore `OPENROUTER_SYSTEM_KEY_EXHAUSTED`.
   - *Direzione di piano*: collegare progressivamente OpenRouter a **tutti i nodi AI** (quelli storicamente legati a Gemini e le nuove funzioni in sviluppo: Forge, pipeline di reasoning, Financial Lab, ecc.) tramite un **registry per‑nodo** (`task_name` → `model_id`) come descritto nelle guide OpenRouter, così che i modelli non siano più hard‑codati ma configurabili.

Questa sezione non è un backlog operativo, ma una **mappa di consapevolezza**: quando in futuro verranno assegnati task specifici (es. "collegare DidacticLayer al payload kernel" o "agganciare rating finance al kernel"), questi punti sono i candidati naturali da cui partire.

---

## 8. Futuri cluster di lavoro & playbook di consapevolezza mirata

Questa sezione usa la mappa attuale per suggerire **dove** ha senso intervenire in futuro e **quale consapevolezza minima** richiamare a seconda del tipo di lavoro.

### 8.1 Cluster di lavoro (backlog ad alto livello)

1. **Kernel ↔ Cockpit (API & Didactic)**  
   - Collegare il cockpit ai payload ricchi del kernel:
     - usare `result.didactic.timeline`, `lattice`, `tensor_field`, `gravity`, `entropy` nelle viste React (OmegaCockpit, DidacticLayer),
     - arricchire `/api/state` con `metrics`/`didactic` o usare direttamente `/api/intent` dove ha senso.
   - Allineare la DSL client (`omegaDSL`) con quella interna di `OmegaKernel._generate_dsl_trace`.

2. **Financial Lab stabilizzata**  
   - Riparare i bug tecnici:
     - mismatch di props `WidgetBuilderModal` ↔ `Dashboard` (prop `modules`, `onSave`, ecc. → oggi causa crash del Forge),
     - warning Recharts (width/height = -1) con contenitori/minWidth corretti.
   - Decidere se e come collegare la Financial Lab al kernel/dati reali (mapping kernel→finance) o dichiararla esplicitamente come sandbox interna.

3. **LLM / OpenRouter governance**  
   - Pulizia naming e UI:
     - aggiornare etichette “GEMINI-3-PRO” e nomi come `geminiService` per riflettere lo stack OpenRouter.
   - Implementare il **registry per-nodo** (task_name → model_id) e le feature avanzate previste (Cost HUD, badge chiavi, modal BYOK automatico).

4. **Experimental Forge & Architect**  
   - Aggiornare il system prompt del Forge eliminando riferimenti legacy (`CognitiveField`, import inesistenti) e riallineandolo alla struttura reale (Genesis, MetricTensor, UnifiedEquation).
   - Decidere il ruolo del Forge rispetto al cockpit: generatore di esperimenti THRML, di protocolli FINANCE, di pipeline DSL?

5. **UX & Design system omogeneo**  
   - Consolidare pattern comuni (modali, tooltip, resizer, HUD) tra cockpit React, tester ed eventuali nuove viste.
   - Integrare gradualmente un tema condiviso (quando SITEMAN/OMEGAMAN saranno definiti) partendo dal tema Deep Void.

6. **Legacy & cleanup**  
   - Vecchia UI vanilla (`legacy_frontend/`) da trattare come reference finché serve; in un futuro sprint si potrà:
     - rimuoverla o archiviarla fuori dalla main repo,
     - aggiornare piani e manifesti per riferirsi solo al cockpit React.

7. **Doc & narrazione multi‑pubblico**  
   - Documentazione tecnica per team Extropic (kernel, THRML, SACS, cockpit, OpenRouter).
   - Doc “narrativa” per visitatori/investitori (overview concettuale, casi d’uso, demo).
   - Assistente in‑app dedicato alla UI (guidato da un subset curato di: questo onboarding, manifesti kernel, whitepaper, UX brief) per spiegare/pilotare l’uso del sistema.

Questi cluster non sono task obbligatori, ma **direzioni naturali** in cui la repo può evolvere.

### 8.2 Playbook di consapevolezza mirata

Per evitare di dover rileggere tutto ogni volta, qui ci sono mini‑procedure di “bootstrap mentale” in base a cosa stai per fare.

1. **Se lavori su esperimenti / kernel (SACS, Omega, THRML)**  
   - Leggi / rivedi prima di toccare codice:
     - `Extropic_Integration/architect/sacs.py`, `dnd_kernel/omega.py`, `hardware_dynamics/*`,
     - `Extropic_Integration/docs/D-ND_Extropic_Technical_Whitepaper.md`, `D-ND_KERNEL_ARCHITECTURE_THRML_v1.md`,
     - test rilevanti in `tests/` (es. `test_omega_autological.py`, `test_hardware_dynamics.py`, `test_ising.py`).
   - Dopo le modifiche:
     - lancia i test pertinenti,
     - aggiorna AGENT_AWARENESS se hai cambiato qualcosa di significativo,
     - valuta se una nuova versione va annotata nel `SEMANTIC_KERNEL_MANIFEST.md`.

2. **Se lavori su nuove UI (cockpit React, nuove viste)**  
   - Richiama prima:
     - `docs/system/kernel/UX_INTEGRATION_BRIEF.md`, `UI_DESIGN_VADEMECUM.md`,
     - se la vista tocca il kernel: sezioni su OmegaCockpit/DidacticLayer in questo onboarding.
   - In codice:
     - entra da `App.tsx`/`Dashboard.tsx`/`OmegaCockpit.tsx` a seconda della zona,
     - verifica i tipi in `types.ts`.
   - Dopo le modifiche:
     - aggiorna il vademecum UI se hai introdotto un pattern nuovo (modale, resizer, tooltip, HUD),
     - annota in AGENT_AWARENESS cosa è cambiato lato UX.

3. **Se fai manutenzione / refactor**  
   - Prima:
     - scorri rapidamente `System_Coder_Onboarding.md` (sezione 3 per mappa file, 7 per known gaps),
     - controlla i test relativi alla zona che tocchi.
   - Durante:
     - spezza il refactor in passi piccoli e testabili,
     - mantieni i nomi coerenti con la tassonomia esistente.
   - Dopo:
     - esegui `pytest` + eventuali test FE manuali,
     - aggiorna AGENT_AWARENESS solo se il refactor ha impatto osservabile.

4. **Se fai commit / deploy su server**  
   - Prima del commit:
     - `pytest` dalla root,
     - eventuale `npm test`/controlli FE se presenti,
     - `pre-commit run --all-files` se attivo.
   - Prima del deploy:
     - verifica che gli endpoint usati dal FE (`/api/intent`, `/api/state`, `/api/docs`, `/openrouter/*`) rispondano come previsto,
     - aggiorna, se serve, le istruzioni operative (README o doc di deploy).
   - Dopo:
     - registra in AGENT_AWARENESS la versione effettivamente messa su server e come verificarla.

5. **Se tocchi documentazione/narrazione (Extropic tech, visitatori, investitori)**  
   - Per doc tecniche Extropic: parti da 
     - `SEMANTIC_KERNEL_MANIFEST.md`, questo onboarding, whitepaper, e aggiornali in modo coerente.
   - Per doc pubbliche/investitori:
     - mantieni il focus su concetti, casi d’uso e limiti attuali (es. Financial Lab come sandbox), senza promettere capacità non implementate.
   - Per l’assistente in‑app:
     - usa come sorgenti principali questo file, il manifesto kernel e i doc UX; evita di esporre internamente tutto il rumore di blueprint non implementati.

### 8.3 Appunti su aree non ancora esplorate a fondo

Queste note servono solo come promemoria per future sessioni di approfondimento mirato.

- **system_memory.json**  
  Non abbiamo ancora ispezionato nel dettaglio il contenuto della memoria persistente (numero di cicli, intent ricorrenti, tassonomia evoluta). Può valere la pena guardarla quando:
  - si progettano protocolli di reset/migrazione di stato,
  - si vogliono usare esempi reali di ciclo per doc/demo.

- **Performance e scalabilità**  
  Non sono stati fatti benchmark sistematici su dimensioni di lattice molto grandi o su carichi multi‑utente. Da esplorare se/quando il kernel passa a scenari di produzione intensiva.

- **Warning/minuzie FE**  
  A parte i warning Recharts già noti, non è stato ancora fatto un giro completo di QA frontend (dev server + console) alla ricerca di tutti i micro‑warning React/TypeScript. Questo controllo può accompagnare una fase di hardening UI.

- **Catalogo completo errori OpenRouter**  
  Sappiamo come funziona il flusso base e l’errore chiave (esaurimento chiave di sistema), ma non esiste ancora una mappa completa di tutti i codici/risposte errore possibili e della strategia di handling per ciascuno.

- **DOC_DEV storici**  
  Molti documenti in `DOC_DEV/` sono blueprint storici non ancora implementati; non li abbiamo letti tutti parola per parola. Quando un futuro task farà riferimento esplicito a uno di essi, converrà rileggerlo in profondità e aggiornarlo allo stato attuale.

---

Questa versione 0.1 è un **primo checkpoint di consapevolezza** per il System coder.  
Ogni volta che emergeranno nuove capacità stabili o nuovi pattern, questo file dovrà essere aggiornato insieme ai manifesti principali, in coerenza con il principio di autopoiesi del sistema.
