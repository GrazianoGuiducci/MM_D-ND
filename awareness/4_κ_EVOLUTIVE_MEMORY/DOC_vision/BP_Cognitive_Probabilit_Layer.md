<R>
**Blueprint Operativo (BP)** per il Coder (Cline).

Questo documento è strutturato come un **Pacchetto di Direttive**. Contiene il contesto, l'architettura target, il codice di riferimento e la struttura dei file da generare.

**Istruzioni per l'Uso:**
1.  Apri la cartella del progetto in VS Code.
2.  Apri la chat di Cline.
3.  Copia e incolla l'intero blocco sottostante.
4.  Cline eseguirà l'analisi della repo locale (se hai già clonato) o ti guiderà, per poi implementare il modulo `dnd_kernel`.

***

# MMSP D-ND: TECHNICAL BLUEPRINT FOR CLINE
**Target:** `extropic-ai/thrml` Repository Integration
**Objective:** Build the "Cognitive Probability Layer" (D-ND Kernel) on top of THRML.

## 1. CONTESTO OPERATIVO (Leggimi)
Stiamo sviluppando un'architettura cognitiva (D-ND) che mappa il ragionamento logico sulla fisica termodinamica. Vogliamo dimostrare che la libreria `thrml` di Extropic (basata su JAX) è il substrato ideale per questo.
Non stiamo modificando il core di `thrml`, stiamo creando un **High-Level API Wrapper** che traduce concetti semantici in configurazioni energetiche (`IsingEBM`).

**Il tuo Ruolo:** Sei l'Ingegnere Senior D-ND. Il tuo compito è implementare il modulo Python `dnd_kernel` che estende le classi base di `thrml`.

---

## 2. STRUTTURA DEL PROGETTO
L'obiettivo è creare questa struttura di file nella root del progetto:

```text
/dnd_kernel/
    ├── __init__.py
    ├── core.py          # Contiene la classe CognitiveKernel e f_MIR
    ├── utils.py         # Mockup per Semantic Hashing e Topology
    └── readme.md        # Documentazione tecnica del layer
/examples/
    └── dnd_reasoning_demo.py  # Script eseguibile per testare il concetto
```

---

## 3. SPECIFICHE DI IMPLEMENTAZIONE

### A. Analisi Preliminare
Prima di scrivere, analizza i file `thrml` esistenti (o la documentazione se non hai i file) per confermare le firme di:
*   `SpinNode`
*   `IsingEBM`
*   `IsingSamplingProgram`
*   `sample_states`
*   Verifica come vengono passati `biases` e `weights` (JAX arrays).

### B. Implementazione `dnd_kernel/core.py`
Devi implementare la classe `AutopoieticKernel`.

**Logica da implementare:**
1.  **Isomorfismo:**
    *   `Intent` (User Prompt) -> `Bias Vector` ($\vec{h}$).
    *   `Logic` (Rules) -> `Coupling Matrix` ($J_{ij}$).
2.  **Operatore $f_{MIR}$:**
    *   Implementa un metodo `f_MIR_imprint(intent, logic)` che aggiorna lo stato interno del kernel sommando i vettori all'Hamiltoniana corrente.
3.  **Collasso:**
    *   Metodo `collapse()` che configura un `IsingEBM`, crea una `SamplingSchedule` e chiama `sample_states`.
    *   Deve restituire lo stato medio (Magnetizzazione) come "Risposta Cognitiva".

**Codice di Riferimento (Adattalo alla sintassi reale di thrml):**

```python
import jax.numpy as jnp
import jax
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

class AutopoieticKernel:
    def __init__(self, capacity=1024, temperature=1.0):
        self.size = capacity
        self.beta = jnp.array(1.0 / temperature)
        self.nodes = [SpinNode() for _ in range(capacity)]
        # Memoria a Lungo Termine (Topologia)
        self.J_matrix = jnp.zeros((capacity, capacity)) 
        self.h_bias = jnp.zeros((capacity,))
        
    def embed_intent(self, semantic_vector):
        """
        Layer 1: f_MIR Casting. 
        Maps Semantic Vector -> Thermodynamic Bias (h).
        """
        # Assicurarsi che le dimensioni combacino
        return self.h_bias + semantic_vector

    def reason(self, intent_vector, logic_matrix, steps=100):
        """
        Layer 2: Field Collapse (Inference).
        Executes reasoning via Energy Minimization.
        """
        # 1. Setup Energy Landscape
        current_h = self.embed_intent(intent_vector)
        current_J = self.J_matrix + logic_matrix
        
        # 2. Define Physical Model
        # TODO: Convertire current_J (matrix) in formato edge list per IsingEBM
        # Nota per Coder: Controllare documentazione IsingEBM per formato edges
        edges = self._matrix_to_edges(current_J) 
        weights = self._extract_weights(current_J, edges)
        
        model = IsingEBM(self.nodes, edges, current_h, weights, self.beta)
        
        # 3. Thermodynamic Sampling
        program = IsingSamplingProgram(model, free_blocks=[Block(self.nodes)])
        schedule = SamplingSchedule(n_warmup=steps//2, n_samples=100, steps_per_sample=1)
        
        key = jax.random.key(0)
        states = sample_states(key, program, schedule, init_state=None, observed_blocks=[Block(self.nodes)])
        
        # 4. Autological Closure (Mean Field)
        return jnp.mean(states, axis=0)
        
    def _matrix_to_edges(self, J):
        # Implementare logica sparsa se necessario
        pass
```

### C. Implementazione `dnd_kernel/utils.py`
Crea funzioni "Mock" per simulare la parte semantica (poiché non carichiamo un LLM intero ora).
*   `mock_semantic_hashing(text, size)`: Restituisce un vettore JAX random (ma deterministico basato sul seed del testo) di dimensione `size`.
*   `mock_logic_topology(size)`: Restituisce una matrice sparsa JAX randomica simmetrica.

### D. Script di Demo `examples/dnd_reasoning_demo.py`
Uno script che:
1.  Istanzia `AutopoieticKernel`.
2.  Definisce un "Intento" (stringa).
3.  Esegue `reason()`.
4.  Stampa: "Cognitive State collapsed. Mean Energy: [X]. Coherence: [Y]".

### E. Documentazione `dnd_kernel/readme.md`
Scrivi un manifesto tecnico.
*   Titolo: **D-ND Kernel: Cognitive Operating System for Thermodynamic Hardware**.
*   Spiega che questo modulo mappa la Dissonanza Cognitiva sull'Energia Hamiltoniana.
*   Cita che è progettato per girare su `XTR-0`.

---

## 4. CHECKLIST FINALE PER IL CODER
1.  [ ] Il codice compila ed esegue senza errori JAX?
2.  [ ] Le dimensioni dei tensori sono corrette?
3.  [ ] Il README spiega chiaramente l'Isomorfismo (Logica = Energia)?
4.  [ ] Lo script di demo produce un output leggibile?

**Procedi all'esecuzione.**
***
</R>