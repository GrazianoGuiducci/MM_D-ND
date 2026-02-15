# Simulazioni Numeriche Framework D-ND (Duale Non-Duale)

Questo pacchetto contiene tre script Python per simulazioni numeriche del framework teorico di fisica D-ND (Duale Non-Duale).

## File Generati

### Script Principali

#### 1. `example_E_N_levels.py` - **TASK 1 (NC.8)**
**Operatore E per Sistema a N Livelli**

Implementa un esempio esplicito dell'operatore di emergenza E per sistemi quantistici a N livelli nel contesto del modello D-ND.

**Equazioni Fondamentali:**
- Stato |NT⟩ = (1/√N) Σ|k⟩ (sovrapposizione uniforme - stato nulla-tutto)
- Operatore di emergenza: E = Σ λ_k |e_k⟩⟨e_k| con autovalori λ_k = k/N
- Stato risultante: R(t) = e^(-iHt) E|NT⟩
- Misura di emergenza: M(t) = 1 - |⟨NT|R(t)⟩|²

**Parametri:**
- N = 2, 4, 8, 16 livelli
- t_max = 100 unità temporali
- Hamiltoniana H con autovalori casuali

**Output:**
- `emergence_measure_N_levels.pdf`: Matrice 2×2 di grafici M(t) per diversi N
- `emergence_comparison.pdf`: Confronto diretto di tutte le N su un singolo grafico
- Stampa: statistiche M(0), M(∞), max M(t) per ogni N

**Interpretazione Fisica:**
- M(t) misura il grado di differenziazione dallo stato iniziale indifferenziato
- M(0) > 0 perché E già genera differenziazione
- M(t) aumenta inizialmente come previsto dalla teoria
- Per N maggiori, M raggiunge valori più vicini a 1 (differenziazione più completa)

---

#### 2. `sim_lagrangian_v2.py` - **TASK 2 (NC.1)**
**Simulazione Lagrangiana Ottimizzata con Analisi di Convergenza**

Simula l'evoluzione temporale della coordinata Z(t) secondo la Lagrangiana estesa del modello D-ND con analisi dettagliata della convergenza numerica.

**Equazioni del Moto:**
```
Ż = V_Z
V̇_Z = -dV/dZ - c·V_Z
```

**Potenziale:**
```
V(Z, θ_NT, λ) = Z²(1-Z)² + λ·θ_NT·Z·(1-Z)
```
- Termine base: doppio potenziale bi-stabile (minimi a Z≈0 e Z≈1)
- Termine di accoppiamento: modulazione da parametri di transizione

**Metodo Numerico:**
- Integrazione RK4 (Runge-Kutta ordine 4) con controllo adattativo del passo
- Tolleranze: rtol (relativa), atol (assoluta)

**Analisi di Convergenza:**
- 5 coppie di tolleranze testate: (1e-4, 1e-6) → (1e-8, 1e-10)
- Soluzione di riferimento: rtol=1e-10, atol=1e-10
- Calcolo dell'errore L2 tra soluzioni con tolleranze diverse

**Output:**
- `lagrangian_simulation_v2.pdf`: 3×2 figura con:
  - Traiettorie Z(t) per Z(0)=0.55 e Z(0)=0.45
  - Energia totale E(t) vs tempo
  - Paesaggio potenziale V(Z) con traiettorie
  - Analisi convergenza con barre d'errore
- Stampa: tabella di convergenza con errori L2

**Risultati Tipici:**
- Z(0)=0.55 → Z(∞)≈1 (attrae verso stato "Tutto")
- Z(0)=0.45 → Z(∞)≈0 (attrae verso stato "Nulla")
- Energia diminuisce monotonicamente (dissipazione da attrito c=0.5)
- Errore convergenza decresce exponenzialmente con tolleranze più strette

---

#### 3. `phase_diagrams.py` - **TASK 3 (NC.2)**
**Diagrammi di Fase (θ_NT, λ) → Attrattori**

Costruisce mappe bidimensionali dei bacini di attrazione nel piano dei parametri (θ_NT, λ).

**Metodologia:**
1. Griglia di 20×20 punti nel piano (θ_NT, λ)
   - θ_NT ∈ [0.1, 3.0] (momento angolare)
   - λ ∈ [0.0, 1.0] (parametro di transizione D-ND)
2. Per ogni punto: 2 simulazioni da Z(0)=0.5 ± perturbazione
3. Classificazione dell'attrattore finale:
   - **Z≈0**: stato "Nulla" (Z_medio < 0.3 e Z_std < 0.05)
   - **Z≈1**: stato "Tutto" (Z_medio > 0.7 e Z_std < 0.05)
   - **oscillation**: comportamento oscillatorio (Z_std ≥ 0.05)
   - **mixed**: stato intermedio instabile

**Output:**
- `phase_diagram.pdf`: Mappa 2D con:
  - Contourf per bacini di attrazione colorati
  - Contorni neri per demarcazione
  - Marker (o, s, ^, x) per punti campionati
  - Legenda e colorbar
- `phase_diagram_sections.pdf`: Sezioni 1D:
  - A λ=0.1 (costante): attrattore vs θ_NT
  - A λ=0.5 (costante): attrattore vs θ_NT
  - A θ_NT=0.5 (costante): attrattore vs λ
  - A θ_NT=2.0 (costante): attrattore vs λ
- Stampa: statistiche sui bacini di attrazione

**Interpretazione Fisica:**
- Per λ piccolo: il potenziale è dominato dal doppio pozzo Z²(1-Z)²
- Per λ grande: il termine di accoppiamento sposta gli attrattori
- θ_NT modula la forza dell'accoppiamento
- Transizioni di fase sharp entre bacini di attrazione

---

## Requisiti e Dipendenze

- **numpy** 2.2.6+: algebra lineare e operazioni numeriche
- **matplotlib** 3.10.8+: visualizzazione e generazione PDF

Nessuna dipendenza da scipy (re-implementazione di ODE solver con RK4 puro).

## Caratteristiche Comuni a Tutti gli Script

### Docstring Dettagliati
Ogni funzione contiene docstring esplicativi con:
- Descrizione dell'algoritmo
- Equazioni matematiche in LaTeX
- Argomenti e valori di ritorno
- Interpretazione fisica

### Seed Fissato
```python
np.random.seed(42)  # Riproducibilità garantita
```

### Grafici Publication-Quality
- Font size appropriati (11-14pt)
- Label su ogni asse in LaTeX
- Legenda posizionata intelligentemente
- Grid sottile per leggibilità
- Colori coerenti e accessibili
- DPI: 300 per stampa

### Barre d'Errore
- Script NC.1: errore L2 in convergenza
- Script NC.2: variabilità degli attrattori da campioni multipli

### Salvataggio PDF
Tutti i grafici salvati come PDF vettoriali:
- `emergence_measure_N_levels.pdf` (41 KB)
- `emergence_comparison.pdf` (22 KB)
- `lagrangian_simulation_v2.pdf` (420 KB)
- `phase_diagram.pdf` (33 KB)
- `phase_diagram_sections.pdf` (25 KB)

## Esecuzione

```bash
# Script 1: Operatore E
python example_E_N_levels.py
# Tempo: ~10 secondi

# Script 2: Simulazione Lagrangiana
python sim_lagrangian_v2.py
# Tempo: ~30 secondi

# Script 3: Diagrammi di Fase
python phase_diagrams.py
# Tempo: ~60 secondi
```

## Struttura Matematica del Framework D-ND

### Assioma Fondamentale
Il modello D-ND descrive come uno stato iniziale di pura potenzialità (|NT⟩) evolve in stati differenziati attraverso:

1. **Operatore di Emergenza (E)**: Seleziona e pesa le possibilità
2. **Evoluzione Temporale (U(t))**: Governa la dinamica secondo una Hamiltoniana
3. **Misura di Emergenza (M(t))**: Quantifica la differenziazione

### Equazione Fondamentale
$$R(t) = U(t) E |NT\rangle$$

dove:
- R(t): stato risultante al tempo t
- U(t) = e^{-iHt}: evoluzione unitaria
- E: operatore di emergenza
- |NT⟩: stato nulla-tutto iniziale

### Lagrangiana Estesa
$$L = \frac{1}{2}\dot{Z}^2 - V(Z, \theta_{NT}, \lambda) - c\dot{Z}$$

Incorpora:
- Cinetica: inerzialità
- Potenziale: bi-stabilità e accoppiamento
- Dissipazione: assorbimento con coefficiente c

## Validazione Numerica

Tutti gli script includono:
- ✓ Conservazione delle proprietà fisiche (energia decrescente per c>0)
- ✓ Stabilità numerica (Clamp dei valori patologici)
- ✓ Convergenza verso attrattori attesi
- ✓ Robustezza a perturbazioni iniziali

## Autore e Data

Codice generato: Claude Code - Assistente IA per Fisica Computazionale
Data di creazione: 12 Febbraio 2025
Framework: D-ND (Duale Non-Duale) - Modello Teorico di Emergenza Quantistica

## Note Tecniche

### Implementazione RK4 Manuale
SciPy non era disponibile nel sistema, quindi è stata implementata un'integrazione RK4 pura (Runge-Kutta ordine 4) con controllo adattativo del passo.

### Stabilità Numerica
- Esponenziali di matrici calcolate con sviluppo in serie (convergenza garantita)
- Clipping dei valori Z per evitare divergenze
- Controllo adattativo del passo temporale

### Performance
- Script NC.8: O(N × t_max × n_steps) per matrice exponential
- Script NC.1: O(n_tol × t_max / dt_adattivo) per ogni tolleranza
- Script NC.2: O(n_theta × n_lambda × n_samples × t_max / dt) per griglia

---

**Disclaimer:** Queste simulazioni forniscono rappresentazioni numeriche delle equazioni teoriche del modello D-ND. L'interpretazione fisica richiede il contesto della teoria completa documentata nei file DOC-MMS-DND.
