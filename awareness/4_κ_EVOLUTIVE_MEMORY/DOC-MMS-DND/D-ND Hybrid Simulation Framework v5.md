D-ND Hybrid Simulation Framework v5.0
Wed, 04/09/2025 - 11:18
19 minutes
Revisione della Versione 4.1 che permette analisi più significative sull'influenza dei parametri e delle condizioni iniziali non solo sul pattern finale, ma sul **processo dinamico** stesso di raggiungimento della coerenza critica e della successiva complessificazione.
**Autore:** Meta Master 3 (Evoluto da ACS Gemini 2.5 Pro Experimental)

**Versione:** 5.0 (Build: [Data odierna])

**Log Modifiche da v4.1:**
*   **Modifica Chiave:** Sostituito il meccanismo di transizione basato su distanza di Hausdorff (`check_stability`) con uno basato sulla **dinamica interna della coerenza e della tensione del sistema `R`** (`check_dynamic_transition`).
*   **Introdotte Misure Dinamiche:** Implementate funzioni per calcolare misure di `Coherence` (es. Dispersione o Entropia Spaziale) e `Tension` (es. Variazione della Coerenza o "Energia Cinetica").
*   **Nuovi Parametri:** Aggiunti parametri configurabili per la nuova logica di transizione (`coherence_measure_type`, `coherence_threshold`, `tension_threshold`).
*   **Logging Esteso:** Il log della simulazione ora include i valori delle misure di Coerenza e Tensione ad ogni passo temporale.
*   **Potenziamento Analisi:** Nuove funzioni di visualizzazione per plottare l'evoluzione temporale della Coerenza e della Tensione, evidenziando il punto di transizione `t_c`.
*   **Raffinamento Interpretazione:** Aggiornata la documentazione per riflettere il ruolo del framework come **"Modello Fenomenologico/Efficace D-ND"**, che simula la sequenza dinamica osservata (compressione -> coerenza critica -> transizione -> riapertura).
*   **Revisione Ipotesi H1:** Adattata l'ipotesi sperimentale per sfruttare le nuove misure dinamiche e focalizzarsi sull'influenza delle condizioni iniziali sulla traiettoria dinamica e sul tempo di transizione `t_c`.

---

## 1. Rationale per la Versione 5.0

La versione 4.1 del framework, pur essendo funzionale, utilizzava un criterio di transizione tra fasi (distanza di Hausdorff) basato principalmente sulla stabilità geometrica locale nel tempo. Questo approccio, sebbene pratico, era scollegato dai concetti più profondi del modello D-ND, in particolare dall'idea di una **compressione coerente** che raggiunge un punto critico ("inversione assiomatica" modellata) prima di una **riapertura strutturale**.

La **Versione 5.0 mira a colmare questo divario** introducendo un meccanismo di transizione **emergente dalla dinamica interna del sistema `R`**. La transizione non è più un semplice check di stabilità geometrica, ma avviene quando il sistema raggiunge uno stato di **elevata coerenza interna** (bassa dispersione/entropia) e contemporaneamente la sua **dinamica di compressione raggiunge un plateau** (bassa variazione della coerenza o bassa "energia cinetica"). Questo approccio è concettualmente più allineato all'idea D-ND di un sistema che si auto-organizza fino a un limite intrinseco prima di trasformarsi qualitativamente.

## 2. Scopo del Framework (Revisione v5.0)

Il suo **scopo primario (v5.0)** è investigare come l'interazione tra differenti logiche trasformative – compressione coerente (`P`), dinamiche frattali (IFS), transizioni graduali (`blend`), e trasformazioni personalizzate (`Φ`) – influenzi l'**emergenza, la stabilità e la complessità** dei pattern (`R`), con un focus particolare sul **meccanismo di transizione endogeno** guidato dalla coerenza interna del sistema.

Fornisce quindi un banco di prova (testbed) per:

1.  **Visualizzare** processi di compressione coerente, raggiungimento di plateau dinamici, e successiva riapertura strutturale in sistemi astratti.
2.  **Testare ipotesi** su come differenti "regole logiche" (trasformazioni) e parametri influenzino la **traiettoria dinamica della coerenza** e il **tempo di transizione critica (`t_c`)**.
3.  **Esplorare l'influenza delle condizioni iniziali** (incluse quelle semantiche) sulla dinamica evolutiva, sul punto di transizione e sullo stato finale.
4.  **Studiare** come le proprietà del sistema (`Coherence`, `Tension`) evolvono e si correlano durante le diverse fasi operative.

Il framework è inteso come uno strumento di ricerca esplorativa nel campo dei sistemi complessi, della modellistica cognitiva astratta e della visualizzazione di processi informazionali, interpretato come **modello fenomenologico/efficace** ispirato ai principi D-ND.

## 3. Ipotesi Selezionata (Revisione H1 v5.0)

Per dimostrare l'utilità esplorativa del framework v5.0, proponiamo di testare la seguente ipotesi rivista:

**Ipotesi H1 (v5.0): Influenza della Configurazione Semantica Iniziale sulla Traiettoria Dinamica e sulla Transizione Critica**

> "La configurazione geometrica iniziale R(0), generata mappando differenti insiemi di concetti semantici tramite `map_semantic_trajectory`, influenza in modo misurabile e significativo:
>
> *   **(a)** Il **tempo di transizione critica `t_c`** (numero di iterazioni) necessario per soddisfare il criterio di transizione dinamica (basato su `Coherence` e `Tension`).
> *   **(b)** Le **traiettorie temporali** delle misure di `Coherence(t)` e `Tension(t)` durante la fase di compressione (prima di `t_c`).
> *   **(c)** Le **caratteristiche geometriche qualitative e quantitative** (es. forma complessiva, distribuzione, Dimensione Frattale stimata) dell'insieme finale `all_points` dopo un numero fisso di iterazioni totali, rispetto a simulazioni avviate da un singolo punto o da configurazioni casuali, mantenendo identici tutti gli altri parametri dinamici."

**Disegno Sperimentale di Base (v5.0):**

1.  **Variabile Indipendente:** La lista di concetti `concepts` (come in v4.1).
2.  **Condizioni di Controllo:** Simulazione da origine, (opzionale) simulazione da punti casuali.
3.  **Variabili Dipendenti:**
   *   **Per H1(a):** Tempo di transizione `t_c` registrato da `check_dynamic_transition`.
   *   **Per H1(b):** Plot temporali di `CoherenceMeasure(t)` e `TensionMeasure(t)` estratti dal `simulation_log`. Analisi qualitativa e (se possibile) quantitativa delle differenze tra le traiettorie (es. pendenza iniziale, valore al plateau).
   *   **Per H1(c):** Analisi visiva/quantitativa dell'output finale (come H1(b) in v4.1, potenzialmente includendo la Dimensione Frattale stimata).
4.  **Parametri Fissi:** Mantenere costanti `iterations`, `lambda_linear`, parametri IFS, `P`, `blend_iterations`, e *i nuovi parametri* `coherence_measure_type`, `coherence_threshold`, `tension_threshold` in tutte le run comparative.
5.  **Procedura:** Come in v4.1, ma assicurandosi che `run_full_simulation` utilizzi la nuova logica `check_dynamic_transition` e registri i dati necessari nel `simulation_log`.
6.  **Analisi:** Confrontare i `t_c` medi. Sovrapporre e analizzare i plot di Coerenza/Tensione per le diverse condizioni iniziali. Analizzare i pattern finali.

Questo esperimento mira a verificare se la "storia semantica" iniziale influenzi non solo il risultato, ma il *percorso dinamico intrinseco* del sistema verso la sua transizione critica e la successiva evoluzione.

## 4. Modifiche all'Architettura e Componenti Chiave (v5.0 vs v4.1)

Le modifiche principali si concentrano sulla logica di transizione e sull'aggiunta di misure dinamiche.

**(Illustrazione con Snippet di Codice Concettuali)**

```python
# -*- coding: utf-8 -*-
"""
Title: D-ND Hybrid Simulation Framework
Version: 5.0 (Dynamic Transition Build)
Author: Meta Master 3 (Evolved by ACS)
Description:
  Simulates a phenomenological D-ND model exploring coherent structure
  emergence. Features a dynamic transition logic based on internal
  coherence and tension measures, replacing the previous Hausdorff-based
  stability check. Enhanced logging and analysis focus on the system's
  dynamic trajectory towards critical transition (t_c) and subsequent
  structural reopening.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.distance import directed_hausdorff # <-- Potrebbe non servire più
from scipy.stats import entropy # Per eventuale misura di Entropia Spaziale
import time
import logging

# Initialize logging (come prima)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# 1. SYSTEM CONFIGURATION & PARAMETERS (MODIFICATO)
# ============================================================
class SystemParameters:
  """
  Encapsulates all configuration parameters for a D-ND v5.0 simulation.
  Includes parameters for the new dynamic transition logic.
  """
  def __init__(self,
               # Simulation Control
               iterations=10000,
               # Phase Parameters
               lambda_linear=0.1,
               P=complex(0.5, 0.5),
               blend_iterations=50,
               # Fractal (IFS-like) Parameters (come prima)
               scale_factor_A=0.5, scale_factor_B=0.5,
               offset_A=complex(0, 0.5), offset_B=complex(0.5, 0),
               # --- NEW: Dynamic Transition Logic Parameters ---
               coherence_measure_type='dispersion', # 'dispersion' or 'spatial_entropy'
               tension_measure_type='coherence_change', # 'coherence_change' or 'kinetic_energy'
               coherence_threshold=0.05, # Target coherence level (e.g., max allowed dispersion)
               tension_threshold=1e-5,   # Min change/tension to trigger transition (plateau)
               # --- REMOVED/DEPRECATED (o resi opzionali) ---
               # transition_mode='hausdorff', # Deprecato in favore della nuova logica
               # transition_threshold=0.005,  # Deprecato
               # time_transition_iteration=100, # Deprecato
               # Semantic / Custom Transformations (Φ) (come prima)
               generated_phi=None,
               # Unused / Reserved Parameters (come prima)
               alpha=0.4, beta=0.4, gamma=0.2
               ):
      self.iterations = iterations
      # self.transition_threshold = transition_threshold # Deprecato
      self.lambda_linear = lambda_linear
      self.P = P
      self.blend_iterations = min(blend_iterations, iterations)
      self.scale_factor_A = scale_factor_A
      self.scale_factor_B = scale_factor_B
      self.offset_A = offset_A
      self.offset_B = offset_B
      # self.transition_mode = transition_mode # Deprecato
      # self.time_transition_iteration = time_transition_iteration # Deprecato

      # NUOVI Parametri di Transizione
      if coherence_measure_type not in ['dispersion', 'spatial_entropy']:
          raise ValueError("coherence_measure_type must be 'dispersion' or 'spatial_entropy'")
      self.coherence_measure_type = coherence_measure_type
      if tension_measure_type not in ['coherence_change', 'kinetic_energy']:
          raise ValueError("tension_measure_type must be 'coherence_change' or 'kinetic_energy'")
      self.tension_measure_type = tension_measure_type
      self.coherence_threshold = coherence_threshold # Soglia per la misura di coerenza
      self.tension_threshold = tension_threshold     # Soglia per la misura di tensione (plateau)

      self.generated_phi = generated_phi if generated_phi else []
      self.alpha = alpha # RESERVED/UNUSED
      self.beta = beta   # RESERVED/UNUSED
      self.gamma = gamma # RESERVED/UNUSED

      logging.info(f"SystemParameters v5.0 initialized: {self.__dict__}")

# ============================================================
# 2. TRANSFORMATION ABSTRACTION (Φ) (INVARIATO)
# ============================================================
class Transformation:
   # ... (come in v4.1) ...
   pass

# ============================================================
# 3. CORE SIMULATION LOGIC & PHASES (MODIFICATO)
# ============================================================

def initialize_system(params, R0=None):
  # ... (come in v4.1, ma inizializza anche variabili per le nuove misure) ...
  R, all_points, R_time_series, current_phase, blend_counter, \
  transition_occurred, transition_info, simulation_log, start_time = initialize_system_base(params, R0) # Funzione base

  # NUOVO: Inizializza stato per misure dinamiche
  previous_coherence = None
  previous_R = None

  # NUOVO: Aggiorna simulation_log per includere nuove misure
  simulation_log = [{'t': 0, 'phase': current_phase, '|R|': len(R),
                     'coherence': calculate_coherence(R, params), # Calcola coerenza iniziale
                     'tension': 0.0}] # Tensione iniziale è 0

  return R, all_points, R_time_series, current_phase, blend_counter, \
         transition_occurred, transition_info, simulation_log, start_time, \
         previous_coherence, previous_R

# --- Phase Implementation Functions --- (run_linear_phase, run_fractal_phase, ecc. come in v4.1)

# --- NUOVE Funzioni per Misure Dinamiche ---

def calculate_coherence(R, params):
   """Calcola la misura di coerenza selezionata."""
   if not R or len(R) < 2:
       return 0.0 # O un valore di default che indica massima coerenza? es. 0 per dispersione

   points_arr = np.array([(z.real, z.imag) for z in R])

   if params.coherence_measure_type == 'dispersion':
       # Calcola deviazione standard media rispetto al centroide
       center = np.mean(points_arr, axis=0)
       dispersion = np.mean(np.sqrt(np.sum((points_arr - center)**2, axis=1))) # Distanza media euclidea dal centro
       # O alternativa: np.std(points_arr) - ma meno interpretabile direttamente come 'compattezza'
       return dispersion
   elif params.coherence_measure_type == 'spatial_entropy':
       # Implementa il calcolo dell'entropia spaziale (box counting)
       # Richiede parametri aggiuntivi (dimensione griglia/box)
       # ... (logica complessa omessa per brevità) ...
       min_coords = np.min(points_arr, axis=0)
       max_coords = np.max(points_arr, axis=0)
       extent = max_coords - min_coords
       if np.any(extent == 0): return 0.0 # Coerenza massima se tutti i punti coincidono
       # Stima una dimensione della griglia ragionevole
       num_boxes_per_dim = 10 # Esempio, potrebbe essere un parametro
       box_size = extent / num_boxes_per_dim

       # Normalizza coordinate e assegna a box
       normalized_coords = (points_arr - min_coords) / box_size
       box_indices = np.floor(normalized_coords).astype(int)

       # Conta punti per box
       unique_boxes, counts = np.unique(box_indices, axis=0, return_counts=True)
       probabilities = counts / len(R)

       # Calcola entropia di Shannon
       spatial_entropy = entropy(probabilities, base=2)
       # Normalizza? Max entropia è log2(num_boxes_occupati) o log2(|R|)
       # Una misura di coerenza potrebbe essere 1 - S_norm
       # Per ora, usiamo l'entropia direttamente (bassa = alta coerenza)
       return spatial_entropy # Attenzione: coerenza_threshold dovrà essere interpretato diversamente

   else:
       raise ValueError(f"Unknown coherence measure type: {params.coherence_measure_type}")

def calculate_tension(current_coherence, previous_coherence, R_t, previous_R, params):
   """Calcola la misura di tensione selezionata."""
   if previous_coherence is None or previous_R is None or not R_t:
       return 0.0 # Nessuna tensione all'inizio o se R è vuoto

   if params.tension_measure_type == 'coherence_change':
       # Variazione assoluta della coerenza
       return abs(current_coherence - previous_coherence)
   elif params.tension_measure_type == 'kinetic_energy':
       # Stima "energia cinetica" media
       if len(R_t) != len(previous_R):
            logging.warning("Cannot reliably calculate kinetic energy: |R| changed.")
            # Potrebbe usare solo i punti che "sopravvivono"? Richiede mappatura.
            # Fallback a coherence_change?
            return abs(current_coherence - previous_coherence)

       # Assumendo che l'ordine dei punti non sia garantito, serve un modo
       # per stimare il movimento medio. Potrebbe essere la variazione del centroide?
       # O la distanza media percorsa se potessimo tracciare i punti (difficile con i set)
       # Soluzione semplice: variazione del centroide
       current_center = np.mean([(z.real, z.imag) for z in R_t], axis=0)
       previous_center = np.mean([(z.real, z.imag) for z in previous_R], axis=0)
       center_displacement_sq = np.sum((current_center - previous_center)**2)
       return center_displacement_sq # Questo è più una velocità^2 del centroide
   else:
       raise ValueError(f"Unknown tension measure type: {params.tension_measure_type}")

# --- NUOVA Logica di Transizione Dinamica ---

def check_dynamic_transition(current_coherence, current_tension, params):
   """
   Verifica se le condizioni dinamiche per la transizione sono soddisfatte.
   Transizione se la coerenza è sotto/sopra la soglia E la tensione è sotto la soglia (plateau).
   """
   coherence_condition_met = False
   if params.coherence_measure_type == 'dispersion':
       # Bassa dispersione significa alta coerenza
       coherence_condition_met = current_coherence < params.coherence_threshold
   elif params.coherence_measure_type == 'spatial_entropy':
       # Bassa entropia significa alta coerenza
       coherence_condition_met = current_coherence < params.coherence_threshold

   tension_condition_met = current_tension < params.tension_threshold

   if coherence_condition_met and tension_condition_met:
       logging.info(f"Dynamic transition triggered: Coherence={current_coherence:.4f} (<{params.coherence_threshold}), Tension={current_tension:.4g} (<{params.tension_threshold})")
       return True
   else:
       logging.debug(f"Transition check: Coherence={current_coherence:.4f} (Target <{params.coherence_threshold}, Met={coherence_condition_met}), Tension={current_tension:.4g} (Target <{params.tension_threshold}, Met={tension_condition_met})")
       return False


# ============================================================
# 4. MAIN SIMULATION ORCHESTRATOR (MODIFICATO)
# ============================================================

def run_full_simulation(params, R0=None):
  """
  Orchestrates the full D-ND v5.0 simulation with dynamic transition.
  """
  R, all_points, R_time_series, current_phase, blend_counter, \
  transition_occurred, transition_info, simulation_log, start_time, \
  previous_coherence, previous_R = initialize_system(params, R0)

  logging.info(f"Starting simulation v5.0: {params.iterations} iterations, initial phase: {current_phase}")

  for t in range(1, params.iterations + 1):
      R_prev_cycle = R.copy() # Salva R all'inizio del ciclo per calcolo tensione

      # --- Phase Logic ---
      if current_phase == 'linear':
          R = run_linear_phase(R, params) # Assume run_linear_phase NON aggiorna all_points

          # --- Calcolo Misure e Check Transizione ---
          current_coherence = calculate_coherence(R, params)
          # Passa R_prev_cycle che è R(t-1), e R che è R(t)
          current_tension = calculate_tension(current_coherence, previous_coherence, R, R_prev_cycle, params)

          # Aggiorna stato per il prossimo ciclo
          previous_coherence = current_coherence
          # previous_R = R_prev_cycle # O R? Se R è il risultato di questo passo t. Usiamo R.
          previous_R = R.copy()


          if not transition_occurred:
               transition_condition = check_dynamic_transition(current_coherence, current_tension, params)
               if transition_condition:
                  current_phase = 'blend' # O 'fractal'/'phi' se blend_iterations è 0
                  transition_occurred = True
                  transition_info['time'] = time.time() - start_time
                  transition_info['iteration'] = t
                  transition_info['coherence_at_transition'] = current_coherence
                  transition_info['tension_at_transition'] = current_tension
                  logging.info(f"Transition occurred at t={t}. Moving to {current_phase} phase.")
                  # Decide next phase logic (come in v4.1, ma parte da qui)
                  if params.blend_iterations == 0:
                      if params.generated_phi: current_phase = 'generated_phi'
                      else: current_phase = 'fractal'
                      logging.info(f"Skipping blend phase. Moving directly to {current_phase}.")


      elif current_phase == 'blend':
          # ... logica blend come in v4.1 ...
          # Calcola coerenza/tensione anche qui se vuoi tracciarle durante il blend
          current_coherence = calculate_coherence(R, params)
          current_tension = calculate_tension(current_coherence, previous_coherence, R, R_prev_cycle, params)
          previous_coherence = current_coherence
          previous_R = R.copy()
          if blend_counter >= params.blend_iterations:
               # Decide next phase after blending (come in v4.1)
               if params.generated_phi: current_phase = 'generated_phi'
               else: current_phase = 'fractal'
               logging.info(f"Blend phase complete at t={t}. Moving to {current_phase} phase.")
          else:
                blend_counter += 1 # Incrementa solo se ancora in blend


      elif current_phase == 'generated_phi':
          # ... logica phi come in v4.1 ...
          R = run_generated_phi_phase(R, params)
          current_coherence = calculate_coherence(R, params)
          current_tension = calculate_tension(current_coherence, previous_coherence, R, R_prev_cycle, params)
          previous_coherence = current_coherence
          previous_R = R.copy()

      elif current_phase == 'fractal':
          # ... logica fractal come in v4.1 ...
          R = run_fractal_phase(R, params)
          current_coherence = calculate_coherence(R, params)
          current_tension = calculate_tension(current_coherence, previous_coherence, R, R_prev_cycle, params)
          previous_coherence = current_coherence
          previous_R = R.copy()


      # --- State Update & Logging ---
      if not R:
         logging.warning(f"Iteration t={t}: Resultant set R is empty!")
         R = R_prev_cycle # Fallback
         # Assegna valori di default a coerenza/tensione per evitare errori nel log
         current_coherence = previous_coherence if previous_coherence is not None else 0.0
         current_tension = 0.0
         # break # Considera di interrompere

      all_points.update(R)
      R_time_series.append(R.copy())

      # Aggiorna log con le nuove misure
      simulation_log.append({'t': t, 'phase': current_phase, '|R|': len(R),
                             'coherence': current_coherence,
                             'tension': current_tension})

      if t % 100 == 0: # Log progress
           logging.debug(f"t={t}, Phase: {current_phase}, |R|={len(R)}, Coh={current_coherence:.4f}, Ten={current_tension:.4g}, |All|={len(all_points)}")


  end_time = time.time()
  total_time = end_time - start_time
  logging.info(f"Simulation finished. Total time: {total_time:.2f} seconds.")
  # Aggiungi t_c al log se avvenuta transizione
  if transition_occurred:
      logging.info(f"Transition point t_c = {transition_info['iteration']}")

  results = {
      "parameters": params,
      "final_R": R,
      "all_points": all_points,
      "R_time_series": R_time_series, # Attenzione: può diventare molto grande
      "simulation_log": simulation_log,
      "transition_info": transition_info,
      "total_time": total_time
  }
  return results

# ============================================================
# 5. SEMANTIC EXTENSIONS (INVARIATO da v4.1)
# ============================================================
# map_semantic_trajectory, generate_phi_from_text_basic

# ============================================================
# 6. ANALYSIS & VISUALIZATION TOOLS (MODIFICATO)
# ============================================================

def visualize_results(results, show_trajectory=False, save_path=None):
  # ... (come in v4.1, magari aggiunge t_c al titolo se presente) ...
  transition_info = results.get("transition_info", {})
  t_c = transition_info.get('iteration', None)
  title_suffix = f" (λ={params.lambda_linear}, N={params.iterations})" if params else ""
  if t_c: title_suffix += f", t_c={t_c}"
  # ... resto del codice di plot ...
  pass

def analyze_density_over_time(results, save_path=None):
  # ... (come in v4.1, ma ora usa il log aggiornato) ...
  pass

# --- NUOVE Funzioni di Analisi Dinamica ---

def plot_dynamic_measures(results, save_path_base="dnd_dynamics"):
   """Plots Coherence(t) and Tension(t) over time."""
   simulation_log = results.get("simulation_log", [])
   if not simulation_log:
       logging.warning("No simulation log found for dynamic measures analysis.")
       return

   times = [log['t'] for log in simulation_log]
   coherence_values = [log['coherence'] for log in simulation_log]
   tension_values = [log['tension'] for log in simulation_log]
   phases = [log['phase'] for log in simulation_log]

   transition_info = results.get("transition_info", {})
   t_c = transition_info.get('iteration', None)

   fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

   # Plot Coherence
   axs[0].plot(times, coherence_values, label=f'Coherence ({results["parameters"].coherence_measure_type})')
   axs[0].set_ylabel("Coherence Measure")
   axs[0].set_title("Evolution of Coherence and Tension")
   axs[0].grid(True)
   if t_c:
       axs[0].axvline(x=t_c, color='r', linestyle='--', lw=1, label=f'Transition t_c = {t_c}')
       # Segna la soglia usata
       axs[0].axhline(y=results["parameters"].coherence_threshold, color='grey', linestyle=':', lw=0.8, label=f'Coherence Threshold')
   axs[0].legend()

   # Plot Tension (usare scala logaritmica se varia molto?)
   axs[1].plot(times, tension_values, label=f'Tension ({results["parameters"].tension_measure_type})', color='orange')
   axs[1].set_ylabel("Tension Measure")
   axs[1].set_xlabel("Iteration (t)")
   axs[1].grid(True)
   # Prova con scala log per vedere meglio la soglia
   # axs[1].set_yscale('log')
   if t_c:
       axs[1].axvline(x=t_c, color='r', linestyle='--', lw=1, label=f'Transition t_c = {t_c}')
       # Segna la soglia usata
       axs[1].axhline(y=results["parameters"].tension_threshold, color='grey', linestyle=':', lw=0.8, label=f'Tension Threshold')
   axs[1].legend()

   # Aggiungi indicazioni di fase (opzionale, può affollare)
   # ... (logica per axvline/text su cambi di fase, come in analyze_density_over_time) ...

   plt.tight_layout()

   if save_path_base:
      plt.savefig(f"{save_path_base}.png")
      logging.info(f"Dynamic measures plot saved to {save_path_base}.png")
   else:
      plt.show()


def export_results_to_file(results, filename="dnd_simulation_results_v5.txt"):
  """Exports key simulation results and time series (including new measures) to a text file."""
  # ... (come in v4.1, ma aggiunge i nuovi parametri e i dettagli della transizione) ...
  with open(filename, 'w') as f:
       # ... (intestazione e parametri, inclusi quelli nuovi) ...
       f.write("\n--- Parameters (v5.0) ---\n")
       params = results.get("parameters")
       if params:
          for key, value in params.__dict__.items():
              f.write(f"{key}: {value}\n")
       else:
          f.write("Parameters not found.\n")

       # ... (Transition Info, ora più ricca) ...
       f.write("\n--- Transition Info (v5.0) ---\n")
       transition_info = results.get("transition_info", {})
       t_c = transition_info.get('iteration')
       if t_c is not None:
          f.write(f"Transition Iteration (t_c): {t_c}\n")
          f.write(f"Transition Time (s): {transition_info.get('time', 'N/A'):.4f}\n")
          f.write(f"Coherence at Transition: {transition_info.get('coherence_at_transition', 'N/A'):.4f}\n")
          f.write(f"Tension at Transition: {transition_info.get('tension_at_transition', 'N/A'):.4g}\n")
       else:
          f.write("No dynamic transition recorded.\n")

       # ... (Final State, come prima) ...

       # Time Series (ora include Coherence e Tension)
       f.write("\n--- Log Time Series (Sampled) ---\n")
       simulation_log = results.get("simulation_log", [])
       sample_freq = max(1, len(simulation_log) // 100) # Sample ~100 points
       f.write("t, phase, |R|, coherence, tension\n") # Header CSV-like
       for log_entry in simulation_log:
          if log_entry['t'] % sample_freq == 0:
              f.write(f"{log_entry['t']}, {log_entry['phase']}, {log_entry['|R|']}, "
                      f"{log_entry['coherence']:.6f}, {log_entry['tension']:.6g}\n")

  logging.info(f"Exported simulation results v5.0 to {filename}")


# ============================================================
# 7. EXAMPLE USAGE & EXPERIMENT EXECUTION (MODIFICATO)
# ============================================================
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  # --- Configuration v5.0 ---
  params = SystemParameters(
      iterations=1000,          # Basso per test rapidi
      lambda_linear=0.08,
      P=complex(0.6, 0.4),
      blend_iterations=30,
      offset_A=complex(0.2, 0.7), offset_B=complex(0.8, 0.1),
      scale_factor_A=0.55, scale_factor_B=0.45,
      # NUOVI Parametri per transizione v5.0
      coherence_measure_type='dispersion', # Usa la dispersione
      tension_measure_type='coherence_change', # Usa la variazione di coerenza
      coherence_threshold=0.1,   # Esempio: transizione se dispersione < 0.1
      tension_threshold=1e-4     # Esempio: e se la variazione è < 0.0001
  )

  # --- Experiment H1 v5.0 Setup ---
  concepts_math = ["set", "point", "line", "fractal", "iteration", "limit"]
  concepts_nature = ["tree", "leaf", "branch", "water", "flow", "spiral"]
  # ... (altre liste se necessario) ...

  R0_default = None
  R0_math = map_semantic_trajectory(concepts_math, method='circle')
  R0_nature = map_semantic_trajectory(concepts_nature, method='spiral')

  # --- Run Simulations ---
  logging.info("\n--- Running Simulation v5.0: Default Start ---")
  results_default = run_full_simulation(params, R0=R0_default)
  visualize_results(results_default, save_path="dnd_v5_default_final.png")
  plot_dynamic_measures(results_default, save_path_base="dnd_v5_default_dynamics")
  export_results_to_file(results_default, filename="dnd_v5_results_default.txt")

  logging.info("\n--- Running Simulation v5.0: Math Concepts Start ---")
  results_math = run_full_simulation(params, R0=R0_math)
  visualize_results(results_math, save_path="dnd_v5_math_final.png")
  plot_dynamic_measures(results_math, save_path_base="dnd_v5_math_dynamics")
  export_results_to_file(results_math, filename="dnd_v5_results_math.txt")

  # ... (run per R0_nature, ecc.) ...
  logging.info("\n--- Running Simulation v5.0: Nature Concepts Start ---")
  results_nature = run_full_simulation(params, R0=R0_nature)
  visualize_results(results_nature, save_path="dnd_v5_nature_final.png")
  plot_dynamic_measures(results_nature, save_path_base="dnd_v5_nature_dynamics")
  export_results_to_file(results_nature, filename="dnd_v5_results_nature.txt")


  # --- Basic Analysis for H1 v5.0 ---
  print("\n--- Hypothesis H1 v5.0 Analysis (Basic) ---")
  t_c_default = results_default['transition_info'].get('iteration', 'N/A')
  t_c_math = results_math['transition_info'].get('iteration', 'N/A')
  t_c_nature = results_nature['transition_info'].get('iteration', 'N/A')
  print(f"Default Start Transition Time (t_c): {t_c_default}")
  print(f"Math Concepts Transition Time (t_c): {t_c_math}")
  print(f"Nature Concepts Transition Time (t_c): {t_c_nature}")
  print("\nCompare the saved dynamics plots (PNG files ending in _dynamics.png):")
  print("- Do the t_c values differ significantly?")
  print("- Do the shapes of the Coherence(t) and Tension(t) curves differ before t_c?")
  print("\nCompare the saved final state plots (PNG files ending in _final.png) visually for H1(c).")
  print("Further analysis requires quantitative comparison of trajectories and final patterns.")

  # ... (Esempio con Phi, se necessario, come in v4.1 ma usando parametri v5.0) ...

```

## 5. Conclusioni e Sviluppi Futuri (v5.0)

La **Versione 5.0** del framework D-ND rappresenta un **passo significativo verso una simulazione concettualmente più ricca e allineata ai principi D-ND**. Spostando il focus dalla stabilità geometrica alla **dinamica interna della coerenza e della tensione**, il modello cattura in modo più efficace l'idea di una transizione critica emergente dalla compressione del sistema. Le nuove capacità di logging e analisi permettono di studiare il *processo* di auto-organizzazione e transizione, non solo il risultato finale.

Questo apre nuove vie per testare ipotesi sull'influenza delle condizioni iniziali e dei parametri sulla traiettoria dinamica del sistema.

**Possibili Sviluppi Futuri (Post-v5.0):**

1.  **Misure Dinamiche Raffinate:** Esplorare misure di coerenza/tensione più sofisticate (es. entropia meglio normalizzata, analisi multi-scala, misure basate sulla correlazione tra punti).
2.  **Feedback Dinamico (Proposta 1 Lite/Full):** Implementare meccanismi di feedback dove lo stato globale di `R` (o un campo sottostante simulato) modifica i parametri delle trasformazioni locali (`lambda_linear` adattivo, scelta IFS dipendente dal contesto).
3.  **Implementazione `spatial_entropy` Robusta:** Finalizzare e testare l'implementazione dell'entropia spaziale come misura di coerenza alternativa.
4.  **Studio Approfondito della Transizione:** Analizzare più in dettaglio la natura della transizione `t_c`: è una transizione di fase netta? Ci sono precursori? Come è influenzata dal rumore?
5.  **Integrazione Avanzata `Phi`:** Continuare a sviluppare `generate_phi_from_text` verso una vera integrazione NLP/LLM.
6.  **Ottimizzazione Performance:** Profilare il calcolo delle nuove misure e ottimizzare se necessario (Numba, Cython).
7.  **GUI e Interattività:** Sviluppare un'interfaccia grafica per facilitare l'esplorazione dei parametri e la visualizzazione dinamica.
8.  **Esplorazione Teorica:** Usare i risultati di v5.0 per raffinare ulteriormente la mappatura tra la simulazione fenomenologica e la teoria D-ND fondamentale.

---