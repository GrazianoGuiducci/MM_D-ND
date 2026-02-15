D-ND Hybrid Simulation Framework
Wed, 04/09/2025 - 22:48
21 minutes
Simula modello fenomenologico/efficace D-ND esplorando l'emergenza di strutture coerenti. Introduce logica di transizione dinamica basato su misure interne di coerenza e tensione, sostituendo il precedente controllo di stabilità basato su distanza di Hausdorff.

"""

Titolo: D-ND Hybrid Simulation Framework

Versione: 5.0 (Dynamic Transition Build)

Autore: Meta Master 3 (Evoluto da ACS Gemini 2.5 Pro Experimental) - Implementazione: Gemini

Descrizione:

  Simula modello fenomenologico/efficace D-ND esplorando l'emergenza

  di strutture coerenti. Introduce logica di transizione dinamica basata

  su misure interne di coerenza e tensione, sostituendo il precedente

  controllo di stabilità basato su distanza di Hausdorff.

  Logging e analisi potenziati si focalizzano sulla traiettoria dinamica

  del sistema verso la transizione critica (t_c) e la successiva

  riapertura strutturale.

"""

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import entropy # Per misura di Entropia Spaziale

import time

import logging

import os # Per creare cartelle per i risultati

 

# Inizializzazione del logging

# Configurazione base, può essere personalizzata esternamente se necessario

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

 

# ============================================================

# 1. CONFIGURAZIONE DI SISTEMA E PARAMETRI (MODIFICATO v5.0)

# ============================================================

class SystemParameters:

    """

    Incapsula tutti i parametri di configurazione per simulazione D-ND v5.0.

    Include parametri per la nuova logica di transizione dinamica.

    """

    def __init__(self,

                 # Controllo Simulazione

                 iterations=10000,

                 # Parametri di Fase

                 lambda_linear=0.1, # Coefficiente per la fase lineare (compressione)

                 P=complex(0.5, 0.5), # Punto P per la trasformazione lineare

                 blend_iterations=50, # Numero di iterazioni per la fase di blend

                 # Parametri Frattali (tipo IFS)

                 scale_factor_A=0.5, scale_factor_B=0.5, # Fattori di scala

                 offset_A=complex(0, 0.5), offset_B=complex(0.5, 0), # Offset complessi

                 # --- NUOVO: Parametri Logica di Transizione Dinamica ---

                 coherence_measure_type='dispersion', # 'dispersion' o 'spatial_entropy'

                 tension_measure_type='coherence_change', # 'coherence_change' o 'kinetic_energy'

                 coherence_threshold=0.05, # Soglia di coerenza (es. dispersione massima)

                 tension_threshold=1e-5,   # Soglia di tensione (es. variazione minima per plateau)

                 # Parametri opzionali per misure specifiche

                 spatial_entropy_bins=10, # Numero di bin per dimensione per entropia spaziale

                 # Semantica / Trasformazioni Custom (Φ)

                 generated_phi=None, # Lista di oggetti Transformation

                 # Parametri Non Usati / Riservati (ereditati o per futuro)

                 alpha=0.4, beta=0.4, gamma=0.2

                 ):

        """Inizializza i parametri della simulazione."""

        self.iterations = iterations

        self.lambda_linear = lambda_linear

        self.P = P

        # Assicura che blend_iterations non superi iterations totali

        self.blend_iterations = min(blend_iterations, iterations) if blend_iterations > 0 else 0

 

        # Parametri IFS

        self.scale_factor_A = scale_factor_A

        self.scale_factor_B = scale_factor_B

        self.offset_A = offset_A

        self.offset_B = offset_B

 

        # Validazione e assegnazione parametri di transizione v5.0

        if coherence_measure_type not in ['dispersion', 'spatial_entropy']:

            raise ValueError("coherence_measure_type deve essere 'dispersion' o 'spatial_entropy'")

        self.coherence_measure_type = coherence_measure_type

 

        if tension_measure_type not in ['coherence_change', 'kinetic_energy']:

            raise ValueError("tension_measure_type deve essere 'coherence_change' o 'kinetic_energy'")

        self.tension_measure_type = tension_measure_type

 

        self.coherence_threshold = coherence_threshold # Soglia per la misura di coerenza

        self.tension_threshold = tension_threshold     # Soglia per la misura di tensione (plateau)

        self.spatial_entropy_bins = spatial_entropy_bins # Parametro per entropia spaziale

 

        # Trasformazioni custom

        self.generated_phi = generated_phi if generated_phi else []

 

        # Parametri riservati

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

 

        logging.info(f"SystemParameters v5.0 inizializzati.")

        # Log dettagliato dei parametri a livello DEBUG

        logging.debug(f"Parametri Dettagliati: {self.__dict__}")

 

# ============================================================

# 2. ASTRAZIONE TRASFORMAZIONE (Φ) (Placeholder)

# ============================================================

class Transformation:

    """

    Classe base (o interfaccia) per trasformazioni personalizzate (Φ).

    La logica specifica dipenderà dall'implementazione di v4.1 o da requisiti futuri.

    """

    def __init__(self, name="DefaultTransformation"):

        self.name = name

        logging.debug(f"Inizializzata Transformation: {self.name}")

 

    def apply(self, point_set):

        """

        Applica la trasformazione all'insieme di punti.

        Deve essere implementata dalle sottoclassi.

        """

        logging.warning(f"Metodo apply non implementato per {self.name}. Ritorna l'insieme originale.")

        return point_set

 

    def __str__(self):

        return f"Transformation({self.name})"

 

# ============================================================

# 3. LOGICA DI SIMULAZIONE CORE E FASI (MODIFICATO v5.0)

# ============================================================

 

# --- Funzioni Ausiliarie di Inizializzazione ---

def initialize_system_base(params, R0=None):

    """Inizializzazione di base comune a diverse versioni."""

    if R0 is None:

        # Inizia dall'origine se R0 non è fornito

        R = {complex(0, 0)}

        logging.info("Inizializzazione dall'origine (0,0).")

    elif isinstance(R0, set) and all(isinstance(p, complex) for p in R0):

        R = R0.copy()

        logging.info(f"Inizializzazione con R0 fornito, dimensione: {len(R)} punti.")

    else:

        logging.error("R0 fornito non è valido (deve essere set di numeri complessi). Inizializzazione dall'origine.")

        R = {complex(0, 0)}

 

    all_points = R.copy() # Insieme di tutti i punti generati

    R_time_series = [R.copy()] # Storico degli insiemi R (attenzione alla memoria)

    current_phase = 'linear' # Fase iniziale

    blend_counter = 0 # Contatore per la fase di blend

    transition_occurred = False # Flag per avvenuta transizione

    transition_info = { # Dizionario per informazioni sulla transizione

        'occurred': False,

        'iteration': None,

        'time': None,

        'coherence_at_transition': None,

        'tension_at_transition': None

    }

    simulation_log = [] # Log dettagliato per analisi dinamica

    start_time = time.time() # Tempo di inizio simulazione

 

    return R, all_points, R_time_series, current_phase, blend_counter, \

           transition_occurred, transition_info, simulation_log, start_time

 

def initialize_system(params, R0=None):

    """Inizializza il sistema per la v5.0, includendo stato per misure dinamiche."""

    R, all_points, R_time_series, current_phase, blend_counter, \

    transition_occurred, transition_info, simulation_log, start_time = initialize_system_base(params, R0)

 

    # NUOVO v5.0: Inizializza stato per misure dinamiche

    previous_coherence = None

    previous_R = None # R(t-1)

 

    # Calcola coerenza iniziale e aggiorna log

    # Gestisce il caso di R vuoto o con un solo punto

    initial_coherence = calculate_coherence(R, params) if len(R) > 0 else 0.0

    simulation_log = [{'t': 0, 'phase': current_phase, '|R|': len(R),

                       'coherence': initial_coherence,

                       'tension': 0.0}] # Tensione iniziale è 0

 

    logging.info(f"Sistema v5.0 inizializzato. Coerenza iniziale: {initial_coherence:.4f}")

 

    return R, all_points, R_time_series, current_phase, blend_counter, \

           transition_occurred, transition_info, simulation_log, start_time, \

           previous_coherence, previous_R

 

# --- Implementazione delle Fasi ---

# Queste funzioni rappresentano la logica applicata in ciascuna fase.

# La loro implementazione specifica può variare o richiedere dettagli da v4.1.

 

def run_linear_phase(R, params):

    """Applica la trasformazione lineare (compressione verso P)."""

    if not R: return set()

    # R(t+1) = R(t) + lambda * (P - R(t)) = (1-lambda)*R(t) + lambda*P

    lambda_lin = params.lambda_linear

    P = params.P

    # Applica la trasformazione a ogni punto nell'insieme R

    # Utilizza set comprehension per efficienza e per evitare duplicati

    next_R = {(1 - lambda_lin) * z + lambda_lin * P for z in R}

    return next_R

 

def run_fractal_phase(R, params):

    """Applica le trasformazioni tipo IFS (Iterated Function System)."""

    if not R: return set()

    next_R = set()

    # Applica entrambe le trasformazioni IFS a tutti i punti in R

    # Trasformazione A: z -> scale_A * z + offset_A

    # Trasformazione B: z -> scale_B * z + offset_B

    for z in R:

        next_R.add(params.scale_factor_A * z + params.offset_A)

        next_R.add(params.scale_factor_B * z + params.offset_B)

    return next_R

 

def run_blend_phase(R_linear, R_fractal, blend_progress):

    """

    Combina linearmente i risultati delle fasi lineare e frattale.

    Questa è un'interpretazione; la logica esatta di blend potrebbe differire.

    Assumiamo che R_linear e R_fractal siano gli output *potenziali* delle rispettive fasi.

    Il blend potrebbe operare sui punti stessi o sulle trasformazioni.

    Qui implementiamo un blend semplice basato sull'unione pesata (concettuale).

    Una logica più fedele a v4.1 potrebbe essere necessaria.

 

    Approccio alternativo: Applicare una trasformazione interpolata.

    T_blend(z) = (1-alpha)*T_linear(z) + alpha*T_fractal(z)

    Questo è complesso perché T_fractal produce due punti da uno.

 

    Approccio più semplice: Eseguire entrambe e prendere un sottoinsieme? O unire?

    Qui uniamo i risultati, che è l'interpretazione più probabile.

    """

    # Questa implementazione assume che R_linear e R_fractal siano già calcolati

    # all'interno del ciclo principale basandosi su R(t).

    # Il blend_progress (da 0 a 1) determina come combinarli.

    # Potrebbe non essere il modo corretto, dipende da v4.1.

    # Se il blend è un'interpolazione *della trasformazione*, la logica cambia.

 

    # Interpretazione: La fase di blend applica *entrambe* le logiche per N iterazioni.

    # Questo sembra più plausibile dal contesto di "riapertura graduale".

    # Quindi, questa funzione potrebbe non essere necessaria se la logica è nel ciclo principale.

 

    # Assumiamo per ora che la fase 'blend' nel ciclo principale applichi

    # una logica specifica per blend_iterations volte.

    # Per semplicità, replichiamo la logica frattale durante il blend,

    # assumendo che rappresenti la "riapertura".

    # Questo è un placeholder e probabilmente necessita revisione.

    logging.debug(f"Esecuzione fase Blend (Placeholder - usa logica Frattale)")

    # Ritorna l'unione dei due set, come interpretazione base

    # return R_linear.union(R_fractal)

    # Placeholder: usa la logica frattale come esempio di "riapertura"

    # Questo richiede che R sia passato a questa funzione.

    # return run_fractal_phase(R_linear, params) # Passa R della fase precedente

    pass # La logica di blend è gestita nel ciclo principale per ora.

 

 

def run_generated_phi_phase(R, params):

    """Applica le trasformazioni personalizzate definite in params.generated_phi."""

    if not R or not params.generated_phi:

        return R

    

    current_R = R.copy()

    # Applica sequenzialmente ogni trasformazione definita

    for transformation in params.generated_phi:

        if isinstance(transformation, Transformation):

            current_R = transformation.apply(current_R)

        else:

            logging.warning(f"Elemento in generated_phi non è oggetto Transformation: {transformation}")

            

    return current_R

 

# --- NUOVE Funzioni per Misure Dinamiche (v5.0) ---

 

def calculate_coherence(R, params):

    """Calcola la misura di coerenza selezionata."""

    num_points = len(R)

    # Se R è vuoto o ha un solo punto, la coerenza è massima (valore = 0 per dispersione/entropia)

    if num_points < 2:

        return 0.0

 

    # Converte l'insieme di complessi in array NumPy (N, 2)

    points_arr = np.array([(z.real, z.imag) for z in R])

 

    measure_type = params.coherence_measure_type

    coherence_value = 0.0

 

    try:

        if measure_type == 'dispersion':

            # Calcola deviazione standard media rispetto al centroide (distanza media)

            center = np.mean(points_arr, axis=0)

            # Distanza euclidea di ogni punto dal centro

            distances = np.sqrt(np.sum((points_arr - center)**2, axis=1))

            # Dispersione = distanza media dal centro

            coherence_value = np.mean(distances)

 

        elif measure_type == 'spatial_entropy':

            # Implementa calcolo dell'entropia spaziale (basato su box counting)

            min_coords = np.min(points_arr, axis=0)

            max_coords = np.max(points_arr, axis=0)

            extent = max_coords - min_coords

 

            # Se tutti i punti coincidono (extent è zero in una dimensione), entropia è 0 (max coerenza)

            # Aggiunge epsilon per evitare divisione per zero se extent è 0

            epsilon = 1e-9

            extent = np.maximum(extent, epsilon)

 

            num_boxes_per_dim = params.spatial_entropy_bins

            # Calcola la dimensione dei box (cella della griglia)

            box_size = extent / num_boxes_per_dim

 

            # Normalizza coordinate e assegna a indici di box (interi)

            # Aggiunge epsilon a points_arr per gestire punti sui bordi massimi

            normalized_coords = (points_arr + epsilon - min_coords) / box_size

            # Usa floor per assegnare a indici di box

            box_indices = np.floor(normalized_coords).astype(int)

            

            # Correggi indici che potrebbero finire nel bin N a causa di epsilon

            box_indices = np.minimum(box_indices, num_boxes_per_dim - 1)

 

 

            # Conta punti per box univoco

            # np.unique richiede che gli elementi siano tuple per righe multiple

            unique_boxes, counts = np.unique(box_indices, axis=0, return_counts=True)

 

            # Calcola le probabilità (frequenza relativa)

            probabilities = counts / num_points

 

            # Calcola entropia di Shannon (base 2)

            spatial_entropy = entropy(probabilities, base=2)

 

            # L'entropia è inversamente correlata alla coerenza.

            # Potremmo normalizzarla (es. diviso log2(num_points) o log2(num_boxes_occupati))

            # Per ora, usiamo l'entropia direttamente. Bassa entropia = alta coerenza.

            coherence_value = spatial_entropy

        else:

            # Questo non dovrebbe accadere grazie alla validazione in __init__

             raise ValueError(f"Tipo misura coerenza sconosciuto: {measure_type}")

 

    except Exception as e:

        logging.error(f"Errore nel calcolo della coerenza ({measure_type}): {e}")

        # Ritorna valore che indica bassa coerenza o errore?

        # Per dispersione, valore alto. Per entropia, valore alto.

        # Usiamo un valore molto grande per indicare problema.

        return np.inf 

 

    # Assicura che il risultato non sia NaN o Inf (può accadere con pochi punti o configurazioni degenerate)

    if not np.isfinite(coherence_value):

        logging.warning(f"Valore coerenza non finito ({coherence_value}) per {measure_type}. Sostituito con 0.0 (max coerenza). |R|={num_points}")

        return 0.0 # Ritorna massima coerenza in caso di problemi numerici

 

    return coherence_value

 

 

def calculate_tension(current_coherence, previous_coherence, R_t, previous_R, params):

    """Calcola la misura di tensione selezionata."""

    # Se non ci sono dati precedenti o R corrente è vuoto, la tensione è 0

    if previous_coherence is None or previous_R is None or not R_t:

        return 0.0

 

    measure_type = params.tension_measure_type

    tension_value = 0.0

 

    try:

        if measure_type == 'coherence_change':

            # Variazione assoluta della coerenza tra t-1 e t

            tension_value = abs(current_coherence - previous_coherence)

 

        elif measure_type == 'kinetic_energy':

            # Stima "energia cinetica" basata sullo spostamento quadratico medio

            # Questo richiede di poter confrontare R(t) e R(t-1).

            # Se il numero di punti cambia, il confronto diretto è difficile.

            

            num_current = len(R_t)

            num_previous = len(previous_R)

 

            if num_current == 0 or num_previous == 0:

                 return 0.0 # Tensione nulla se uno degli insiemi è vuoto

 

            # Converte in array NumPy

            points_t = np.array([(z.real, z.imag) for z in R_t])

            points_t_minus_1 = np.array([(z.real, z.imag) for z in previous_R])

 

            # Approccio 1: Variazione del centroide (semplice ma limitato)

            # current_center = np.mean(points_t, axis=0)

            # previous_center = np.mean(points_t_minus_1, axis=0)

            # center_displacement_sq = np.sum((current_center - previous_center)**2)

            # tension_value = center_displacement_sq # Rappresenta (velocità media)^2

 

            # Approccio 2: Distanza media quadratica tra i punti (richiede corrispondenza o assunzioni)

            # Se |R| è costante, potremmo assumere corrispondenza per indice (rischioso).

            # Se |R| cambia, potremmo usare la distanza di Hausdorff o simili? Complesso.

            

            # Fallback robusto se |R| cambia: usa 'coherence_change'

            if num_current != num_previous:

                logging.warning(f"|R| cambiato ({num_previous} -> {num_current}). Impossibile calcolare 'kinetic_energy' in modo affidabile. Uso 'coherence_change' come fallback.")

                tension_value = abs(current_coherence - previous_coherence)

            else:

                 # Se |R| è costante, calcola lo spostamento quadratico medio

                 # Assumendo che l'ordine sia irrilevante, calcoliamo la differenza quadratica media

                 # Questo non è fisicamente accurato come energia cinetica, ma misura la "vibrazione"

                 # Potrebbe essere meglio usare la variazione del centroide? O Hausdorff?

                 # Per ora, usiamo la variazione del centroide come implementato nello snippet originale.

                 current_center = np.mean(points_t, axis=0)

                 previous_center = np.mean(points_t_minus_1, axis=0)

                 center_displacement_sq = np.sum((current_center - previous_center)**2)

                 tension_value = center_displacement_sq

 

        else:

            raise ValueError(f"Tipo misura tensione sconosciuto: {measure_type}")

 

    except Exception as e:

        logging.error(f"Errore nel calcolo della tensione ({measure_type}): {e}")

        return 0.0 # Ritorna 0 in caso di errore

 

    # Assicura che il risultato sia finito

    if not np.isfinite(tension_value):

        logging.warning(f"Valore tensione non finito ({tension_value}) per {measure_type}. Sostituito con 0.0.")

        return 0.0

        

    return tension_value

 

# --- NUOVA Logica di Transizione Dinamica (v5.0) ---

def check_dynamic_transition(current_coherence, current_tension, params):

    """

    Verifica se le condizioni dinamiche per la transizione sono soddisfatte.

    Transizione = Alta Coerenza E Bassa Tensione (Plateau).

    """

    coherence_condition_met = False

    # La condizione dipende dal tipo di misura:

    # - Dispersione: Bassa dispersione = alta coerenza -> coherence < threshold

    # - Entropia Spaziale: Bassa entropia = alta coerenza -> coherence < threshold

    if params.coherence_measure_type in ['dispersion', 'spatial_entropy']:

        coherence_condition_met = current_coherence < params.coherence_threshold

    # Aggiungere altri tipi se necessario

 

    # La tensione deve essere bassa (indicando plateau o stabilità dinamica)

    tension_condition_met = current_tension < params.tension_threshold

 

    # La transizione avviene se entrambe le condizioni sono vere

    transition_triggered = coherence_condition_met and tension_condition_met

 

    if transition_triggered:

        logging.info(f"Transizione dinamica RILEVATA: Coerenza={current_coherence:.4f} (< {params.coherence_threshold}), Tensione={current_tension:.4g} (< {params.tension_threshold})")

        return True

    else:

        # Logga lo stato del check a livello DEBUG per non inondare l'output

        logging.debug(f"Check Transizione: Coerenza={current_coherence:.4f} (Soglia <{params.coherence_threshold}, Raggiunta={coherence_condition_met}), Tensione={current_tension:.4g} (Soglia <{params.tension_threshold}, Raggiunta={tension_condition_met})")

        return False

 

# ============================================================

# 4. ORCHESTRATORE PRINCIPALE SIMULAZIONE (MODIFICATO v5.0)

# ============================================================

def run_full_simulation(params, R0=None, run_id="sim"):

    """

    Orchestra la simulazione completa D-ND v5.0 con transizione dinamica.

    """

    # Inizializza stato del sistema e variabili per misure dinamiche

    R, all_points, R_time_series, current_phase, blend_counter, \

    transition_occurred, transition_info, simulation_log, start_time, \

    previous_coherence, previous_R = initialize_system(params, R0)

 

    logging.info(f"[{run_id}] Avvio simulazione v5.0: {params.iterations} iterazioni, fase iniziale: {current_phase}, |R0|={len(R)}")

 

    # Ciclo principale della simulazione

    for t in range(1, params.iterations + 1):

        # Salva R all'inizio del ciclo (R(t-1) per calcolo tensione)

        R_prev_cycle = R.copy() 

        

        # --- Logica di Fase ---

        next_R = set() # Insieme risultato di questo passo

        phase_executed = current_phase # Fase effettivamente eseguita in questo step

 

        if current_phase == 'linear':

            next_R = run_linear_phase(R, params)

            # Durante la fase lineare, controlla la transizione dinamica

            if not transition_occurred:

                 # Calcola misure DOPO aver applicato la trasformazione lineare

                 current_coherence = calculate_coherence(next_R, params)

                 current_tension = calculate_tension(current_coherence, previous_coherence, next_R, R_prev_cycle, params)

 

                 # Aggiorna stato per il prossimo ciclo (solo se in fase lineare)

                 previous_coherence = current_coherence

                 previous_R = next_R.copy() # Salva R(t) per il prossimo calcolo di tensione

 

                 # Esegui il check per la transizione

                 transition_condition = check_dynamic_transition(current_coherence, current_tension, params)

                 if transition_condition:

                    # Transizione avvenuta!

                    transition_occurred = True

                    transition_info['occurred'] = True

                    transition_info['time'] = time.time() - start_time

                    transition_info['iteration'] = t

                    transition_info['coherence_at_transition'] = current_coherence

                    transition_info['tension_at_transition'] = current_tension

                    

                    # Determina la fase successiva

                    if params.blend_iterations > 0:

                        current_phase = 'blend'

                        blend_counter = 0 # Resetta contatore blend

                    elif params.generated_phi:

                        current_phase = 'generated_phi'

                    else:

                        current_phase = 'fractal'

                    

                    logging.info(f"[{run_id}] Transizione avvenuta a t={t}. Passaggio a fase '{current_phase}'.")

            else:

                 # Se la transizione è già avvenuta, calcola comunque le misure per il log

                 # ma usa R(t) e R(t-1) della fase corrente (che non è più lineare)

                 # Questo richiede di calcolare le misure *dopo* l'esecuzione della fase

                 pass # Calcolo misure spostato dopo lo switch delle fasi

 

        elif current_phase == 'blend':

            # Logica di blend: potrebbe applicare entrambe o interpolare.

            # Placeholder: applica logica frattale per 'blend_iterations' volte.

            # Questo è probabilmente da rivedere basandosi su v4.1.

            # Assumiamo che 'blend' significhi applicare la logica della fase successiva

            # (frattale o phi) per un certo numero di iterazioni.

            

            # Determina quale logica applicare durante il blend (frattale o phi)

            blend_logic_phase = 'fractal'

            if params.generated_phi:

                 blend_logic_phase = 'generated_phi' # Assume che Phi sovrascriva frattale se presente

                 

            if blend_logic_phase == 'fractal':

                 next_R = run_fractal_phase(R, params)

            else: # 'generated_phi'

                 next_R = run_generated_phi_phase(R, params)

 

            blend_counter += 1

            if blend_counter >= params.blend_iterations:

                # Fine fase blend, passa alla fase successiva definitiva

                if params.generated_phi:

                    current_phase = 'generated_phi'

                else:

                    current_phase = 'fractal'

                logging.info(f"[{run_id}] Fase Blend completata a t={t}. Passaggio a fase '{current_phase}'.")

            # Altrimenti rimane in fase 'blend'

 

        elif current_phase == 'generated_phi':

            next_R = run_generated_phi_phase(R, params)

 

        elif current_phase == 'fractal':

            next_R = run_fractal_phase(R, params)

 

        # --- Aggiornamento Stato e Logging ---

        

        # Gestisce caso in cui next_R risulti vuoto (potrebbe accadere?)

        if not next_R:

            logging.warning(f"[{run_id}] Iterazione t={t}: L'insieme R risultante è vuoto! Fase={phase_executed}. Mantengo R precedente.")

            # Non aggiornare R, ma calcola misure su R precedente per consistenza log

            # O assegna valori di default? Assegniamo coerenza/tensione precedenti.

            current_coherence = previous_coherence if previous_coherence is not None else 0.0

            current_tension = 0.0 # Tensione è 0 se R non cambia

            # Non aggiornare R: R = R_prev_cycle # O semplicemente non aggiornare R

        else:

            # Aggiorna R per la prossima iterazione

             R = next_R.copy()

             # Calcola coerenza e tensione per il log (se non già fatto in fase lineare pre-transizione)

             if phase_executed != 'linear' or transition_occurred:

                 current_coherence = calculate_coherence(R, params)

                 # Nota: R_prev_cycle è R(t-1), R è R(t)

                 current_tension = calculate_tension(current_coherence, previous_coherence, R, R_prev_cycle, params)

                 # Aggiorna stato per il prossimo ciclo

                 previous_coherence = current_coherence

                 previous_R = R.copy()

 

 

        # Aggiorna l'insieme di tutti i punti

        all_points.update(R)

        # Aggiungi R corrente allo storico (opzionale, per analisi post)

        # Considerare di campionare per risparmiare memoria

        if t % 10 == 0: # Esempio: salva ogni 10 iterazioni

             R_time_series.append(R.copy())

 

        # Aggiorna il log della simulazione con le misure calcolate

        simulation_log.append({'t': t, 'phase': phase_executed, '|R|': len(R),

                               'coherence': current_coherence,

                               'tension': current_tension})

 

        # Log di progresso ogni N iterazioni

        if t % (params.iterations // 10 if params.iterations >= 10 else 1) == 0:

            logging.info(f"[{run_id}] t={t}/{params.iterations}, Fase: {phase_executed}, |R|={len(R)}, "

                         f"Coh={current_coherence:.4f}, Ten={current_tension:.4g}, |Tot|={len(all_points)}")

 

    # Fine Simulazione

    end_time = time.time()

    total_time = end_time - start_time

    logging.info(f"[{run_id}] Simulazione completata. Tempo totale: {total_time:.2f} secondi.")

 

    # Log finale del punto di transizione, se avvenuta

    if transition_occurred:

        logging.info(f"[{run_id}] Punto di transizione finale t_c = {transition_info['iteration']}")

    else:

        logging.info(f"[{run_id}] Nessuna transizione dinamica rilevata entro {params.iterations} iterazioni.")

 

    # Prepara il dizionario dei risultati

    results = {

        "run_id": run_id,

        "parameters": params,

        "final_R": R,

        "all_points": all_points,

        "R_time_series": R_time_series, # Attenzione: può essere molto grande

        "simulation_log": simulation_log, # Contiene coerenza e tensione nel tempo

        "transition_info": transition_info,

        "total_time_s": total_time

    }

    return results

 

# ============================================================

# 5. ESTENSIONI SEMANTICHE (Placeholders/Base da v4.1)

# ============================================================

# Queste funzioni dipendono fortemente dall'implementazione specifica di v4.1.

# Fornisco implementazioni di base o placeholder.

 

def map_semantic_trajectory(concepts, method='circle', scale=1.0):

    """

    Mappa lista di concetti in un insieme di punti R0 (numeri complessi).

    Placeholder basato su metodi semplici ('circle', 'line', 'random').

    """

    n = len(concepts)

    if n == 0: return {complex(0,0)} # Default a origine se non ci sono concetti

 

    points = set()

    if method == 'circle':

        # Dispone i punti su cerchio unitario (scalato)

        for i, concept in enumerate(concepts):

            angle = 2 * np.pi * i / n

            point = complex(np.cos(angle), np.sin(angle)) * scale

            points.add(point)

            logging.debug(f"Concetto '{concept}' mappato a {point:.2f} (metodo: {method})")

    elif method == 'line':

        # Dispone i punti su segmento di linea da -scale/2 a +scale/2

         for i, concept in enumerate(concepts):

             # Mappa i in [0, n-1] a [-scale/2, scale/2]

             pos = -scale/2 + (scale * i / (n-1)) if n > 1 else 0

             point = complex(pos, 0)

             points.add(point)

             logging.debug(f"Concetto '{concept}' mappato a {point:.2f} (metodo: {method})")

    elif method == 'spiral':

         # Dispone i punti a spirale

         a = scale / (2 * np.pi * n) # Controlla la spaziatura

         for i, concept in enumerate(concepts):

              angle = i * (2 * np.pi / np.sqrt(n)) # Angolo aumenta

              radius = a * angle # Raggio aumenta con angolo

              point = complex(radius * np.cos(angle), radius * np.sin(angle))

              points.add(point)

              logging.debug(f"Concetto '{concept}' mappato a {point:.2f} (metodo: {method})")

    elif method == 'random':

        # Punti casuali in quadrato [-scale/2, scale/2] x [-scale/2, scale/2]

        for concept in concepts:

             rx = random.uniform(-scale/2, scale/2)

             ry = random.uniform(-scale/2, scale/2)

             point = complex(rx, ry)

             points.add(point)

             logging.debug(f"Concetto '{concept}' mappato a {point:.2f} (metodo: {method})")

    else:

        logging.warning(f"Metodo '{method}' non riconosciuto per map_semantic_trajectory. Uso 'circle'.")

        return map_semantic_trajectory(concepts, method='circle', scale=scale)

 

    logging.info(f"Mappati {len(concepts)} concetti in {len(points)} punti iniziali (metodo: {method}).")

    return points

 

 

def generate_phi_from_text_basic(text):

    """

    Genera trasformazione(i) Phi da stringa di testo (Placeholder molto basilare).

    La logica reale sarebbe complessa (NLP, mapping a parametri, ecc.).

    """

    # Esempio: crea trasformazione fittizia basata sulla lunghezza del testo

    num_chars = len(text)

    # Esempio: se testo lungo, crea trasformazione che "espande"

    if num_chars > 20:

        class ExpandTransform(Transformation):

            def apply(self, point_set):

                return {p * 1.1 for p in point_set} # Espande del 10%

        phi = ExpandTransform(name=f"Expand_from_text_{num_chars}")

        logging.info(f"Generata trasformazione Phi 'Expand' da testo (lunghezza {num_chars}).")

        return [phi]

    # Esempio: se testo corto, crea trasformazione che "ruota"

    else:

        class RotateTransform(Transformation):

             def apply(self, point_set):

                 angle = np.pi / 6 # Ruota di 30 gradi

                 rot = complex(np.cos(angle), np.sin(angle))

                 return {p * rot for p in point_set}

        phi = RotateTransform(name=f"Rotate_from_text_{num_chars}")

        logging.info(f"Generata trasformazione Phi 'Rotate' da testo (lunghezza {num_chars}).")

        return [phi]

 

# ============================================================

# 6. STRUMENTI DI ANALISI E VISUALIZZAZIONE (MODIFICATO v5.0)

# ============================================================

 

def visualize_results(results, show_trajectory=False, save_path=None, show_plot=True):

    """

    Visualizza i risultati della simulazione (punti finali o traiettoria).

    Aggiornato per includere t_c nel titolo.

    """

    if not results:

        logging.error("Nessun risultato fornito per la visualizzazione.")

        return

 

    all_points = results.get("all_points")

    final_R = results.get("final_R")

    params = results.get("parameters")

    transition_info = results.get("transition_info", {})

    t_c = transition_info.get('iteration', None)

    run_id = results.get("run_id", "sim")

 

    if not all_points or not final_R or not params:

        logging.error("Dati mancanti nei risultati per la visualizzazione.")

        return

 

    plt.figure(figsize=(10, 10))

 

    # Estrai coordinate x, y da all_points (set di complessi)

    points_array = np.array([(p.real, p.imag) for p in all_points])

 

    if show_trajectory:

        # Colora i punti basandosi sul tempo (iterazione) - Richiede log più dettagliato

        # Questa implementazione usa solo all_points, quindi colora uniformemente

        plt.scatter(points_array[:, 0], points_array[:, 1], s=1, alpha=0.5, label="Traiettoria (Tutti i punti)")

        title = f"[{run_id}] Traiettoria D-ND v5.0"

    else:

        # Mostra solo i punti finali (o tutti i punti generati)

        plt.scatter(points_array[:, 0], points_array[:, 1], s=1, alpha=0.5, label="Punti Generati")

        # Evidenzia i punti dell'insieme finale R

        final_points_array = np.array([(p.real, p.imag) for p in final_R])

        if final_points_array.size > 0:

             plt.scatter(final_points_array[:, 0], final_points_array[:, 1], s=5, color='red', alpha=0.8, label=f"R Finale (|R|={len(final_R)})")

        title = f"[{run_id}] Stato Finale D-ND v5.0"

 

    # Aggiungi informazioni al titolo

    title += f" (Iter={params.iterations}, λ={params.lambda_linear:.2f})"

    if t_c:

        title += f", t_c={t_c}"

    

    plt.title(title)

    plt.xlabel("Reale")

    plt.ylabel("Immaginario")

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend()

    plt.axis('equal') # Assicura che le proporzioni siano corrette

 

    if save_path:

        try:

            # Crea la directory se non esiste

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path)

            logging.info(f"Plot salvato in: {save_path}")

        except Exception as e:

            logging.error(f"Impossibile salvare il plot in {save_path}: {e}")

            

    if show_plot:

        plt.show()

    else:

        plt.close() # Chiude la figura se non viene mostrata

 

 

def analyze_density_over_time(results, save_path=None, show_plot=True):

    """

    Analizza e visualizza la cardinalità (|R|) nel tempo.

    Utilizza il simulation_log aggiornato.

    """

    if not results: logging.error("Nessun risultato fornito."); return

    simulation_log = results.get("simulation_log", [])

    params = results.get("parameters")

    run_id = results.get("run_id", "sim")

 

    if not simulation_log or not params:

        logging.error("Log di simulazione o parametri mancanti per analisi densità.")

        return

 

    times = [log['t'] for log in simulation_log]

    cardinality = [log['|R|'] for log in simulation_log]

    phases = [log['phase'] for log in simulation_log]

 

    plt.figure(figsize=(12, 6))

    plt.plot(times, cardinality, label="|R| (Cardinalità)")

    

    # Evidenzia cambi di fase

    last_phase = None

    for i, log in enumerate(simulation_log):

        if log['phase'] != last_phase and i > 0:

            plt.axvline(x=log['t'], color='grey', linestyle=':', lw=1, 

                        label=f"Inizio Fase '{log['phase']}'" if last_phase is None else f"Fine Fase '{last_phase}'")

            # Aggiunge testo per indicare la fase (può sovrapporsi)

            plt.text(log['t'] + 5, max(cardinality)*0.9, f"Fase {log['phase']}", rotation=90, verticalalignment='center', alpha=0.7)

        last_phase = log['phase']

 

    # Evidenzia punto di transizione t_c

    transition_info = results.get("transition_info", {})

    t_c = transition_info.get('iteration', None)

    if t_c:

        plt.axvline(x=t_c, color='r', linestyle='--', lw=1.5, label=f'Transizione t_c = {t_c}')

 

    plt.title(f"[{run_id}] Evoluzione Cardinalità |R(t)|")

    plt.xlabel("Iterazione (t)")

    plt.ylabel("Numero di Punti |R|")

    plt.grid(True, linestyle='--', alpha=0.6)

    # Rimuove etichette duplicate dalla legenda

    handles, labels = plt.gca().get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())

    

    if save_path:

        try:

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path)

            logging.info(f"Plot densità salvato in: {save_path}")

        except Exception as e:

            logging.error(f"Impossibile salvare il plot densità in {save_path}: {e}")

 

    if show_plot:

        plt.show()

    else:

        plt.close()

 

# --- NUOVE Funzioni di Analisi Dinamica (v5.0) ---

def plot_dynamic_measures(results, save_path_base=None, show_plot=True):

    """Visualizza Coherence(t) e Tension(t) nel tempo."""

    if not results: logging.error("Nessun risultato fornito."); return

    simulation_log = results.get("simulation_log", [])

    params = results.get("parameters")

    run_id = results.get("run_id", "sim")

 

    if not simulation_log or not params:

        logging.warning("Log simulazione o parametri mancanti per analisi misure dinamiche.")

        return

 

    times = [log['t'] for log in simulation_log]

    # Gestisce eventuali valori non finiti nel log (anche se non dovrebbero esserci)

    coherence_values = [log.get('coherence', np.nan) for log in simulation_log]

    tension_values = [log.get('tension', np.nan) for log in simulation_log]

    phases = [log['phase'] for log in simulation_log]

 

    # Rimuove eventuali NaN per plottare correttamente

    valid_indices = [i for i, (c, t) in enumerate(zip(coherence_values, tension_values)) if np.isfinite(c) and np.isfinite(t)]

    if len(valid_indices) < len(times):

         logging.warning(f"Rimosse {len(times) - len(valid_indices)} voci non finite dal log per il plot dinamico.")

         times = [times[i] for i in valid_indices]

         coherence_values = [coherence_values[i] for i in valid_indices]

         tension_values = [tension_values[i] for i in valid_indices]

         phases = [phases[i] for i in valid_indices]

         

    if not times: # Se non ci sono dati validi

         logging.error("Nessun dato valido nel log per plottare le misure dinamiche.")

         return

 

 

    transition_info = results.get("transition_info", {})

    t_c = transition_info.get('iteration', None)

 

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    fig.suptitle(f"[{run_id}] Evoluzione Misure Dinamiche (v5.0)", fontsize=14)

 

    # Plot Coerenza

    coh_label = f'Coerenza ({params.coherence_measure_type})'

    axs[0].plot(times, coherence_values, label=coh_label, color='blue')

    axs[0].set_ylabel("Misura di Coerenza")

    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Aggiungi linea soglia coerenza

    axs[0].axhline(y=params.coherence_threshold, color='cyan', linestyle=':', lw=1, label=f'Soglia Coerenza ({params.coherence_threshold:.2g})')

    if t_c:

        axs[0].axvline(x=t_c, color='r', linestyle='--', lw=1, label=f'Transizione t_c = {t_c}')

        # Segna valore coerenza alla transizione

        coh_at_tc = transition_info.get('coherence_at_transition')

        if coh_at_tc is not None:

             axs[0].plot(t_c, coh_at_tc, 'ro', markersize=6, label=f'Coh(t_c)={coh_at_tc:.3f}')

             

    # Rimuove etichette duplicate

    handles, labels = axs[0].get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    axs[0].legend(by_label.values(), by_label.keys(), loc='best')

 

 

    # Plot Tensione

    ten_label = f'Tensione ({params.tension_measure_type})'

    axs[1].plot(times, tension_values, label=ten_label, color='orange')

    axs[1].set_ylabel("Misura di Tensione")

    axs[1].set_xlabel("Iterazione (t)")

    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Aggiungi linea soglia tensione

    axs[1].axhline(y=params.tension_threshold, color='magenta', linestyle=':', lw=1, label=f'Soglia Tensione ({params.tension_threshold:.2g})')

    # Considera scala logaritmica per tensione se varia molto

    # Prova a impostarla se la soglia è molto piccola rispetto ai valori massimi

    max_tension = max(tension_values) if tension_values else 0

    if params.tension_threshold > 0 and max_tension / params.tension_threshold > 1000: # Se c'è grande differenza

         try:

             # Filtra valori <= 0 prima di applicare scala log

             valid_tension_indices = [i for i, ten in enumerate(tension_values) if ten > 0]

             if valid_tension_indices:

                 min_positive_tension = min(tension_values[i] for i in valid_tension_indices)

                 axs[1].set_yscale('log')

                 # Imposta limite inferiore per evitare problemi con valori molto bassi o zero

                 axs[1].set_ylim(bottom=max(min_positive_tension * 0.1, 1e-9)) 

                 logging.debug("Applicata scala logaritmica all'asse Y della Tensione.")

             else:

                 logging.debug("Impossibile applicare scala log: nessun valore di tensione positivo.")

         except ValueError as e:

             logging.warning(f"Impossibile impostare scala log per tensione: {e}")

 

 

    if t_c:

        axs[1].axvline(x=t_c, color='r', linestyle='--', lw=1, label=f'Transizione t_c = {t_c}')

         # Segna valore tensione alla transizione

        ten_at_tc = transition_info.get('tension_at_transition')

        if ten_at_tc is not None:

             # Aggiusta y per plot log

             y_val = ten_at_tc if ten_at_tc > 0 else axs[1].get_ylim()[0] # Usa limite inf se 0 o neg

             axs[1].plot(t_c, y_val, 'ro', markersize=6, label=f'Ten(t_c)={ten_at_tc:.3g}')

 

    # Rimuove etichette duplicate

    handles, labels = axs[1].get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    axs[1].legend(by_label.values(), by_label.keys(), loc='best')

 

 

    # Aggiungi indicazioni di fase (opzionale, può affollare)

    # Potrebbe essere utile aggiungere background colorato per le fasi

    last_phase_ax = None

    for i, log in enumerate(simulation_log):

         if log['phase'] != last_phase_ax and i > 0:

             axs[0].axvline(x=log['t'], color='grey', linestyle=':', lw=0.8)

             axs[1].axvline(x=log['t'], color='grey', linestyle=':', lw=0.8)

             # Aggiunge testo solo una volta per evitare sovrapposizioni

             if last_phase_ax is None:

                 axs[0].text(log['t'] + 5, axs[0].get_ylim()[1]*0.9, f"Fase {log['phase']}", rotation=90, verticalalignment='center', alpha=0.7, fontsize=8)

         last_phase_ax = log['phase']

 

 

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Aggiusta layout per titolo principale

 

    if save_path_base:

        try:

            # Crea path completo per il file

            save_file = f"{save_path_base}_dynamics.png"

            os.makedirs(os.path.dirname(save_file), exist_ok=True)

            plt.savefig(save_file)

            logging.info(f"Plot misure dinamiche salvato in: {save_file}")

        except Exception as e:

            logging.error(f"Impossibile salvare il plot dinamico in {save_file}: {e}")

            

    if show_plot:

        plt.show()

    else:

        plt.close()

 

 

def export_results_to_file(results, filename="dnd_simulation_results_v5.txt"):

    """Esporta risultati chiave e serie temporali (incluse nuove misure) in file di testo."""

    if not results: logging.error("Nessun risultato fornito per l'export."); return

 

    params = results.get("parameters")

    transition_info = results.get("transition_info", {})

    simulation_log = results.get("simulation_log", [])

    final_R = results.get("final_R")

    total_time = results.get("total_time_s")

    run_id = results.get("run_id", "sim")

 

    try:

        # Crea directory se non esiste

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:

            f.write(f"=== Risultati Simulazione D-ND Framework v5.0 ===\n")

            f.write(f"Run ID: {run_id}\n")

            f.write(f"Data Esecuzione: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write(f"Tempo Totale Simulazione: {total_time:.2f} secondi\n")

 

            # Scrive i parametri usati

            f.write("\n--- Parametri Sistema (v5.0) ---\n")

            if params:

                # Usa vars() per ottenere dizionario da oggetto

                param_dict = vars(params)

                for key, value in param_dict.items():

                    # Formatta valori complessi/liste in modo leggibile

                    if isinstance(value, complex):

                        f.write(f"{key}: {value.real:.4f}{value.imag:+.4f}j\n")

                    elif isinstance(value, list) and all(isinstance(item, Transformation) for item in value):

                         f.write(f"{key}: {[str(t) for t in value]}\n")

                    else:

                        f.write(f"{key}: {value}\n")

            else:

                f.write("Parametri non trovati.\n")

 

            # Scrive informazioni sulla transizione

            f.write("\n--- Informazioni Transizione Dinamica (v5.0) ---\n")

            t_c = transition_info.get('iteration')

            if transition_info.get('occurred'):

                f.write(f"Transizione Avvenuta: Sì\n")

                f.write(f"Iterazione Transizione (t_c): {t_c}\n")

                f.write(f"Tempo Transizione (s): {transition_info.get('time', 'N/A'):.4f}\n")

                f.write(f"Coerenza a t_c: {transition_info.get('coherence_at_transition', 'N/A'):.6f}\n")

                f.write(f"Tensione a t_c: {transition_info.get('tension_at_transition', 'N/A'):.6g}\n")

            else:

                f.write("Transizione Avvenuta: No (entro le iterazioni date)\n")

 

            # Scrive informazioni sullo stato finale

            f.write("\n--- Stato Finale ---\n")

            if final_R:

                f.write(f"Cardinalità Insieme Finale |R|: {len(final_R)}\n")

                # Campiona alcuni punti finali per l'output

                sample_size = min(10, len(final_R))

                final_R_sample = random.sample(list(final_R), sample_size)

                f.write(f"Campione Punti Finali (max 10):\n")

                for p in final_R_sample:

                    f.write(f"  {p.real:.6f} + {p.imag:.6f}j\n")

            else:

                f.write("Insieme finale R è vuoto.\n")

 

            # Scrive la serie temporale del log (campionata)

            f.write("\n--- Serie Temporale Log (Campionata) ---\n")

            if simulation_log:

                # Campiona circa 100 punti o tutti se meno

                num_logs = len(simulation_log)

                sample_freq = max(1, num_logs // 100)

                f.write("t, phase, |R|, coherence, tension\n") # Header CSV-like

                # Include sempre il primo e l'ultimo punto

                f.write(f"{simulation_log[0]['t']}, {simulation_log[0]['phase']}, {simulation_log[0]['|R|']}, "

                        f"{simulation_log[0]['coherence']:.6f}, {simulation_log[0]['tension']:.6g}\n")

                for i in range(1, num_logs - 1):

                    if i % sample_freq == 0:

                         log_entry = simulation_log[i]

                         f.write(f"{log_entry['t']}, {log_entry['phase']}, {log_entry['|R|']}, "

                                 f"{log_entry.get('coherence', np.nan):.6f}, {log_entry.get('tension', np.nan):.6g}\n")

                # Assicura che l'ultimo punto sia sempre incluso

                if num_logs > 1:

                     log_entry = simulation_log[-1]

                     f.write(f"{log_entry['t']}, {log_entry['phase']}, {log_entry['|R|']}, "

                             f"{log_entry.get('coherence', np.nan):.6f}, {log_entry.get('tension', np.nan):.6g}\n")

 

            else:

                f.write("Log di simulazione vuoto.\n")

 

        logging.info(f"Esportati risultati simulazione v5.0 in: {filename}")

 

    except IOError as e:

        logging.error(f"Errore durante l'esportazione dei risultati in {filename}: {e}")

    except Exception as e:

         logging.error(f"Errore imprevisto durante l'esportazione: {e}")

 

 

# ============================================================

# 7. ESEMPIO USO E ESECUZIONE ESPERIMENTO H1 (MODIFICATO v5.0)

# ============================================================

if __name__ == "__main__":

    # Configura logging a livello INFO per l'esecuzione principale

    # (può essere cambiato a DEBUG per più dettagli)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    

    # Crea cartella per i risultati di questo run

    output_dir = f"dnd_v5_results_{time.strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Salvataggio risultati e plot in: ./{output_dir}/")

 

    # --- Configurazione Parametri v5.0 ---

    # Usa parametri dall'esempio originale, adattati per test rapidi

    params_base = SystemParameters(

        iterations=1500,          # Aumentato leggermente per vedere meglio dinamica

        lambda_linear=0.08,

        P=complex(0.6, 0.4),

        blend_iterations=50,      # Aumentato per vedere effetto blend

        offset_A=complex(0.2, 0.7), offset_B=complex(0.8, 0.1),

        scale_factor_A=0.55, scale_factor_B=0.45,

        # Parametri chiave v5.0 per transizione

        coherence_measure_type='dispersion', # Usa dispersione come misura coerenza

        tension_measure_type='coherence_change', # Usa variazione coerenza per tensione

        coherence_threshold=0.08,  # Soglia coerenza leggermente più alta per test

        tension_threshold=5e-5,    # Soglia tensione

        spatial_entropy_bins=15    # Esempio per entropia spaziale (se usata)

    )

 

    # --- Setup Esperimento H1 v5.0 ---

    # Definisci insiemi di concetti per condizioni iniziali diverse

    concepts_math = ["set", "point", "line", "fractal", "iteration", "limit", "convergence"]

    concepts_nature = ["tree", "leaf", "branch", "water", "flow", "spiral", "growth"]

    concepts_empty = [] # Per testare R0 = {0}

    concepts_random = ["random_" + str(i) for i in range(10)] # 10 concetti casuali

 

    # Genera R0 per ciascuna condizione

    # Nota: map_semantic_trajectory è un placeholder, i risultati dipendono dalla sua implementazione

    R0_default = None # Inizia da origine (implicito se R0=None)

    R0_math = map_semantic_trajectory(concepts_math, method='spiral', scale=1.5)

    R0_nature = map_semantic_trajectory(concepts_nature, method='circle', scale=1.0)

    # R0_empty = map_semantic_trajectory(concepts_empty) # Dovrebbe dare {0}

    R0_random = map_semantic_trajectory(concepts_random, method='random', scale=2.0)

 

    # Lista delle configurazioni da testare

    test_configs = [

        {"id": "default", "R0": R0_default, "desc": "Partenza da Origine"},

        {"id": "math", "R0": R0_math, "desc": "Partenza da Concetti Matematici (Spirale)"},

        {"id": "nature", "R0": R0_nature, "desc": "Partenza da Concetti Naturali (Cerchio)"},

        {"id": "random", "R0": R0_random, "desc": "Partenza da Punti Casuali"},

    ]

 

    # Dizionario per conservare i risultati di ogni run

    all_results = {}

 

    # --- Esegui Simulazioni per ogni Configurazione ---

    for config in test_configs:

        run_id = config["id"]

        desc = config["desc"]

        R0 = config["R0"]

        

        logging.info(f"\n{'='*10} ESECUZIONE SIMULAZIONE: {run_id} ({desc}) {'='*10}")

        

        # Esegui la simulazione

        results = run_full_simulation(params_base, R0=R0, run_id=run_id)

        all_results[run_id] = results # Salva i risultati

 

        # Genera output per questa run

        base_filename = os.path.join(output_dir, f"dnd_v5_{run_id}")

        visualize_results(results, save_path=f"{base_filename}_final.png", show_plot=False)

        plot_dynamic_measures(results, save_path_base=base_filename, show_plot=False)

        analyze_density_over_time(results, save_path=f"{base_filename}_density.png", show_plot=False)

        export_results_to_file(results, filename=f"{base_filename}_results.txt")

 

        logging.info(f"--- Fine Esecuzione {run_id} ---")

 

    # --- Analisi Base per Ipotesi H1 v5.0 ---

    print("\n" + "="*20)

    print("=== Analisi Base Ipotesi H1 v5.0 ===")

    print("Confronto Tempo di Transizione Critica (t_c):")

    print("---------------------------------------------")

    

    t_c_values = {}

    for run_id, results in all_results.items():

        t_c = results['transition_info'].get('iteration', 'N/A')

        t_c_values[run_id] = t_c

        print(f"- {run_id.capitalize()} Start: t_c = {t_c}")

 

    # Commenti sull'analisi da fare

    print("\nAnalisi Suggerite:")

    print("1. Confrontare i valori di t_c ottenuti. Ci sono differenze significative tra le condizioni iniziali?")

    print(f"2. Esaminare i plot salvati nella cartella '{output_dir}':")

    print("   - Plot Dinamici (_dynamics.png): Le forme delle curve Coerenza(t) e Tensione(t) differiscono prima di t_c?")

    print("   - Plot Finali (_final.png): Le strutture geometriche finali mostrano differenze qualitative o quantitative?")

    print("   - Plot Densità (_density.png): L'evoluzione di |R(t)| varia con R0?")

    print("3. Per analisi più rigorosa (H1(b), H1(c)): confrontare quantitativamente le traiettorie (es. pendenze, valori medi) e le metriche geometriche finali (es. dimensione frattale stimata, misure di forma).")

    print("="*20)

 

    # Esempio aggiuntivo: run con trasformazione Phi (se necessario)

    # text_phi = "Un lungo testo descrittivo per generare una trasformazione Phi espansiva."

    # generated_transformations = generate_phi_from_text_basic(text_phi)

    # params_phi = SystemParameters(...) # Copia params_base e aggiungi generated_phi

    # params_phi.generated_phi = generated_transformations

    # logging.info("\n--- Running Simulation v5.0: Default Start with Phi ---")

    # results_phi = run_full_simulation(params_phi, R0=None, run_id="default_phi")

    # visualize_results(results_phi, save_path=os.path.join(output_dir,"dnd_v5_default_phi_final.png"), show_plot=False)

    # plot_dynamic_measures(results_phi, save_path_base=os.path.join(output_dir,"dnd_v5_default_phi"), show_plot=False)

    # export_results_to_file(results_phi, filename=os.path.join(output_dir,"dnd_v5_results_default_phi.txt"))