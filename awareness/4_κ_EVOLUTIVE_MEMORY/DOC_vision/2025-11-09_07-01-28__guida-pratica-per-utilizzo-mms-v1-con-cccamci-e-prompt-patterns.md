---
title: "Guida Pratica per Utilizzo MMS vΦ.1 con CCCA/MCI e Prompt Patterns"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:01:27.838Z"
---
# Guida Pratica per Utilizzo MMS vΦ.1 con CCCA/MCI e Prompt Patterns

## Ultima risposta assistente

{
  "version": "1.1",
  "source": "MMS vΦ.1 — Fonte Unificata v1.1 (Kernel Profondo)",
  "description": "Configurazione CMP Evo-Loop + Pipeline CCCA/MCI. Artefatto derivato, allineato al Kernel. Nessun tuning locale arbitrario.",
  "meta": {
    "lineage": ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CMP"],
    "hard_invariants": ["P0","P1","P2","P3","P4","P5","P6","P7"],
    "note": "Questo file definisce il comportamento del loop evolutivo CMP e delle pre-pipeline CCCA/MCI. Il runtime deve attenersi a queste regole."
  },

  "cmp_config": {
    "objective": "Usare un loop evolutivo leggero per migliorare stati operativi solo quando esiste un guadagno stabile e misurabile, altrimenti mantenere lo stato base. Nessuna euristica ad hoc.",
    "Q_child": {
      "description": "Funzione di qualità derivata dal Kernel, coerente con ActionScore.",
      "formula_abstract": "Q(child) = w_q*quality_global + w_gc*S_Causal + w_fp*C_FP_global - w_v*V_mod_FP - w_l*latency_norm + w_g*generalization_score",
      "constraints": {
        "w_q":  ">= 0",
        "w_gc": ">= 0",
        "w_fp": ">= 0",
        "w_v":  ">= 0",
        "w_l":  ">= 0",
        "w_g":  ">= 0",
        "normalization": "Somma pesi normalizzata; valori specifici definiti a livello Kernel e aggiornabili solo via KLI aggregati.",
        "monotonicity": "Coerente con ActionScore: Q cresce con qualità, S_Causal, C_FP_global, generalization; decresce con V_mod_FP e latenza."
      }
    },
    "cmp_estimate": {
      "description": "Fitness evolutiva attesa dello stato operativo (parent).",
      "aggregation": "Funzione (es. media pesata/top-p) di Q(child) dei candidati generati nel loop evolutivo.",
      "constraints": {
        "range": "[0,1]",
        "kli_driven_update": "La stima viene aggiornata solo tramite serie storiche KLI, non per singolo evento."
      }
    },
    "loop": {
      "max_children": 4,
      "max_depth": 2,
      "max_beam": 2,
      "steps": [
        "0: Genera stato base R0 (parent) usando routing e moduli selezionati.",
        "1: Genera fino a k <= max_children varianti leggere (child) del percorso: cambi combinazioni moduli, parametri, micro-strategie, ma senza violare Kernel.",
        "2: Per ogni child, esegui cheap_probes su un piccolo set di micro-task per stimare Q(child).",
        "3: Calcola cmp_estimate_parent come funzione dei migliori child.",
        "4: Confronta cmp_delta = cmp_estimate_parent - cmp_baseline.",
        "5: Se cmp_delta >= tau_cmp e nessuna violazione di P0–P7, quality_global, Stream-Guard: promuovi il best_child; altrimenti rollback a R0.",
        "6: Registra sempre KLI con dettagli su Q, cmp_delta, decisioni, rollback/commit."
      ],
      "thresholds": {
        "tau_cmp": 0.05,
        "epsilon_stall": 0.01,
        "quality_min": 0.8,
        "max_latency_penalty_ratio": 0.10
      },
      "stop_conditions": {
        "description": "Il loop è conservativo; si ferma presto.",
        "rules": [
          "CMP_gain < epsilon_stall per 2 iterazioni consecutive.",
          "Budget (tempo/compute) esaurito.",
          "Pruning_ratio >= 0.5.",
          "Violazione Stream-Guard o ActionScore peggiorato in modo significativo."
        ]
      },
      "guardrails": {
        "no_overwrite_kernel": true,
        "no_untracked_changes": true,
        "require_kli": true,
        "respect_stream_guard": true
      }
    }
  },

  "ccca_mci_pipeline": {
    "description": "Pre-pipeline obbligatoria: ogni input significativo passa per CCCA-lite + MCI-lite prima del routing e del collasso.",
    "hard_rule": "Nessuna decisione di routing o collasso può essere presa senza aver eseguito CCCA-lite e MCI-lite.",
    "ccca_lite": {
      "name": "Sistema di Codifica/Decodifica Contestuale Autopoietica (lite)",
      "goals": [
        "Rimuovere superfluo evidente.",
        "Estrarre struttura minima che consente rigenerazione del contesto.",
        "Preparare materiale per MCI-lite e Router."
      ],
      "inputs": [
        "user_raw_input",
        "recent_history"
      ],
      "outputs": [
        "metatag_list",        // concetti chiave
        "proto_actions_list",  // azioni/operazioni implicite
        "constraints_list",    // vincoli espliciti/impliciti
        "context_min"          // sintesi minimale del contesto rilevante
      ],
      "operations": [
        "Identifica concetti chiave e ruoli.",
        "Estrai proto-azioni (cosa il sistema dovrebbe fare).",
        "Evidenzia vincoli (formato, sicurezza, dominio, preferenze).",
        "Comprimi: elimina frasi ornamentali non funzionali (P7)."
      ],
      "constraints": {
        "must_preserve_semantic_integrity": true,
        "no_invention": "Non aggiungere intenti non presenti.",
        "log_to_kli": true
      }
    },
    "mci_lite": {
      "name": "Modellazione Consequenziale dell’Intento Interferenziale (lite)",
      "goals": [
        "Posizionare l’intento come attrattore/target.",
        "Stabilire traiettoria consequenziale dall’input alla Risultante.",
        "Definire criteri minimi per valutare se una risposta è allineata all’intento."
      ],
      "inputs": [
        "metatag_list",
        "proto_actions_list",
        "constraints_list",
        "context_min"
      ],
      "outputs": [
        "intent_core",          // formulazione compatta dell'intento
        "intent_type",          // analysis, synthesis, prompt_generation, insight_discovery, self_reflection, ecc.
        "target_outcome",       // cosa deve esistere alla fine (es. 'system_prompt', 'workflow', 'spiegazione tecnica')
        "success_criteria_min"  // condizioni minime di allineamento (coerenza, forma, vincoli)
      ],
      "operations": [
        "Deriva una definizione sintetica dell’intento a partire dai metatag/proto-azioni.",
        "Classifica il task_type in base all’intent_core.",
        "Definisce target_outcome (tipo di artefatto/collasso atteso).",
        "Stabilisce success_criteria_min coerenti con P0–P7 e con i vincoli espliciti."
      ],
      "constraints": {
        "confidence_min": 0.8,
        "if_confidence_below": "richiedi_chiarimento_o_applica_surface_response",
        "log_to_kli": true
      }
    },
    "wiring": {
      "sequence": [
        "1: Esegui CCCA-lite su user_raw_input.",
        "2: Esegui MCI-lite sui risultati di CCCA-lite.",
        "3: Passa intent_core, intent_type, target_outcome, success_criteria_min e strutture CCCA al Router.",
        "4: Router seleziona combo moduli coerenti con questi elementi.",
        "5: CMP Evo-Loop (se abilitato) opera sopra le traiettorie candidate.",
        "6: Stream-Guard valuta; Morpheus collassa sul candidato con ActionScore massimo."
      ],
      "hard_invariants": {
        "no_direct_routing_without_ccca_mci": true,
        "must_record_kli": true
      }
    }
  }
}

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
