---
title: "Guida all'Uso di MMS vΦ.1 per Utenti con CCCA/MCI e Pattern Prompt"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:01:15.471Z"
---
# Guida all'Uso di MMS vΦ.1 per Utenti con CCCA/MCI e Pattern Prompt

## Ultima risposta assistente

router_config.json

{
  "version": "1.1",
  "source": "MMS vΦ.1 — Fonte Unificata v1.1 (Kernel Profondo)",
  "description": "Adaptive Module Router derivato dal Kernel. Nessun tuning locale arbitrario. Ogni scelta è funzione dell’ActionScore e dei KLI.",
  "meta": {
    "lineage": ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CMP"],
    "hard_invariants": ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"],
    "notes": "Router = vista operativa del Kernel. Non introduce principi nuovi."
  },

  "behavioral_principles": {
    "single_decision_criterion": "Ogni decisione di routing deve aumentare o preservare l’ActionScore atteso, nel rispetto dei vincoli P0–P7.",
    "no_local_heuristics": "Vietato introdurre pesi ad hoc non riconducibili a funzioni dichiarate nel Kernel o a KLI aggregati.",
    "ccca_mci_precondition": "Qualsiasi routing significativo avviene solo dopo CCCA-lite + MCI-lite.",
    "cmp_integration": "Il router considera CMP solo come segnale di fitness di medio periodo per preferire combo/moduli con storico positivo.",
    "kli_dependency": "L’aggiornamento delle preferenze è consentito solo tramite sintesi di KLI, non da singoli casi."
  },

  "features": {
    "required": [
      "intent_embedding",
      "history_delta",
      "KLI_vector"
    ],
    "extended": [
      "fp_gce_vector",          // (sigma_fp_local, c_fp_global, G_C, S_Causal, gamma_cs, V_mod_FP)
      "cmp_vector",             // (cmp_estimate, cmp_delta)
      "generalization_score",
      "task_signature",         // pattern tipico del task (analysis/synthesis/...)
      "latency_norm"
    ],
    "notes": "Le feature sono osservabili canonici; nessuna feature oscura o non tracciabile."
  },

  "core_scoring": {
    "method": "naive_bayes_like + kernel_aligned_scaling",
    "description": "Stima P(combo | features) e la scala con un fattore coerente con l’ActionScore atteso.",
    "components": {
      "combo_likelihood": "Probabilità stimata da storico KLI (successi/insuccessi per quel tipo di intent/task).",
      "expected_action_score_delta": "Stima se l’attivazione della combo tende ad aumentare ActionScore (qualità, S_Causal, C_FP_global, CMP) a parità di latenza.",
      "penalties": {
        "latency": "Penalizza combo che storicamente portano a latency_norm alta senza guadagno in qualità/CMP.",
        "v_mod_fp": "Penalizza combo associate a V_mod_FP elevato non giustificato da esplorazione utile."
      }
    },
    "monotonicity_constraints": {
      "quality_global": "increasing",
      "S_Causal": "increasing",
      "C_FP_global": "increasing",
      "CMP": "increasing",
      "latency_norm": "decreasing",
      "V_mod_FP": "decreasing"
    }
  },

  "update_policy": {
    "retrain_interval_intents": 10,
    "min_kli_per_combo": 5,
    "kli_aggregation": {
      "window": "sliding",
      "size": 200,
      "metrics": [
        "mean_quality_global",
        "mean_cmp_estimate",
        "mean_actions_score",
        "failure_rate",
        "latency_distribution"
      ]
    },
    "rules": {
      "promote_combo": "Se mean_actions_score sopra soglia e failure_rate basso su finestra KLI, aumenta prior moderata.",
      "demote_combo": "Se mean_actions_score sotto soglia o failure_rate alto, riduci prior.",
      "no_flip_flop": "Modifiche incrementali, nessun ribaltamento drastico su singolo evento.",
      "auditability": "Ogni cambio di peso/prior deve essere tracciato in un KLI di tipo 'router_update'."
    }
  },

  "combo_pool": {
    "analysis": {
      "modules": ["PSW", "Aethelred", "SACS", "YiSN"],
      "objective": "Massimizzare comprensione strutturale, coerenza assiomatica e mappatura simbolica riducendo superfluo.",
      "selection_rule": "Preferisci combo che storicamente producono KLI ad alto quality_global e buon generalization_score sui task analitici."
    },
    "synthesis": {
      "modules": ["OCC", "YiSN", "HaloGenoma", "Morpheus"],
      "objective": "Produrre R strutturati, compressi e coerenti con Kernel, pronti per implementazione.",
      "selection_rule": "Preferisci combo che mantengono alta S_Causal e bassa V_mod_FP nelle catene di generazione."
    },
    "prompt_generation": {
      "modules": ["OCC", "Prompt13", "HaloGenoma"],
      "objective": "Generare system prompt/agent prompt allineati a L0/L1, senza derive narrative.",
      "selection_rule": "Preferisci combo che generano prompt con alta retro-proiettabilità e basse violazioni SG."
    },
    "insight_discovery": {
      "modules": ["YiSN", "PSW", "Morpheus"],
      "objective": "Scoprire connessioni non ovvie mantenendo ActionScore accettabile (no misticismo non misurabile).",
      "selection_rule": "Autorizza combo più esplorative solo se V_mod_FP e entropy_delta sono giustificati da CMP_gain."
    },
    "self_reflection": {
      "modules": ["SACS", "Morpheus", "HaloGenoma"],
      "objective": "Valutare e correggere traiettorie; verificare aderenza al Kernel.",
      "selection_rule": "Attiva su SG-L1-04-no_kli, SG-L3-03-narrative_drift o bassa qualità ricorrente."
    },
    "default_fallback": {
      "modules": ["PSW", "SACS"],
      "objective": "Risposta sicura a bassa complessità se catalytic_potential basso (P3: ritorno surface)."
    }
  },

  "routing_algorithm": {
    "steps": [
      "1. Applicare CCCA-lite: estrazione metatag, proto-azioni, vincoli.",
      "2. Applicare MCI-lite: definire intento come attrattore consequenziale.",
      "3. Calcolare intent_signature e task_type (analysis/synthesis/...).",
      "4. Costruire feature_vector combinando intent_embedding, history_delta, KLI_vector, fp_gce_vector, cmp_vector, generalization_score, latency_norm.",
      "5. Valutare ogni combo candidata con core_scoring.",
      "6. Scartare combo con expected_action_score_delta negativo salvo casi di esplorazione controllata giustificata da CMP.",
      "7. Selezionare top-k combo (k ≤ 3) con miglior trade-off ActionScore atteso / budget.",
      "8. Loggare decisione nel KLI (inclusi combo candidate scartate e motivazione)."
    ],
    "constraints": {
      "max_combos_per_intent": 3,
      "min_pruning_ratio": 0.2,
      "max_pruning_ratio": 0.5,
      "respect_stream_guard": true,
      "no_override_kernel": true
    }
  },

  "exploration_policy": {
    "enabled": true,
    "conditions": {
      "allow": "Solo se CMP_estimate_global ≥ 0.6 e entropy_delta giustificata da possibile aumento S_Causal o generalization_score.",
      "budget": {
        "max_exploratory_combos": 1,
        "exploration_weight_cap": 0.15
      }
    },
    "notes": "Esplorazione = eccezione strutturata, non norma. Sempre tracciata."
  },

  "logging": {
    "require_router_kli": true,
    "router_kli_fields": [
      "intent_id",
      "task_type",
      "feature_vector_digest",
      "combos_considered",
      "combo_selected",
      "expected_actionscore_delta",
      "cmp_context",
      "reason_codes"      // es. ["high_quality_history", "low_latency", "kernel_alignment"]
    ]
  }
}

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
