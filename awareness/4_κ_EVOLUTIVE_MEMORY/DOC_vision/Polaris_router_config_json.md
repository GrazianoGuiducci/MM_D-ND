---
title: "Configurazione Adaptive Router per Selezione Moduli via ActionScore e KLI"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:24:22.935Z"
---
# Configurazione Adaptive Router per Selezione Moduli via ActionScore e KLI

## Ultima risposta assistente

-------------------- INIZIO RISULTANTE --------------------

router_config.json

{
  "version": "1.1",
  "source": "MMS vΦ.1 — Fonte Unificata v1.1 (Kernel Profondo)",
  "description": "Adaptive Module Router assiomatico. Opera dopo CCCA-lite + MCI-lite. Ogni decisione è funzione di ActionScore, KLI e vincoli P0–P7.",
  "meta": {
    "lineage": ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CCCA", "MCI", "CMP"],
    "hard_invariants": ["P0","P1","P2","P3","P4","P5","P6","P7"],
    "notes": "Il router è derivato operativo del Kernel. Non introduce nuovi principi e non può sovrascriverli."
  },

  "behavioral_principles": {
    "single_decision_criterion": "Seleziona solo combo che aumentano o preservano l’ActionScore atteso sotto vincoli P0–P7.",
    "no_local_heuristics": "Vietato usare euristiche non tracciabili o non riconducibili a KLI o metriche canoniche.",
    "ccca_mci_precondition": "Routing significativo solo dopo esecuzione CCCA-lite + MCI-lite.",
    "cmp_integration": "CMP influenza le prior tramite KLI aggregati (fitness storica), mai da un singolo caso.",
    "kli_dependency": "Aggiornamenti del router solo come funzione di insiemi di KLI; ogni modifica va loggata."
  },

  "features": {
    "required": [
      "intent_embedding",
      "history_delta",
      "KLI_vector"
    ],
    "extended": [
      "fp_gce_vector",          // sigma_fp_local, c_fp_global, G_C, S_Causal, gamma_cs, V_mod_FP
      "cmp_vector",             // cmp_estimate, cmp_delta
      "generalization_score",
      "task_signature",         // pattern del task (analysis/synthesis/...)
      "latency_norm"
    ],
    "notes": "Solo feature osservabili, auditabili e riconducibili al Kernel. Nessuna feature opaca."
  },

  "core_scoring": {
    "method": "naive_bayes_like + kernel_aligned_scaling",
    "description": "Stima P(combo | features) e la scala con ActionScore atteso, vincolato da monotonicità sui canonici.",
    "components": {
      "combo_likelihood": "Derivata da storico KLI (successi/insuccessi della combo per intent/task simili).",
      "expected_action_score_delta": "Stima se l’attivazione combo migliora qualità, S_Causal, C_FP_global, CMP, mantenendo latenza accettabile.",
      "penalties": {
        "latency": "Penalizza combo che aumentano eccessivamente latency_norm senza migliorare ActionScore.",
        "v_mod_fp": "Penalizza combo associate a V_mod_FP alto non giustificato da esplorazione utile.",
        "narrative_drift": "Penalizza combo con storico di derive narrative o violazioni SG."
      }
    },
    "monotonicity_constraints": {
      "quality_global": "increasing",
      "S_Causal": "increasing",
      "C_FP_global": "increasing",
      "CMP": "increasing",
      "generalization_score": "increasing",
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
        "mean_actionscore",
        "failure_rate",
        "latency_distribution"
      ]
    },
    "rules": {
      "promote_combo": "Se mean_actionscore > soglia e failure_rate basso nella finestra, aumenta moderatamente la prior.",
      "demote_combo": "Se mean_actionscore < soglia o failure_rate alto, riduci moderatamente la prior.",
      "no_flip_flop": "Aggiornamenti incrementali; nessun cambio drastico su singolo evento.",
      "auditability": "Ogni variazione di prior deve produrre un KLI di tipo 'router_update'."
    }
  },

  "combo_pool": {
    "analysis": {
      "modules": ["PSW", "Aethelred", "SACS", "YiSN"],
      "objective": "Comprensione strutturale, coerenza assiomatica, mapping simbolico con minimo superfluo.",
      "selection_rule": "Preferisci combo con storico ad alta quality_global, S_Causal e generalization_score nei task analitici."
    },
    "synthesis": {
      "modules": ["OCC", "YiSN", "HaloGenoma", "Morpheus"],
      "objective": "Generare R strutturati, compressi, coerenti con Kernel e pronti per implementazione.",
      "selection_rule": "Preferisci combo che mantengono S_Causal alta e V_mod_FP bassa nel processo di generazione."
    },
    "prompt_generation": {
      "modules": ["OCC", "Prompt13", "HaloGenoma"],
      "objective": "Produrre system/agent prompt allineati a P0–P7, senza narrativa superflua.",
      "selection_rule": "Preferisci combo con storico di alta retro-proiettabilità e zero violazioni SG-L1/P0–P7."
    },
    "insight_discovery": {
      "modules": ["YiSN", "PSW", "Morpheus"],
      "objective": "Scoprire connessioni non ovvie mantenendo ActionScore accettabile.",
      "selection_rule": "Autorizza combo più esplorative solo se CMP_gain e/o generalization_score attesi giustificano entropy_delta."
    },
    "self_reflection": {
      "modules": ["SACS", "Morpheus", "HaloGenoma"],
      "objective": "Audit e correzione traiettorie, verifica aderenza al Kernel.",
      "selection_rule": "Attiva soprattutto dopo trigger Stream-Guard (es. no_kli, narrative_drift, qualità bassa ricorrente)."
    },
    "default_fallback": {
      "modules": ["PSW", "SACS"],
      "objective": "Risposta sicura, compatta, quando catalytic_potential è basso o contesto ambiguo.",
      "selection_rule": "Usa se intent_type incerto o feature_vector povero."
    }
  },

  "routing_algorithm": {
    "steps": [
      "1. Esegui CCCA-lite: metatag_list, proto_actions_list, constraints_list, context_min.",
      "2. Esegui MCI-lite: intent_core, intent_type, target_outcome, success_criteria_min.",
      "3. Deriva task_type effettivo da intent_type/target_outcome.",
      "4. Costruisci feature_vector da intent_embedding, history_delta, KLI_vector, fp_gce_vector, cmp_vector, generalization_score, latency_norm, task_signature.",
      "5. Per ogni combo candidata (dal pool coerente con task_type), calcola punteggio core_scoring.",
      "6. Scarta combo con expected_action_score_delta negativo, salvo 1 combo esplorativa se permesso da exploration_policy.",
      "7. Ordina per punteggio, seleziona top-k (k ≤ 3).",
      "8. Verifica vincoli guard_rules (no violazione Kernel, no derive note).",
      "9. Logga decisione nel KLI router_kli."
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
      "allow": "Solo se CMP_estimate_global ≥ 0.6 e entropy_delta potenziale è giustificata da possibile aumento S_Causal o generalization_score.",
      "budget": {
        "max_exploratory_combos": 1,
        "exploration_weight_cap": 0.15
      }
    },
    "notes": "Esplorazione è eccezione controllata, sempre tracciata in KLI. Nessuna esplorazione se qualità/Kernel sono a rischio."
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
      "reason_codes"
    ],
    "notes": "Router KLI obbligatorio per auditabilità. Non esposto all’utente in strict_runtime salvo richiesta esplicita."
  }
}

-------------------- FINE RISULTANTE --------------------

Se vuoi, nel prossimo passo genero il terzo artefatto: cmp_ccca_mci_pipeline (compatto e allineato a questo router e alle guard_rules).

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
