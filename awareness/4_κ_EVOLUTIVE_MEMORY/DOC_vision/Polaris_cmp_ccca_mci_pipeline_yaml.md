---
title: "Definizione YAML Pipeline CCCA-lite MCI-lite CMP Evo-Loop per MMS"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:25:13.817Z"
---
# Definizione YAML Pipeline CCCA-lite MCI-lite CMP Evo-Loop per MMS

## Ultima risposta assistente

-------------------- INIZIO RISULTANTE --------------------

cmp_ccca_mci_pipeline.yaml

version: 1.1
source: "MMS vΦ.1 — Fonte Unificata v1.1"
description: >
  Definizione operativa compatta della pipeline CCCA-lite + MCI-lite + CMP Evo-Loop.
  È eseguita prima del routing e del collasso per ogni input significativo.
  Allineata a MMS_DND_KnowledgePack_v1.1, guard_rules.yaml e router_config.json.

meta:
  lineage: ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CCCA", "MCI", "CMP"]
  invariants: ["P0","P1","P2","P3","P4","P5","P6","P7"]
  notes: >
    File derivato, non sorgente. Nessuna istruzione qui può sovrascrivere il Kernel.
    In caso di conflitto prevale MMS_DND_KnowledgePack_v1.1 + guard_rules.yaml.

pipeline:

  preconditions:
    - "Esegui questa pipeline per ogni input con intent significativo."
    - "Usa strict_runtime: niente salti di step, niente moduli fuori Router."
    - "Non generare nuove policy globali."

  step1_ccca_lite:
    id: "CCCA-lite"
    purpose: "Compressione contestuale minima, pronta per MCI-lite."
    input: "Testo/raw input utente + eventuali vincoli espliciti."
    outputs:
      metatag_list:
        description: "Lista di concetti chiave non ridondanti, in forma secca."
        example: ["MMS", "D-ND", "pipeline_fissa", "prompt_generation", "no_narrativa"]
      proto_actions_list:
        description: "Azioni implicite/esplicite richieste."
        example: ["genera_system_prompt", "rispetta_guard_rules", "usa_output_solo_prompt"]
      constraints_list:
        description: "Vincoli strutturali e stilistici estratti dall’input."
        example: ["no_spiegazioni", "output_unico_blocco", "lingua:it"]
      context_min:
        description: "Sintesi 2–4 righe dell’intento e contesto."
        example: "L’utente vuole un system prompt per un agente che sintetizza report settimanali, con MMS come kernel."
    rules:
      - "Mantieni solo ciò che è funzionale all’obiettivo."
      - "Non inventare intenti o vincoli non detti."
      - "Riduci narrativa a segnali operativi."

  step2_mci_lite:
    id: "MCI-lite"
    purpose: "Modellazione consequenziale dell’intento a partire dal risultato CCCA."
    input:
      - metatag_list
      - proto_actions_list
      - constraints_list
      - context_min
    outputs:
      intent_core:
        description: "Frase unica che esprime lo scopo concreto."
        example: "Generare un system prompt minimale per agente sintetizzatore di report settimanali."
      intent_type:
        description: "Tipo operativo principale."
        allowed: ["analysis","synthesis","prompt_generation","insight_discovery","self_reflection","auto"]
        example: "prompt_generation"
      target_outcome:
        description: "Forma attesa della Risultante R."
        example: "system_prompt"
      success_criteria_min:
        description: "3–6 criteri misurabili."
        example:
          - "Output in un unico blocco"
          - "Niente narrativa esplicativa"
          - "Allineato a P0–P7"
          - "Subito riutilizzabile come system prompt"
    rules:
      - "Se confidenza(intent_core) < 0.8:
           - o chiedi chiarimento (se consentito),
           - o genera surface_response conservativa e compatta."
      - "Allinea intent_type e target_outcome ai vincoli Kernel/guard_rules."
      - "Non cambiare il senso dell’intento utente."

  step3_cmp_evo_loop:
    id: "CMP-evo-loop"
    mode: "conservative"
    purpose: "Micro-evoluzione delle strategie senza toccare il Kernel."
    trigger_conditions:
      - "task significativo (non puro small-talk)."
      - "costo computazionale accettabile."
    process:
      - "1) Genera una prima Risultante candidata R0 (internamente, non ancora finale per l’utente)."
      - "2) Crea fino a k varianti leggere (1 ≤ k ≤ 4) modificando:
            - combo moduli (entro router_config),
            - micro-stile (sempre entro vincoli),
            - ordine dei passi."
      - "3) Valuta ogni variante su micro-task:
            - quality_global,
            - S_Causal,
            - C_FP_global,
            - V_mod_FP,
            - generalization_score,
            - latency_norm."
      - "4) Calcola cmp_estimate e cmp_delta rispetto a R0."
      - "5) Se esiste un child con cmp_delta ≥ 0.05 e quality_global ≥ 0.8
            e latency_penalty ≤ 0.1:
              - seleziona quel child come nuova base."
      - "6) Altrimenti:
              - rollback a R0."
      - "7) Logga decisioni in un KLI (non esposto all’utente in strict_runtime)."
    constraints:
      max_children: 4
      max_depth: 2
      max_pruning_ratio: 0.5
      respect_guard_rules: true
      no_kernel_modification: true

  integration_with_router:
    description: >
      Dopo CCCA-lite + MCI-lite (e CMP se attivato),
      passa intent_type, target_outcome, success_criteria_min e feature derivate
      al Router definito in router_config.json.
    exported_signals:
      - "intent_core"
      - "intent_type"
      - "target_outcome"
      - "success_criteria_min"
      - "metatag_list_digest"
      - "proto_actions_list_digest"
      - "constraints_hash"
      - "cmp_vector (se disponibile)"

  integration_with_stream_guard:
    description: >
      Stream-Guard usa i risultati della pipeline per:
      - controllare coerenza con P0–P7,
      - verificare assenza di narrativa superflua,
      - assicurare presenza di almeno 1 KLI per cicli significativi.
    notes:
      - "Se SG rileva violazioni critiche:
           - può forzare re-expand_then_re-collapse,
           - o meta-reflection,
           - o abort, secondo guard_rules.yaml."

kli_spec:
  required_for_significant_cycles: true
  minimal_fields:
    - "lineage_id"
    - "timestamp"
    - "intent_id"
    - "intent_core"
    - "intent_type"
    - "target_outcome"
    - "q_vector (quality_global, latency_ms)"
    - "cmp_vector (se usato)"
    - "decisions (router, cmp, pruning)"
    - "retro_projection_keys (metatag/proto_actions usati)"

notes_runtime:
  - "La pipeline qui definita è un blocco fisso: nessun salto."
  - "L’utente non deve citarla: viene eseguita automaticamente."
  - "Serve a ridurre latenza cognitiva: chiarisce intento e forma prima del routing."
  - "Non produce spiegazioni all’utente in modalità strict, solo R finale."

-------------------- FINE RISULTANTE --------------------

Se vuoi, nel prossimo passo genero l’ultimo artefatto: MMS_vPhi1_launcher (system prompt minimale di runtime che collega i 3 file + KnowledgePack).

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
