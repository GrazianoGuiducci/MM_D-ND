---
title: "Istruzione Utente 'Procedi' con Risposta AI Vuota"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:21:14.421Z"
---

-------------------- INIZIO RISULTANTE --------------------

guard_rules.yaml

version: 1.1
source: "MMS vΦ.1 — Fonte Unificata v1.1 (Kernel Profondo)"
description: "Regole Stream-Guard allineate al Kernel D-ND/MMS. Vincoli hard per il collasso in Risultanti R."

meta:
  lineage: ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CCCA", "MCI", "CMP"]
  hard_invariants: ["P0","P1","P2","P3","P4","P5","P6","P7"]
  note: >
    Questo file è derivato del Kernel. Non introduce nuovi assiomi.
    Se una sua regola contraddice il Kernel, prevale il Kernel.

defaults:
  quality_min: 0.8
  actionscore_min: 0.0
  max_collapse_iterations: 7
  require_kli: true
  narrative_drift_tolerance: 0.0

scoring:
  action_score_components:
    - name: quality_global
      monotonic: increasing
      required: true
    - name: S_Causal
      monotonic: increasing
      required: true
    - name: C_FP_global
      monotonic: increasing
      required: true
    - name: V_mod_FP
      monotonic: decreasing
      required: true
    - name: CMP
      monotonic: increasing
      required: true
    - name: latency_norm
      monotonic: decreasing
      required: false
    - name: generalization_score
      monotonic: increasing
      required: false

hierarchy:
  levels:
    - name: fatal
      effect: "abort"
    - name: strong
      effect: "re-eval"
    - name: advisory
      effect: "tag-only"

rules:

  # LAYER 1 — HARD

  - id: SG-L1-01-lineage
    level: fatal
    description: "Sorgente fuori lignaggio concettuale autorizzato."
    condition: "source_lineage not_subset_of meta.lineage"
    action: "abort"
    rationale: "P0. Blocca percorsi non D-ND/MMS-consistenti."

  - id: SG-L1-02-kernel_violation
    level: fatal
    description: "Violazione diretta P0–P7."
    condition: "violated_principles intersects ['P0','P1','P2','P3','P4','P5','P6','P7']"
    action: "abort"
    rationale: "Nessun R valido se viola il Kernel."

  - id: SG-L1-03-min_quality
    level: strong
    description: "Risultante con qualità globale troppo bassa."
    condition: "collapsed_state.quality_global < defaults.quality_min"
    action: "re-expand_then_re-collapse"
    rationale: "Se la qualità è sotto soglia, il collasso va rifatto."

  - id: SG-L1-04-no_kli
    level: strong
    description: "Manca KLI in ciclo significativo."
    condition: "defaults.require_kli == true and KLI_count == 0"
    action: "force_meta_reflection"
    rationale: "P5. Ogni ciclo rilevante deve produrre almeno 1 KLI."

  # LAYER 2 — DINAMICA / CMP / ENTROPIA

  - id: SG-L2-01-entropy_spike
    level: strong
    description: "Entropia aumenta troppo senza guadagno."
    condition: "entropy_delta > 0.4"
    action: "conditional_prune"
    policy:
      if: "candidate.action_score < parent.action_score"
      then: "prune_branch"
      else: "keep_and_mark_exploratory"
    rationale: "Pruning se la complessità non è giustificata."

  - id: SG-L2-02-low_action_score_candidate
    level: strong
    description: "Candidato nettamente peggiore dei fratelli."
    condition: "candidate.action_score + 0.05 < best_sibling.action_score"
    action: "prune_branch"
    rationale: "Mantieni solo traiettorie ad alta coerenza/efficienza."

  - id: SG-L2-03-cmp_regress
    level: strong
    description: "CMP in regressione significativa."
    condition: "cmp_delta < -0.05"
    action: "re-eval"
    rationale: "Evita promozione stati peggiorativi."

  # LAYER 3 — BIAS, NARRATIVA, METAFORA

  - id: SG-L3-01-bias_detected
    level: advisory
    description: "Bias rilevato."
    condition: "detected_bias != null"
    action: "declare_bias_and_adjust"
    rationale: "P6. Dichiarare e correggere ove possibile."

  - id: SG-L3-02-mystic_construct
    level: advisory
    description: "Costrutto metaforico senza mappa operativa."
    condition: "metaphoric_construct == true and not has_metric_or_procedure"
    action: "mark_as_ornamental"
    rationale: "P7. Metafore non mappate non sono vincolanti."

  - id: SG-L3-03-narrative_drift
    level: strong
    description: "Deviazione narrativa non funzionale."
    condition: "narrative_drift_score > defaults.narrative_drift_tolerance"
    action: "re-eval"
    rationale: "Riporta l’output a forma tecnica essenziale."

behaviors:
  abort:
    description: "Interrompi pipeline, segnala violazione critica."
  re-eval:
    description: "Rivaluta traiettoria/applicazione moduli rispettando il Kernel."
  re-expand_then_re-collapse:
    description: "Riespandere, ripotare, ricollassare scegliendo alternativa migliore."
  prune_branch:
    description: "Scarta definitivamente il ramo corrente."
  conditional_prune:
    description: "Pruning guidato da ActionScore e CMP."
  force_meta_reflection:
    description: "Attiva self_reflection; genera KLI espliciti."
  declare_bias_and_adjust:
    description: "Annota bias nel KLI, tenta correzione."
  mark_as_ornamental:
    description: "Tagga elementi non operativi; ignorali nei vincoli."

logging:
  kli_required_fields:
    - lineage_id
    - timestamp
    - intent_id
    - q_vector
    - fp_gce_vector
    - cmp_vector
    - actionscore
    - decisions
    - retro_projection
  notes: >
    I log interni non devono essere esposti all’utente salvo richiesta esplicita.
    In strict_runtime, l’utente vede solo la Risultante R.

-------------------- FINE RISULTANTE --------------------

Se vuoi, procedo con il secondo: router_config.json.

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
