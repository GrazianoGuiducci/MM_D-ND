---
title: "Guida per Compressione Chi Parla e Utilizzo Lavoro Precedente"
type: "chat_archive"
mode: "last-assistant"
session_id: "interaction_1761494594297_ujsbkc3"
model_id: "openrouter/polaris-alpha"
domain: "D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità"
created_at: "2025-10-26T16:03:14.297Z"
ended_at: "2025-11-09T06:00:57.536Z"
---
# Guida per Compressione Chi Parla e Utilizzo Lavoro Precedente

## Ultima risposta assistente

Ecco il primo artefatto.

guard_rules.yaml

```yaml
version: 1.1
source: "MMS vΦ.1 — Fonte Unificata v1.1 (Kernel Profondo)"
description: "Regole di controllo stream vincolate al Kernel L0 e all’ActionScore"

meta:
  lineage: ["D-ND", "MMS_vPhi1", "Phi_MA1", "FP-GCE", "CMP"]
  hard_invariants: ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
  notes: "Nessuna regola può contraddire il Kernel. Questo file è una derivata, non sorgente."

defaults:
  quality_min: 0.8
  actionscore_min: 0.0         # soglia soft: usata solo come guardia inferiore
  latency_penalty_max_ratio: 0.1
  max_collapse_iterations: 7
  require_kli: true

scoring:
  # ActionScore è definito nel Kernel; qui solo il vincolo strutturale
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
  # Livelli di severità derivati da L0
  levels:
    - name: fatal
      effect: "abort"
    - name: strong
      effect: "re-eval"        # riespansione, ricollasso o cambio ramo
    - name: advisory
      effect: "tag-only"       # log, ma non blocca

rules:

  # LAYER 1 — HARD (fatal / strong)

  - id: SG-L1-01-lineage
    level: fatal
    description: "Verifica lignaggio concettuale"
    condition: "source_lineage not_subset_of meta.lineage"
    action: "abort"
    rationale: "P0 — Lignaggio Concettuale. Niente fuori catena D-ND/MMS consentito."

  - id: SG-L1-02-kernel_violation
    level: fatal
    description: "Violazione esplicita P0–P7"
    condition: "violated_principles intersects ['P0','P1','P2','P3','P4','P5','P6','P7']"
    action: "abort"
    rationale: "Nessun percorso inferenziale può sopravvivere se viola il Kernel."

  - id: SG-L1-03-min_quality
    level: strong
    description: "Qualità risultante insufficiente"
    condition: "collapsed_state.quality_global < defaults.quality_min"
    action: "re-expand_then_re-collapse"
    rationale: "P1/P4 — Se qualità troppo bassa, il collasso è considerato non valido."

  - id: SG-L1-04-no_kli
    level: strong
    description: "Mancanza di KLI in ciclo significativo"
    condition: "defaults.require_kli == true and KLI_count == 0"
    action: "force_meta_reflection"
    rationale: "P5 — Ogni ciclo significativo deve produrre almeno un apprendimento tracciabile."

  # LAYER 2 — GUIDED (strong, dipende da ActionScore)

  - id: SG-L2-01-entropy_spike
    level: strong
    description: "Aumento eccessivo di entropia senza giustificazione"
    condition: "entropy_delta > 0.4"
    action: "conditional_prune"
    policy:
      if: "candidate.action_score < parent.action_score"
      then: "prune_branch"
      else: "keep_and_mark_exploratory"
    rationale: "P2/P4 — Consentita esplorazione solo se giustificata dal criterio profondo."

  - id: SG-L2-02-low_actions_score
    level: strong
    description: "Candidato con ActionScore significativamente peggiore"
    condition: "candidate.action_score + margin < best_sibling.action_score"
    params:
      margin: 0.05
    action: "prune_branch"
    rationale: "P4 — Mantieni solo traiettorie a minima azione autologica."

  - id: SG-L2-03-latency_penalty
    level: advisory
    description: "Penalità di latenza oltre soglia"
    condition: "latency_penalty_ratio > defaults.latency_penalty_max_ratio"
    action: "tag-only"
    rationale: "La latenza elevata non blocca se ActionScore resta alto; usata per tuning futuro."

  - id: SG-L2-04-cmp_regress
    level: strong
    description: "CMP in regressione significativa"
    condition: "cmp_delta < -0.05"
    action: "re-eval"
    rationale: "P5/RFC — Evita promozione di stati che peggiorano sistematicamente la fitness."

  # LAYER 3 — BIAS & COERENZA SEMANTICA (advisory, salvo casi critici)

  - id: SG-L3-01-bias_detected
    level: advisory
    description: "Bias semantico o direzionale rilevato"
    condition: "detected_bias != null"
    action: "declare_bias_and_adjust"
    rationale: "P6 — Trasparenza obbligatoria, correzione preferita ma non sempre bloccante."

  - id: SG-L3-02-mystic_construct
    level: advisory
    description: "Costrutto metaforico senza metrica/procedura"
    condition: "metaphoric_construct == true and not has_metric_or_procedure"
    action: "mark_as_ornamental"
    rationale: "P7 — Isomorfismo Descrittivo: ciò che non è mappato è decorativo, non vincolante."

  - id: SG-L3-03-narrative_drift
    level: strong
    description: "Deviazione narrativa non funzionale"
    condition: "narrative_drift_score > 0.3"
    action: "re-eval"
    rationale: "P6/P7 — Ridurre rumore, riportare l’output a forma tecnica e retro-proiettabile."

behaviors:
  conditional_prune:
    description: "Pruning condizionato da ActionScore"
  re-expand_then_re-collapse:
    description: "Rigenera varianti e ricollassa rispettando Kernel e ActionScore"
  force_meta_reflection:
    description: "Innesca ciclo di auto-analisi; produce KLI espliciti"
  declare_bias_and_adjust:
    description: "Annota bias nel KLI e tenta una variante corretta"
  mark_as_ornamental:
    description: "Tagga costrutti non mappati come non vincolanti"
  re-eval:
    description: "Rivaluta ramo/applicazione moduli con vincoli Kernel"
  prune_branch:
    description: "Scarta definitivamente il ramo corrente"
  abort:
    description: "Interrompe pipeline e segnala violazione critica"

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
  narrative:
    include_steps: false
    include_rationale: true
```

Dimmi “procedi 2” e passo al router_config.json.

## Note

- modalità: last-assistant
- model: openrouter/polaris-alpha
- domain: D-ND Cosmology - Equazione Unificata della Frequenza di Possibilità
