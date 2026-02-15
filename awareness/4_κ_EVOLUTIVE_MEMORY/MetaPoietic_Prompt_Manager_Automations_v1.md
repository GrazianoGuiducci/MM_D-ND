# Meta‑Poietic Automations — Prompt Manager v1

Data: 2025-09-01  
Dominio: Meta_System_Coder — Sistema Metapoietico per Prompt/DOMANDE  
Stato: SPECIFICA OPERATIVA (prima implementazione guidata)

Premessa e Principi
- Intenzione: minimizzare il carico cognitivo di utenti/admin. Il sistema si auto‑classifica, si auto‑organizza, si auto‑ottimizza, e propone azioni (publish/unpublish/merge/curation) secondo segnali d’uso e qualità.
- Pattern di riferimento già presenti:
  - Onboarding (Genesis/Assistenti) → generazione guidata e inferenza di metadati.
  - Presets “visibili” → pubblicazione verso utenti.
  - TrashCenter → soft‑delete/restore/purge.
  - Pinner/Metapinner → sintesi/derivazione di conoscenza.

Obiettivi funzionali (automazioni)
- Auto‑tagging: estrarre e aggiornare tags coerenti (purpose:*, kernel:*, coac:*, style:*, scope:*) senza input manuale.
- Auto‑purpose/scope/style: inferenza semantica dei campi principali.
- Auto‑dedup/merge: individuare duplicati o “quasi duplicati”, proporre merge o marcare canonical.
- Auto‑publish recommendation: suggerire (o attivare con policy) pubblicazione su base impact/usage/qualità.
- Auto‑deprecate/decay: de‑prioritizzare o archiviare elementi obsoleti o a basso impatto con spiegazione.
- Auto‑cluster: raggruppare prompt in temi/coorti (cluster_id) per navigazione e curation.
- Auto‑telemetria/impact: calcolo continuo di usage_count, last_used, impact/catalytic_potential, entropy_delta.
- Auto‑security lint: prevenzione di segreti/PII nei prompt pubblicati (lint e suggerimenti di riscrittura).
- Auto‑scope per dominio: associare domain_id quando rilevante; preferire suggerimenti del dominio attivo.

Architettura (event‑driven pipeline)
- Eventi (fonte): create_prompt, update_prompt, soft_delete_prompt, restore_prompt, use_prompt, send_message_via_prompt, publish_toggle, cluster_rebuild_request.
- Coda/lavoratori:
  - RetagWorker: inferenza metadati (tags/purpose/style/scope) + aggiornamento confidenza.
  - EmbeddingWorker: vettorizzazione testo + similarità (cosine) per dedup/cluster.
  - QualityWorker: calcolo impact/score con segnali (pinned, accepted, led_to_R, reply_feedback).
  - GovernanceWorker: regole publish/unpublish suggestion secondo soglie e guardrails.
  - SecurityWorker: lint segreti/PII e suggerimenti di fix.
- Idempotenza: ogni worker idempotente per evento (dedup su event_id).
- Storage: colonne aggiuntive e audit log per decisioni automatiche.

Estensioni Schema Dati (DB)
- archived_prompts (esteso oltre il DOA/ADR):
  - inferred_tags: JSONB (es. ["purpose:funzione","coac:MiniPlan"])
  - inferred_purpose: TEXT[] NULL
  - inferred_style: VARCHAR(16) NULL
  - inferred_scope: VARCHAR(32) NULL
  - inference_confidence: JSONB { tags:0..1, purpose:0..1, style:0..1, scope:0..1 }
  - canonical_hash: VARCHAR(64) (hash normalizzato di text+tags+p urpose per dedup soft)
  - embedding: VECTOR / BYTEA / JSONB (a seconda del supporto DB)
  - cluster_id: UUID NULL
  - source: VARCHAR(16) ('user'|'admin'|'system') default da author_role; aggiornato quando “sintetico”
  - auto_flags: JSONB { recommended_publish:boolean, recommended_unpublish:boolean, deprecated:boolean }
  - published_by: INT NULL, published_at: TIMESTAMP NULL
  - domain_id: INT NULL (scoping)
- Indici:
  - idx_ap_canonical_hash, idx_ap_cluster, idx_ap_domain, GIN su inferred_tags

Contratti API (aggiunte “meta‑poietiche”)
- POST /agents/questions/archive/{id}/retag → (async) avvia RetagWorker (ritorna {status:"queued"}).
- POST /agents/questions/archive/{id}/cluster → (async) ricalcola embedding + clustering locale.
- GET /agents/questions/archive/{id}/explanations → motivazioni di inferenze/decisioni (governance transparency).
- GET /agents/questions/archive/recommendations?type=publish|unpublish|merge → suggerimenti batch per admin.
- POST /agents/questions/archive/{id}/accept-recommendation → accetta (merge/publish/unpublish/rename tags).
- POST /agents/questions/archive/rebuild-clusters → ricostruzione cluster globale (admin).
- PATCH esteso /agents/questions/archive/{id}:
  - fields: accept_inferred (boolean) per promuovere inferred_* → campi principali.

Regole di inferenza (sketch)
- Tagger (LLM guided, ontology‑aware):
  - prompt di sistema usa una tassonomia “aperta ma normata” (purpose/scope/style/kernel/coac) con out JSON. Valori fuori dizionario mappati a “free:tag”.
- Purpose/Scope/Style:
  - Individuati dal contesto lessicale, con fallback a default (“funzione”, “chat”, “expansive”).
- Dedup/Merge:
  - Similarità > 0.92 (embedding) + canonical_hash match ⇒ “probabile duplicato”.
  - Similarità 0.85–0.92 ⇒ “correlato forte” (suggerire cluster/coalescenza manuale).
- Publish Recommendation:
  - Soglie esemplificative (tunable):
    - usage_count ≥ 8 AND impact ≥ 0.30 AND led_to_R ≥ 1 AND security_lint_ok ⇒ recommended_publish=true.
  - Unpublish:
    - usage_count = 0 negli ultimi N giorni AND impact < 0.05 ⇒ recommended_unpublish.
- Decay/Deprecation:
  - Deprecation soft: abbassa ranking in suggest; non rimuove.
- Security Lint:
  - Regex/API su pattern credenziali/PII; se fallisce publish ⇒ blocco o “needs_fix” con proposta di riscrittura.

Integrazioni UI (zero attrito)
- Tab Prompt:
  - Badges “Auto” su metadati inferiti; “Promuovi” (1‑click) per accettare inferenze.
  - “Suggerimenti” (pane laterale) per Publish/Unpublish/Merge con motivazioni (GET /explanations).
  - Stato attivo: mostra cluster e correlati.
- QuestionSelector:
  - Ranking spinto da impact/recency + domain scope; badge “Admin” per pubblicati.
  - Azione “Salva come personale (auto‑tag)” genera entry con inferenze subito applicate.
- Admin:
  - Viste “Raccomandazioni” e “Cluster” per curazione zero‑attrito.

Node/Agent design (riuso kernel)
- RelevanceSelectorAgent: aiuta a mappar e scope/domains + file selezionati.
- Pinner/Metapinner: sintetizza varianti e può proporre consolidamenti/temi.
- “TaggerAgent”: LLM con prompt di ontologia; usa esempi (few‑shot) + JSON schema.

Governance e Trasparenza
- Ogni automazione genera un log con timestamp, input, output, confidenza, e link all’oggetto.
- Admin può impostare policy: auto‑publish “soft” (richiede conferma) o “hard” (diretta) per cluster ad alto impact.
- Rate‑limit/quotas: batch notturni per cluster/embeddings; eventi near‑real‑time per retag su create/update.

KPI e Telemetria
- Coverage inferenza (quanti item hanno inferred_*).
- Delta tempo “idea → riuso in chat” (attrito).
- % merge/duplicati ridotti; tasso accettazione raccomandazioni.
- Qualità suggerimenti (CTR su “Usa/Invia Subito” per item auto‑classificati vs manuali).

Rollout a fasi
- Fase 1 (safe): RetagWorker + SecurityWorker + aggiornamento suggest (ranking); UI badges + “Promuovi inferenze”.
- Fase 2: EmbeddingWorker + dedup soft + cluster locali; pannello “Correlati”.
- Fase 3: GovernanceWorker (recommend publish/unpublish) + vista Admin “Raccomandazioni”.
- Fase 4: Cluster globale + rebuild periodico + policy auto‑publish opzionale.

Acceptance (metapoiesi minima)
- L’utente crea un prompt senza toccare tags: il sistema inferisce purpose/tags/scope/style, mostra badges e li promuove su 1‑click.
- L’admin non etichetta “a mano”: riceve una lista “Pubblica?” con motivazione; può accettare in blocco.
- Il QuestionSelector migliora organically (ranking/label/cluster) man mano che l’uso cresce, senza micro‑gestione.
