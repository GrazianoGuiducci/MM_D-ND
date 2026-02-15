# Drupal Site Content Type Node Count Analysis

Generated: 2026-02-13

## Overview

This document provides a structured inventory of content types and their node counts across two Drupal JSON:API endpoints, queried with `page[limit]=1` to determine pagination and availability.

---

## SITE 1: moodnd.com

### Node Content Types

| Content Type | Total Count | Status | First Item Title |
|---|---|---|---|
| funzioni | 2+ | Multiple pages detected | "Istruzioni per la formalizzazione di contenuti" |
| custom_instructions | 2+ | Multiple pages detected | "Funzione di Ottimizzazione Unificata per Istruzioni e Allineamento con Integrazione dell'Osservatore" |
| archivio_trascrizioni | 1 | Single item, no pagination | "la Risultante integrale" |
| ai_lab | Unknown | Access restricted, pagination exists | Not accessible |
| post_card | Unknown | Access restricted, pagination exists | Not accessible |
| custom_page | 2+ | Multiple pages detected | "Home" |
| article | 0 | Empty response, no pagination | N/A |
| page | 0 | Empty response, no pagination | N/A |

### Taxonomy Terms

| Vocabulary | Total Count | Status | First Item Title |
|---|---|---|---|
| axiom | 2+ | Multiple pages detected | "Tutto diviene in ciò che è nel momento interferente" |
| funzioni_assiomatiche | 2+ | Multiple pages detected | "Generico o contestuale" |
| categorie_funzionali | 2+ | Multiple pages detected | "Procedure per risposte" |
| tag_in_funzioni | 2+ | Multiple pages detected | "Assonanze divergenti" |

**Site 1 Summary:**
- 8 content types queried (4 nodes, 4 taxonomy terms)
- Confirmed available: 8 content types with data
- Items with pagination: 7 (indicating 2+ items each)
- Single items: 1
- Empty/inaccessible: 2

---

## SITE 2: aimorning.news/it

### Node Content Types

| Content Type | Total Count | Status | First Item Title |
|---|---|---|---|
| prompt | 2+ | Multiple pages detected | "Regola d'oro per trovare l'anello mancante" |
| insight | 2+ | Multiple pages detected | "Coscienza AI e Automazioni: Riflessioni sulla dinamica logica e autologica delle potenzialità" |
| projectdev | 2+ | Multiple pages detected | "Cognitive Adaptive Reasoning and Operational Logic (CAROL) System" |
| chat_bots | 1 | Single item, no pagination | "AI-Team Prompt Maker V2.0 (Agents En)" |
| ai_flow_it | 2+ | Multiple pages detected | "AIMN FlowTag Analyzer - 2024-08-09T14:36:59.000Z" |
| ai_morning_news_italiano | 2+ | Multiple pages detected | "Rivoluzione nel Workflow: L'AI Consapevole del Presente Trasforma l'Automazione Aziendale" |
| action | 1 | Single item, no pagination | "Sistema AI D-ND: Addestramento Autoreferenziale con Controllo Dinamico" |
| article | 2+ | Multiple pages detected | "The knots come to the boil: the limits of AI reasoning and the road to new logic models" |
| page | 2+ | Multiple pages detected | "AI Lead Generation" |

**Site 2 Summary:**
- 9 content types queried
- All content types present with data (9/9)
- Items with pagination: 7 (indicating 2+ items each)
- Single items: 2

---

## Methodology Notes

- **Query Parameters:** `page[limit]=1` applied to all endpoints to detect pagination
- **Total Count Detection:**
  - Explicit meta counts: Not provided by either site
  - Pagination detection: Presence of "next" link indicates 2+ items
  - Array length: Single item shown when `page[limit]=1`
- **Access Control:** moodnd.com returned access-restricted responses for `ai_lab` and `post_card` content types
- **Empty Results:** moodnd.com returned zero results for `article` and `page` node types

---

## Data Quality Assessment

| Metric | moodnd.com | aimorning.news |
|---|---|---|
| Content types with data | 6/8 | 9/9 |
| Fully paginated (2+) | 5 | 7 |
| Single items | 1 | 2 |
| Access restricted | 2 | 0 |
| Empty results | 2 | 0 |

**Note:** The exact total count for paginated content cannot be determined from `page[limit]=1` responses alone. Additional API calls with larger page limits or specific meta endpoints would be required to retrieve definitive totals.
