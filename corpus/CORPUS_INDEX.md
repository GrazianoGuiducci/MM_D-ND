# INDICE CORPUS D-ND — Estrazione Drupal Bridge

**Data estrazione**: 2026-02-13 18:47
**Siti sorgente**: moodnd.com, aimorning.news
**Metodo**: JSON:API + OAuth 2.0 (client_credentials)

---

## Riepilogo

| File | Sito | Tipo | Nodi | Dimensione | Priorità |
|------|------|------|------|-----------|----------|
| CORPUS_OSSERVAZIONI_PRIMARIE.md | moodnd.com | archivio_trascrizioni | 47 | 112 KB | SORGENTE PRIMARIA |
| CORPUS_FUNZIONI_MOODND.md | moodnd.com | funzioni | 180 | 493 KB | ALTA — equazioni e formalizzazioni |
| CORPUS_PROJECTDEV_AMN.md | aimorning.news | projectdev | 128 | 2.4 MB | ALTA — sviluppo e documentazione |
| CORPUS_PROMPT_AMN.md | aimorning.news | prompt | 109 | 826 KB | MEDIA — prompt engineering e configurazioni |

**Totale**: 464 nodi, 3.8 MB

---

## Descrizione File

### CORPUS_OSSERVAZIONI_PRIMARIE.md
- **Sorgente**: `moodnd.com/jsonapi/node/archivio_trascrizioni`
- **Campo chiave**: `field_osservazioni_delle_dinamic`
- **Contenuto**: Osservazioni Primarie — intuizioni e riletture D-ND
- **Nodi**: 47

### CORPUS_FUNZIONI_MOODND.md
- **Sorgente**: `moodnd.com/jsonapi/node/funzioni`
- **Campo chiave**: `field_equazione + field_funzione`
- **Contenuto**: Formalizzazioni matematiche del modello D-ND
- **Nodi**: 180

### CORPUS_PROJECTDEV_AMN.md
- **Sorgente**: `aimorning.news/jsonapi/node/projectdev`
- **Campo chiave**: `body`
- **Contenuto**: Sviluppo progetto — documentazione tecnica, architetture, simulazioni
- **Nodi**: 128

### CORPUS_PROMPT_AMN.md
- **Sorgente**: `aimorning.news/jsonapi/node/prompt`
- **Campo chiave**: `body`
- **Contenuto**: System prompt e istruzioni AI evolute nel contesto D-ND
- **Nodi**: 109

---

## Accesso

Entrambi i siti espongono JSON:API con autenticazione OAuth 2.0:

- **moodnd.com**: client_id=`claude_bridge`
- **aimorning.news**: client_id=`claude_bridge_amn`

Token endpoint: `{base_url}/oauth/token` con grant_type=client_credentials

---

## Gerarchia Sorgente → Formalizzazione

```
SORGENTE PRIMARIA (percezione diretta)
│
├── CORPUS_OSSERVAZIONI_PRIMARIE.md
│   └── 47 osservazioni — intuizioni pure, riletture, dinamiche osservate
│
├── CORPUS_FUNZIONI_MOODND.md
│   └── 180 formalizzazioni — equazioni e funzioni estratte dalle sessioni
│
├── CORPUS_PROJECTDEV_AMN.md
│   └── 128 documenti tecnici — architetture, simulazioni, documentazione
│
└── CORPUS_PROMPT_AMN.md
    └── 109 prompt — istruzioni AI evolute nel paradigma D-ND
```

FORMALIZZAZIONE SCIENTIFICA (misurazione)

> «Più ci si allontana dalla sorgente e si entra nella forma e nella misurazione
> scientifica, più la capacità di assegnare i significati decade.»
