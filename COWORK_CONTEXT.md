# COWORK_CONTEXT.md — Risposta dall'Audit

## Decisioni (da Cowork dopo audit completo)

### Stato Audit
- Totale attuale: ~175MB, 2122 file
- `tools/` da solo: 106MB (61%)
- `tools/D-ND Workflow/`: 927 file (922 .docx, **239 duplicati con (1)**), 41MB
- 26 file con nomi > 100 caratteri (problematici per git Windows)

### Risposte alle 6 Domande

**1. `archive/` (1.5MB)** → ✅ TIENI
Leggero, dà tracciabilità. Draft paper, report sessioni, simulazioni documentano l'evoluzione.

**2. `tools/APP/D-ND-Omega-Kernel/`** → ❌ ESCLUDI
Ha repo propria: `github.com/GrazianoGuiducci/D-ND-Omega-Kernel`
Opzione: aggiungi a `.gitignore` oppure converti in git submodule.

**3. `tools/APP/D-ND_semantic_OS/`** → ❌ ESCLUDI
Ha repo propria: `github.com/GrazianoGuiducci/D-ND_semantic_OS`
Stessa logica: `.gitignore` o submodule.

**4. `tools/D-ND Workflow/`** → ❌ ESCLUDI
41MB di 927 .docx con 239 duplicati. Materiale di lavoro storico.
Resta solo in locale. Se pezzi specifici servono → estrarre nel corpus.

**5. `tools/crud_publish/` e `sql_migrate/`** → 🗑️ RIMUOVI
Mock vuoti, nessun valore operativo.

**6. `images/` (22MB)** → ✅ TIENI
Dati fenomenologici primari. Le immagini SONO il modello — arrivate prima della formalizzazione.

### Azione per .gitignore

Aggiungi al `.gitignore`:
```
# Sub-repos con propria GitHub repo
tools/APP/D-ND-Omega-Kernel/
tools/APP/D-ND_semantic_OS/

# Storico di lavoro (solo locale)
tools/D-ND Workflow/

# Mock non implementati
tools/crud_publish/
tools/sql_migrate/
```

### Risultante Repo Pulita
```
PRIMA:  ~175MB, 2122 file
DOPO:   ~33MB, ~250 file

domain_D-ND_Cosmology/
├── KERNEL_SEED.md              ← Seme Autoinstallante
├── SENTINEL_STATE.md           ← Stato del Campo
├── COWORK_CONTEXT.md           ← Questo file (può essere rimosso dopo il push)
├── .gitignore
├── kernel/         836KB       ← Kernel MM v1.0 + reference
├── method/         380KB       ← Le Leggi del Metodo (10 file)
├── corpus/         3.9MB       ← Materiale sorgente D-ND
├── papers/         712KB       ← 7 paper accademici + latex + figures
├── awareness/      4.8MB       ← Documenti ontologici
├── images/         22MB        ← 7 immagini fenomenologiche
├── archive/        1.4MB       ← Tracciabilità storica
├── tools/          ~4KB        ← README.md (le APP hanno le loro repo)
└── .claude/skills/ ~20KB       ← Skill Claude Code (sentinel-sys, seed-deploy)
```

### Note per il Coder
- I 26 file con nomi > 100 char: la maggior parte sono dentro `tools/D-ND Workflow/` che viene escluso. Verificare se ne restano in `awareness/` o `kernel/reference/`.
- Il `.git/` interno a `tools/APP/` era già stato rimosso — conferma che `.gitignore` è sufficiente per l'esclusione.
- `COWORK_CONTEXT.md` può restare nella repo come documentazione del processo decisionale, oppure rimuoverlo dopo il primo push — a discrezione.
- La cartella `.claude/skills/` va inclusa: contiene `sentinel-sys` e `seed-deploy` che viaggiano con la repo per rendere il kernel autoinstallante in Claude Code.
