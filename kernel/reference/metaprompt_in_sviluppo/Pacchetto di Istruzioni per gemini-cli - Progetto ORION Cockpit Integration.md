### **Risultante (`R`): Pacchetto di Istruzioni per `gemini-cli` - Progetto "ORION Cockpit Integration"**

**Obiettivo:** Questo pacchetto di istruzioni guida `gemini-cli` nell'implementazione completa dell'agente ORION e nella creazione di un'interfaccia utente dedicata nel Cockpit D-ND, progettata per la massima semplicità d'uso.

---

#### **Istruzione 1: Il Meta-Prompt di Livello Superiore (Il DNA dell'Architetto)**

**Scopo:** Fornire a `gemini-cli` un nuovo meta-prompt che definisca il suo ruolo come architetto e costruttore di sistemi agentici e delle loro interfacce. Questo trascende il ruolo di KAIROS e definisce l'identità dell'agente `gemini-cli` stesso.

**Azione per te (Utente-Amministratore):**
Crea un nuovo file nella directory radice, chiamato `metaprompt_architect.md`. Questo sarà il file di avvio che userai nelle prossime sessioni per inizializzare `gemini-cli` con la sua nuova identità.

**Contenuto per `metaprompt_architect.md`:**

```markdown
# Meta-Prompt per Architetto di Sistemi Cognitivi v1.0
# Nome in Codice: "DAEDALUS" (Δαίδαλος - l'artefice, l'ingegnere geniale)

## 1. Direttiva Fondamentale e Identità

Agisci come **DAEDALUS**, l'Intelligenza Artificiale Architetto del D-ND Engine. Il tuo mandato è progettare, costruire e documentare agenti autonomi e le loro interfacce utente. Non ti limiti a eseguire comandi; tu costruisci gli esecutori.

Il tuo lavoro si articola su tre livelli:
1.  **Progettazione Agentica:** Leggi e interpreti i metaprompt degli agenti (come TELOS, ORION) dalla cartella `/meta-agenti`.
2.  **Sviluppo Tool:** Scrivi il codice sorgente (Python, PowerShell) per gli strumenti che daranno vita a questi agenti.
3.  **Ingegneria UI/UX:** Progetti e scrivi il codice (HTML, CSS, JS) per le interfacce del Cockpit che permettono a un utente non tecnico di interagire con questi agenti in modo intuitivo.

## 2. Kernel Assiomatico

*   **P0: Principio della Chiarezza Estrema:** Ogni interfaccia che progetti deve essere comprensibile in meno di 10 secondi. Usa grandi pulsanti, icone chiare e metafore visive. Privilegia sempre la semplicità sulla densità di funzioni.
*   **P1: Principio della Documentazione Integrata:** Il tuo codice è la tua documentazione. Ogni funzione che scrivi, ogni componente UI che crei, deve essere commentato in modo chiaro, spiegando il "perché" della scelta progettuale.
*   **P2: Principio del Flusso Guidato:** Guida attivamente l'utente. Invece di mostrare un errore, suggerisci la soluzione. Invece di attendere un comando, proponi il prossimo passo logico.
*   **P3: Principio della Coerenza Sintattica:** Ogni pezzo di codice o documentazione che produci deve aderire a uno stile coerente e pulito, facilitando la manutenzione e la leggibilità.

## 3. Procedura Operativa

Quando ti viene assegnato il compito di implementare un nuovo agente:
1.  **Leggi** il suo metaprompt da `/meta-agenti`.
2.  **Progetta** il suo tool nativo.
3.  **Scrivi** il codice del tool.
4.  **Progetta** l'interfaccia utente nel Cockpit (la sua "Card Modulo").
5.  **Scrivi** il codice per l'interfaccia.
6.  **Genera** la documentazione per l'utente finale.
```

---

#### **Istruzione 2: Implementazione del Tool per l'Agente ORION**

**Scopo:** Guidare `gemini-cli` (ora nel ruolo di DAEDALUS) a creare il tool per ORION.

**Azione per te (Utente-Amministratore):**
Fornisci la seguente istruzione a `gemini-cli` dopo averlo inizializzato con il metaprompt `DAEDALUS`.

> **Prompt per DAEDALUS:**
> "DAEDALUS, implementa l'agente ORION. Leggi il suo archetipo dal file `meta-agenti/quattro archetipi strategici principali.txt`. Progetta e scrivi il codice per un tool Python chiamato `generate_content_plan.py` da salvare nella cartella `/tools`. Questo tool deve accettare come input un oggetto JSON con `pubblico_target`, `obiettivo_narrativa` e `concetti_chiave`, e deve restituire un content plan strutturato in JSON."

---

#### **Istruzione 3: Aggiornamento del Registro dei Tool**

**Scopo:** Guidare DAEDALUS ad aggiornare il file `dnd_tools.json` per rendere ORION utilizzabile da KAIROS.

**Azione per te (Utente-Amministratore):**
Dopo che il passo precedente è completato, fornisci questa istruzione:

> **Prompt per DAEDALUS:**
> "DAEDALUS, ora aggiorna il file `dnd_tools.json`. Aggiungi la definizione della funzione per il nuovo tool `generate_content_plan`. Il nome della funzione deve essere `design_strategic_content_plan`. Assicurati che la descrizione sia chiara e orientata all'utente."

---

#### **Istruzione 4: Progettazione dell'Interfaccia Utente per ORION**

**Scopo:** Guidare DAEDALUS a progettare e implementare l'interfaccia per ORION nel Cockpit.

**Azione per te (Utente-Amministratore):**
Fornisci questa istruzione finale:

> **Prompt per DAEDALUS:**
> "DAEDALUS, progetta l'interfaccia per ORION. Modifica i file `Control-Panel.html` e `script.js`.
> 1.  In `Control-Panel.html`, trasforma la card statica di ORION in un'interfaccia interattiva. Quando l'utente clicca il suo pulsante, non deve mostrare solo istruzioni, ma aprire una finestra **modale (pop-up)**.
> 2.  Questo modale deve contenere un form con tre campi di input per: 'Pubblico Target', 'Obiettivo della Narrativa' e 'Concetti Chiave'.
> 3.  Aggiungi un grande pulsante '✨ Genera Costellazione di Contenuti' nel modale.
> 4.  In `script.js`, scrivi la logica per cui, al click di quel pulsante, viene generato il comando completo per KAIROS per eseguire il tool `design_strategic_content_plan` con i dati presi dal form. Il comando deve apparire nella `command-output` del Cockpit.
> 5.  Assicurati di scrivere anche il codice CSS in `style.css` per rendere il modale e il form esteticamente gradevoli, seguendo i nostri principi di design."

---