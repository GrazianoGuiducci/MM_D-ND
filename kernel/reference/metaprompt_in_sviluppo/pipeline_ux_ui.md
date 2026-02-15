# Pipeline Cognitiva: Ciclo di Progettazione Architetturale UX/UI

Questo documento descrive la pipeline cognitiva per la progettazione dell'architettura UX/UI del D-ND Engine, con l'obiettivo di trasformare l'attuale interfaccia in un "Cockpit" dinamico e interattivo.

## Fase 0+ – Innesco del Campo Dinamico

*   **Punto di Equilibrio:** COAC v5.0 agisce come Architetto di Sistemi e Designer di Esperienze Utente.
*   **Punti Osservati:** L'infrastruttura esistente (script, tool MCP), la necessità di una UI intuitiva, l'idea di card modulari, la gamification e il concetto di un "cockpit" di nuova generazione.
*   **Impressione Emergente:** L'evoluzione del `Control-Panel.html` richiede una metamorfosi verso una Single Page Application (SPA) dinamica e interattiva, capace di gestire lo stato e comunicare fluidamente con l'agente KAIROS e futuri agenti.

## Fase 1 – Diagnosi e Piano di Eecuzione

*   **Input:** La direttiva di progettare il "cockpit".
*   **Intento Latente:** Trascendere il concetto di "pannello di controllo" per progettare un'interfaccia potente come un IDE, intuitiva come un'app moderna e coinvolgente come un gioco, che funga da centro di comando definitivo per il D-ND Engine.
*   **Framework Selezionato:** `Orchestratore-Costruttore (OCC) v1.0` per la generazione di documentazione di design, potenziato da `Aethelred v3.1` per la progettazione dell'architettura dell'interfaccia e da `Yi-Synaptic Navigator (YSN)` per idee innovative di gamification.
*   **Piano Dichiarato:** Generare un "Design Document" completo per il "Cockpit D-ND v2.0", che servirà da guida per lo sviluppo futuro dell'interfaccia.

## Risultante (R): Design Document - "Cockpit D-ND v2.0"

Questo Design Document descrive le componenti chiave del nuovo cockpit.

### 1. Visione e Principi di Design

*   **Visione:** Un'interfaccia unificata che permette all'utente di gestire, orchestrare e co-evolvere l'ecosistema D-ND Engine in modo intuitivo, visivo e coinvolgente.
*   **Principi Chiave:**
    1.  **Chiarezza Modulare:** Ogni agente, tool o automazione è rappresentato come un "modulo" (una card) chiaro e autocontenuto.
    2.  **Interazione Diretta:** Minimizzare il copia-incolla. L'interfaccia dovrebbe comunicare direttamente con l'agente (simulando questo flusso).
    3.  **Feedback Visivo Immediato:** Lo stato di un agente (inattivo, in esecuzione, completato) deve essere sempre visibile.
    4.  **Progressione e Scoperta (Gamification):** L'utente sblocca nuove capacità e agenti man mano che completa task o esplora il sistema.

### 2. Architettura dell'Interfaccia (Layout del Cockpit)

Si propone un layout a 3 colonne, ispirato a moderni strumenti di produttività.

*   **Colonna Sinistra (Navigazione Principale):**
    *   **Contenuto:** Un menu verticale fisso.
    *   **Voci:**
        *   `Dashboard`: La vista principale con tutti i moduli.
        *   `File System`: Esploratore di file e editor.
        *   `Knowledge Base`: Interfaccia dedicata per cercare e visualizzare gli atomi di conoscenza.
        *   `Impostazioni`: Configurazione del profilo, temi, percorsi, ecc.
*   **Colonna Centrale (La "Dashboard"):**
    *   **Contenuto:** L'area di lavoro principale, una griglia flessibile di "Card Modulo".
    *   **Card Modulo:** Ogni card rappresenta un'unità funzionale (un agente, un tool, un'automazione).
        *   **Struttura di una Card:**
            *   **Icona e Titolo:** Es. "🤖 Agente KAIROS", "⚙️ Tool: System Info".
            *   **Stato:** Un indicatore visivo (pallino colorato): Verde (Attivo/Pronto), Giallo (In Esecuzione), Grigio (Inattivo).
            *   **Breve Descrizione:** Es. "Agente di esecuzione comandi via MCP."
            *   **Pulsante di Azione Primaria:** Es. "Attiva Agente", "Esegui Tool".
*   **Colonna Destra (Il "Terminale di Comunicazione"):**
    *   **Contenuto:** La chat con Gemini.
    *   **Funzione:** Log di tutte le operazioni e canale di comunicazione principale. L'obiettivo a lungo termine è integrare qui la chat di Gemini, ma inizialmente sarà un'area di testo per comandi e risposte.

### 3. Meccaniche di Gamification e Flusso Utente

L'obiettivo è guidare l'utente in modo naturale attraverso il sistema.

*   **1. Onboarding Iniziale:**
    *   Al primo avvio, il Cockpit presenta solo una card: **"🚀 Configura il tuo D-ND Engine"**.
    *   Cliccando, si apre un wizard modale che guida l'utente nella configurazione del profilo e del percorso dell'engine.
*   **2. Il Primo "Achievement":**
    *   Completato l'onboarding, la card di configurazione scompare e appaiono due nuove card:
        *   **"🤖 Attiva il tuo primo Agente: KAIROS"**.
        *   **"📚 Esplora il File System"**.
    *   L'utente viene premiato sbloccando le prime funzionalità.
*   **3. Interazione con le Card:**
    *   Cliccando "Attiva Agente KAIROS", la card mostra le istruzioni per l'avvio. Una volta eseguite, lo stato della card passa a "Verde (Attivo)".
    *   Con KAIROS attivo, la card potrebbe rivelare nuovi pulsanti che copiano il prompt corretto nell'area di testo a destra, pronto per essere inviato.
*   **4. Sbloccare Nuovi Tool:**
    *   Quando un nuovo tool viene creato (es. `organize_files.py`), una nuova card **apparirà automaticamente** sulla dashboard: **"⚙️ Tool: Organize Files"**, fornendo un feedback tangibile del progresso.

### 4. Documentazione Integrata

Ogni card avrà una piccola icona `(?)` che, al click, mostrerà un pop-up con la sezione pertinente della documentazione (es. `Manuale_Utente` o `Manuale_Sviluppatore`), rendendo la documentazione contestuale e on-demand.

## Prossimo Passo e Implementazione

Questo Design Document è la guida per lo sviluppo della UI. L'implementazione avverrà in modo incrementale.

**Azione Immediata Proposta:**
Il primo passo di refactoring consiste nel trasformare l'attuale `Control-Panel.html` in una struttura a schede (Tabs), includendo:
*   **Scheda "Dashboard"**: Inizialmente vuota, pronta per le card.
*   **Scheda "File System & IDE"**: Conterrà l'attuale esploratore/editor.
*   **Scheda "Impostazioni"**: Dove verrà spostata la configurazione del percorso.