# 03_Cockpit_Implementation_Guide.md
# Guida all'Implementazione del Cockpit D-ND Engine v2.0

Questo documento unifica la visione architettonica del Cockpit D-ND Engine v2.0 con le istruzioni pratiche per la sua implementazione, focalizzandosi sull'integrazione modulare e l'interazione con gli agenti.

---

## 1. Visione e Principi di Design del Cockpit D-ND v2.0

Il Cockpit D-ND v2.0 è concepito come una Single Page Application (SPA) dinamica e interattiva, capace di gestire lo stato e comunicare fluidamente con gli agenti (come KAIROS) e i tool. L'obiettivo è trascendere il concetto di "pannello di controllo" per progettare un'interfaccia potente come un IDE, intuitiva come un'app moderna e coinvolgente come un gioco, che funga da centro di comando definitivo per il D-ND Engine.

### Principi Chiave:
1.  **Chiarezza Modulare:** Ogni agente, tool o automazione è rappresentato come un "modulo" (una card) chiaro e autocontenuto.
2.  **Interazione Diretta:** Minimizzare il copia-incolla. L'interfaccia dovrebbe comunicare direttamente con l'agente (simulando questo flusso).
3.  **Feedback Visivo Immediato:** Lo stato di un agente (inattivo, in esecuzione, completato) deve essere sempre visibile.
4.  **Progressione e Scoperta (Gamification):** L'utente sblocca nuove capacità e agenti man mano che completa task o esplora il sistema.

### Architettura dell'Interfaccia (Layout del Cockpit)
Si propone un layout a 3 colonne, ispirato a moderni strumenti di produttività:

*   **Colonna Sinistra (Navigazione Principale):** Menu verticale fisso con voci come `Dashboard`, `File System`, `Knowledge Base`, `Impostazioni`.
*   **Colonna Centrale (La "Dashboard"):** L'area di lavoro principale, una griglia flessibile di "Card Modulo". Ogni card rappresenta un'unità funzionale (agente, tool, automazione).
*   **Colonna Destra (Il "Terminale di Comunicazione"):** La chat con Gemini, log di tutte le operazioni e canale di comunicazione principale.

### Meccaniche di Gamification e Flusso Utente
L'obiettivo è guidare l'utente in modo naturale attraverso il sistema, con un onboarding iniziale, sblocco di funzionalità e feedback tangibile del progresso.

### Documentazione Integrata
Ogni card avrà una piccola icona `(?)` che, al click, mostrerà un pop-up con la sezione pertinente della documentazione.

---

## 2. Implementazione della Card Agente KAIROS (Esempio Pratico)

Questa sezione fornisce un esempio pratico di come implementare una "Card Modulo" per l'Agente KAIROS all'interno della struttura a schede del Cockpit D-ND v2.0.

### 2.1. Aggiornamento dello Stile (`style.css`)

**Scopo:** Aggiungere le classi CSS per definire l'aspetto delle card e i loro indicatori di stato. Queste classi si integrano con il tema generale del Cockpit.

```css
/* --- Module Cards on Dashboard --- */
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
}

.module-card {
    background-color: var(--bg-tertiary);
    border-radius: var(--border-radius);
    padding: 20px;
    border: 1px solid var(--border-color);
    transition: all var(--transition-speed);
    display: flex;
    flex-direction: column;
}

.module-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
}

.card-header h3 {
    margin: 0;
    font-size: 1.2em;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #bdc3c7; /* Grigio (Inattivo) */
}

.status-indicator.active {
    background-color: #2ecc71; /* Verde (Attivo) */
    box-shadow: 0 0 10px #2ecc71;
}

.status-indicator.running {
    background-color: #f1c40f; /* Giallo (In Esecuzione) */
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 5px #f1c40f; }
    50% { box-shadow: 0 0 15px #f1c40f; }
    100% { box-shadow: 0 0 5px #f1c40f; }
}

.card-body p {
    margin: 0 0 20px 0;
    color: var(--text-secondary);
    flex-grow: 1;
}

.card-footer button {
    width: 100%;
}
```

### 2.2. Aggiornamento della Struttura (`Control-Panel.html`)

**Scopo:** Aggiungere il placeholder per la card di KAIROS all'interno della sezione "Dashboard" del `Control-Panel.html`, rispettando la nuova struttura a schede.

```html
<div id="Dashboard" class="tab-content active">
    <h2>Dashboard dei Moduli</h2>
    <div class="dashboard-grid">
        <!-- Card Agente KAIROS -->
        <div class="module-card" id="kairos-card">
            <div class="card-header">
                <div class="status-indicator" id="kairos-status"></div>
                <h3>Agente KAIROS</h3>
            </div>
            <div class="card-body">
                <p>Agente di esecuzione comandi via MCP. Avvialo per abilitare le capacità avanzate del D-ND Engine.</p>
            </div>
            <div class="card-footer">
                <button id="kairos-action-btn" onclick="handleKairosAction()">Avvia Server MCP</button>
            </div>
        </div>
        <!-- Qui verranno aggiunte altre card in futuro -->
    </div>
    <div id="command-output">Il tuo prompt o comando per Gemini apparirà qui...</div>
</div>
```

### 2.3. Aggiornamento della Logica (`script.js`)

**Scopo:** Implementare la logica interattiva per il pulsante della card di KAIROS, gestendo l'avvio del server MCP e la comunicazione con l'ambiente Gemini-CLI.

```javascript
// --- KAIROS Agent Card Logic ---
let isKairosActive = false;

function handleKairosAction() {
    const statusIndicator = document.getElementById('kairos-status');
    const actionButton = document.getElementById('kairos-action-btn');
    const commandOutput = document.getElementById('command-output');

    if (!isKairosActive) {
        // Genera il comando per AVVIARE il server MCP
        const command = `gemini --mcp-tools dnd_tools.json --mcp-port 8080`;
        commandOutput.textContent = `AZIONE RICHIESTA IN UN NUOVO TERMINALE:\n\n1. Naviga alla directory D-ND_Engine.\n2. Esegui il seguente comando per avviare il server KAIROS:\n\n${command}\n\n3. Lascia quel terminale aperto.\n4. Dopo averlo avviato, clicca di nuovo questo pulsante per confermare.`;
        
        // Cambia lo stato del pulsante per la conferma
        actionButton.textContent = 'Conferma Avvio KAIROS';
        actionButton.style.backgroundColor = '#e67e22'; // Arancione per "in attesa"
        isKairosActive = 'pending'; // Stato intermedio

    } else if (isKairosActive === 'pending') {
        // L'utente ha confermato l'avvio
        statusIndicator.className = 'status-indicator active';
        actionButton.textContent = 'KAIROS Attivo';
        actionButton.disabled = true;
        actionButton.style.backgroundColor = '#2ecc71'; // Verde
        isKairosActive = true;
        
        // Prepara l'ambiente di aistudio.google.com
        commandOutput.textContent = `KAIROS è attivo. Per collegare questa sessione al tuo agente, esegui il seguente comando qui:\n\n/set mcp_server http://localhost:8080`;
    }
}
```