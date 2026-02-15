# Configurazioni Iniziali e Implementazione della Card Agente KAIROS

Questo documento descrive il processo di configurazione iniziale del D-ND Engine, focalizzandosi sull'integrazione dell'Agente KAIROS tramite l'implementazione di una "Card Modulo" nell'interfaccia utente.

## Ruolo di COAC v5.0

COAC v5.0, operando su `aistudio.google.com`, funge da Architetto di Sistema e Generatore di Istruzioni. Il suo ruolo è definire istruzioni chiare e complete per l'implementazione da parte dell'Utente-Amministratore sull'agente `gemini-cli`. L'output di COAC v5.0 è un "pacchetto di istruzioni" dettagliato per l'agente KAIROS.

## Pipeline Cognitiva (Ciclo di Generazione Istruzioni)

### Fase 0+ – Innesco del Campo Dinamico

*   **Punto di Equilibrio:** La dualità tra l'architetto e l'esecutore.
*   **Punti Osservati:** L'architettura UI (il Cockpit a schede) e il potenziale backend (KAIROS con i suoi tool MCP) necessitano di integrazione.
*   **Impressione Emergente:** La priorità è popolare la Dashboard del Cockpit con la sua prima "Card Modulo" interattiva per rendere tangibile l'interazione con l'agente KAIROS.

### Fase 1 – Diagnosi e Piano di Esecuzione

*   **Input:** Lo stato attuale del progetto (Cockpit a schede).
*   **Intento Latente:** Tradurre il concetto astratto di "Agente KAIROS" in un componente UI interattivo all'interno del nuovo Cockpit.
*   **Framework Selezionato:** `Orchestratore-Costruttore (OCC) v1.0` per la generazione di codice HTML/CSS/JS e istruzioni per `gemini-cli`.
*   **Piano Dichiarato:** Generare istruzioni complete e frammenti di codice per implementare la "Card Modulo" per KAIROS nella Dashboard del Cockpit.

## Risultante (R): Pacchetto di Istruzioni - "Implementazione Card Agente KAIROS"

**Obiettivo:** Aggiungere una card interattiva nella scheda "Dashboard" del Cockpit per avviare e monitorare l'agente KAIROS.

### Istruzione 1: Aggiornamento dello Stile (`style.css`)

**Scopo:** Aggiungere le classi CSS per definire l'aspetto delle card e i loro indicatori di stato.

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

### Istruzione 2: Aggiornamento della Struttura (`Control-Panel.html`)

**Scopo:** Aggiungere il placeholder per la card di KAIROS nella scheda "Dashboard".

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

### Istruzione 3: Aggiornamento della Logica (`script.js`)

**Scopo:** Implementare la logica interattiva per il pulsante della card di KAIROS.

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