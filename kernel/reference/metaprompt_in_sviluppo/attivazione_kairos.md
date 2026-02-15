# Attivazione dell'Agente KAIROS

Questo documento descrive il processo di attivazione dell'Agente KAIROS, inclusi i comandi per avviare il server MCP e configurare l'interazione con l'ambiente di sviluppo.

## Il Comando di Avvio del Server MCP

Il comando seguente avvia `gemini-cli` in una modalità speciale, trasformandolo in un server MCP (Multi-Component Protocol). Questo server rimarrà in esecuzione, esponendo i tool definiti in `dnd_tools.json` sulla porta `8080` (la porta di default).

```powershell
gemini --mcp-tools dnd_tools.json --mcp-port 8080
```

**Output Atteso:**
Dopo l'esecuzione del comando, il terminale dovrebbe mostrare un messaggio simile al seguente, indicando che il server è attivo e in attesa di richieste:

```
Serving MCP tools on port 8080
Tool Definition Source: dnd_tools.json
Registered 1 tool(s):
- get_system_hardware_info
```

È fondamentale mantenere questa finestra del terminale aperta per tutta la durata della sessione, poiché rappresenta il "cervello" di KAIROS in esecuzione.

## Come Interagire con KAIROS

Una volta che il server MCP è in ascolto, è necessario collegare l'istanza `aistudio.google.com` (l'ambiente di sviluppo) a KAIROS.

Per fare ciò, eseguire il seguente comando nella conversazione attuale:

```
/set mcp_server http://localhost:8080
```

Dopo l'esecuzione di questo comando, l'ambiente di sviluppo sarà consapevole della disponibilità dello strumento `get_system_hardware_info`, oltre ai suoi strumenti di base.

## Il Primo Test: Dialogo con KAIROS

Con il server MCP attivo e l'endpoint configurato, è possibile procedere con il primo test di interazione con KAIROS.

**Prompt di Test:**
Invia il seguente prompt nella chat:

> "KAIROS, per favore, forniscimi un riepilogo dello stato del sistema su cui sei in esecuzione."

**Ciclo Operativo di KAIROS (Previsione):**
1.  **COAC (l'agente di sviluppo)** riceve il prompt e lo inoltra all'architettura KAIROS.
2.  **KAIROS (Fase 1 - Analisi):** Analizza l'intento dell'utente: "L'utente desidera conoscere le specifiche della macchina."
3.  **KAIROS (Fase 2 - Mappatura Tool):** Scansiona i tool disponibili (che ora includono `get_system_hardware_info`) e identifica questo come lo strumento appropriato per il task.
4.  **KAIROS (Fase 3 - Esecuzione):** Esegue una chiamata al server MCP su `localhost:8080`, invocando lo script `get_system_info.py`.
5.  **KAIROS (Fase 4 - Sintesi):** Riceve l'output JSON dallo script, lo analizza e lo formatta in una risposta chiara e leggibile per l'utente.
