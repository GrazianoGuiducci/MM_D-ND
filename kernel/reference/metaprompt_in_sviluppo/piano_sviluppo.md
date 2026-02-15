# Piano di Sviluppo D-ND Engine v1.2

Questo documento delinea la Fase 1 del piano di sviluppo per il D-ND Engine, concentrandosi sulla creazione del primo tool nativo. L'obiettivo è stabilire le fondamenta per le capacità operative del sistema.

## Creazione del Primo Tool Nativo

Questa fase prevede la creazione di due componenti fondamentali che abiliteranno la prima capacità nativa del D-ND Engine:

1.  **Script Python (`get_system_info.py`):** Il codice eseguibile che raccoglierà le informazioni di sistema.
2.  **File di Definizione JSON (`dnd_tools.json`):** Questo file renderà lo script "visibile" e utilizzabile dall'agente KAIROS, fungendo da ponte tra l'intenzione dell'agente e l'esecuzione del codice.

Questo processo rappresenta il primo ciclo di "metapoiesi" del sistema, ovvero la creazione di nuovi strumenti per l'agente KAIROS.

### Passo 1: Creazione della Directory per i Tool

È necessario creare una directory dedicata per ospitare tutti i futuri strumenti del D-ND Engine.

```powershell
mkdir tools
```

### Passo 2: Il Codice dello Strumento (`get_system_info.py`)

Questo script Python è progettato per raccogliere informazioni chiave sul sistema operativo, hardware e rete, restituendole in un formato JSON strutturato, ideale per l'elaborazione da parte di un'intelligenza artificiale.

```python
import json
import platform
import psutil
import socket

def get_system_info():
    """
    Raccoglie informazioni chiave sul sistema operativo, hardware e rete
    e le restituisce come un dizionario Python.
    """
    try:
        # Informazioni sulla memoria virtuale
        mem = psutil.virtual_memory()
        
        # Costruisce il dizionario con le informazioni
        info = {
            "platform": {
                "os": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine()
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_frequency_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
                "current_usage_percent": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_percent": mem.percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "used_percent": psutil.disk_usage('/').percent
            },
            "network": {
                "hostname": socket.gethostname(),
                "ip_address": socket.gethostbyname(socket.gethostname())
            }
        }
        return info
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    system_data = get_system_info()
    # Stampa l'output in formato JSON compatto
    print(json.dumps(system_data, indent=None, separators=(',', ':')))
```
**Nota Importante:** Questo script utilizza la libreria `psutil`. È necessario installarla nell'ambiente `gemini-cli` prima dell'utilizzo.

### Passo 3: La Definizione dello Strumento per KAIROS (`dnd_tools.json`)

Questo file funge da "carta d'identità" dello strumento, fornendo a KAIROS le informazioni necessarie per identificarlo, comprenderne la funzione e invocarlo correttamente.

```json
[
  {
    "function_declarations": [
      {
        "name": "get_system_hardware_info",
        "description": "Recupera informazioni dettagliate sull'hardware e sul sistema operativo della macchina corrente, inclusi dettagli su OS, CPU, memoria (RAM), utilizzo del disco e configurazione di rete di base. Utile per diagnosi, controlli di compatibilità o per ottenere un sommario dello stato del sistema.",
        "parameters": {
          "type": "OBJECT",
          "properties": {}
        }
      }
    ],
    "command": [
      "python",
      "tools/get_system_info.py"
    ]
  }
]
```