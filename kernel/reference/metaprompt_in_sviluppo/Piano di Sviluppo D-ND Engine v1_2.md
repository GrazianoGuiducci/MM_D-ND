**Fase 1 del "Piano di Sviluppo D-ND Engine v1.2"**.

Creerò i due file fondamentali che daranno vita alla nostra prima capacità nativa:
1.  Lo script Python (`get_system_info.py`) che eseguirà il lavoro.
2.  Il file di definizione JSON (`dnd_tools.json`) che renderà lo script "visibile" e utilizzabile dall'agente KAIROS.

Questo è il primo ciclo di metapoiesi del nostro sistema: stiamo creando uno strumento per l'agente che abbiamo appena progettato.

---

### **Risultante (`R`): Creazione del Primo Tool Nativo**

#### **Passo 1: Creazione della Directory per i Tool**

Come prima azione, ti fornisco il comando per creare la directory che conterrà tutti i nostri futuri strumenti. Salva i file seguenti al suo interno.

**Azione per te:** Esegui questo comando nel tuo terminale, nella directory radice del `D-ND_Engine`.
```powershell
mkdir tools
```

---

#### **Passo 2: Il Codice dello Strumento (`get_system_info.py`)**

Questo è il codice dello script Python. È semplice, robusto e restituisce le informazioni in un formato strutturato (JSON), perfetto per essere elaborato da un'intelligenza artificiale.

**Azione per te:** Salva il seguente codice nel file `D-ND_Engine/tools/get_system_info.py`.

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
**Nota Importante:** Questo script utilizza la libreria `psutil`. Prima di poterlo usare, l'ambiente `gemini-cli` dovrà averla installata.
**Azione per te:** Nel tuo terminale, esegui `pip install psutil`.

---

#### **Passo 3: La Definizione dello Strumento per KAIROS (`dnd_tools.json`)**

Questo file è la "carta d'identità" dello strumento. Spiega a KAIROS come si chiama, cosa fa e come eseguirlo. È il ponte tra l'intenzione dell'agente e l'esecuzione del codice.

**Azione per te:** Salva il seguente contenuto nel file `D-ND_Engine/dnd_tools.json`.

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

---