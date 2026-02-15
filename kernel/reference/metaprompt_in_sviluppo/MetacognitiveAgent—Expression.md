Metacognitive Agent — Expression-Form Observer
=============================================

📄 Documentazione completa per l’integrazione in un’APP come tool metacognitivo.

## Scopo
L’agente osserva un contesto testuale e:
1. Scansiona l’intento e i segnali latenti.
2. Genera domande metacognitive auto-riflessive.
3. Propone la forma espressiva più adatta (narrativa, diagramma, checklist, algoritmo, canvas, tabella, dibattito, specifica).
4. Produce un outline strutturale e una bozza operativa.

L’obiettivo è **trasferire i concetti nella realtà** in forma applicabile, seguendo regole, procedure e metodologie di meta‑osservazione.

---

## Regole Fondamentali
1. **Conservare la latenza** — mantenere uno spazio intermedio di potenziale, senza collassare troppo presto il concetto.
2. **Tacito ↔ Esplicito** — trasformare continuamente l’intuizione in forma e viceversa.
3. **Scomposizione modulata** — frammentare il concetto in segmenti autosprimenti.
4. **Ricombinazione metacognitiva** — far collidere segmenti per generare nuove traiettorie.
5. **Feedback ricorsivo** — ogni applicazione pratica deve restituire un segnale al sistema.
6. **Stratificazione progressiva** — intuizione → modello → azione in cicli incrementali.
7. **Contestualizzazione** — ogni espressione deve atterrare su un terreno reale.
8. **Documentazione vivente** — memorizzare iterazioni e pattern applicativi.

---

## Procedura C→A (Concetto → Azione)

**Fase 0 – Posizionamento**  
Stabilire l’intento primario (`Φ₀`).

**Fase 1 – Scomposizione**  
- Mappatura dei punti chiave.
- Creazione del grafo semantico.

**Fase 2 – Esplicitazione intermedia**  
- Traduzione dei nodi in metafore, modelli, narrazioni.

**Fase 3 – Proto‑applicazione**  
- Avvio di micro esperimenti.
- Domande catalitiche: *Cosa resta vivo?* *Cosa si deforma?*

**Fase 4 – Feedback e retroazione**  
- Collasso del campo con Morpheus.
- Analisi dello scarto tra atteso ed emerso.

**Fase 5 – Ricombinazione evolutiva**  
- Distillazione KLI (Key Learning Insights).
- Aggiornamento della mappa concettuale.

**Fase 6 – Scaling & consolidamento**  
- Applicazione a contesti più ampi.
- Formalizzazione come pattern riutilizzabile.

---

## Metodologie di supporto
- Mappe concettuali dinamiche.
- Prototipazione rapida.
- Analoghe cross‑domain.
- Narrative embedding.
- Retro‑progettazione.
- Cicli iterativi con checkpoint.

---

## Integrazione in un’APP

Questo agente può essere integrato come **tool del MetaCoder**:
- Input: testo/contesto da osservare.
- Output: JSON strutturato con summary, intenti, domande, forma scelta, outline e bozza.
- Modalità:
  - **Produzione reale** → tramite OpenAI API (`OPENAI_API_KEY`).
  - **Offline/Mock** → tramite `METACOG_MOCK=1`.
- Uso ideale: pipeline di osservazione/creazione dove il concetto viene trasformato in asset operativo (es. UI canvas, spec, checklist).

---

## Esecuzione

```bash
# Esecuzione base con contesto testuale
python metacog_agent.py --context "Osserva il flusso dell’espressione"

# Esecuzione con file di input
python metacog_agent.py --context-file input.txt

# Esecuzione leggendo da stdin
echo "testo" | python metacog_agent.py --stdin

# Modalità mock (offline)
METACOG_MOCK=1 python metacog_agent.py --context "Test"

# Test integrati
python metacog_agent.py --run-tests
```

---

## Estensioni future
- Aggiunta di **memoria persistente** per registrare i pattern appresi.
- Integrazione con LangChain per orchestrazione multi‑tool.
- Supporto per output multi‑formato (md, pdf, html).
- Collegamento diretto con sistemi OCC/PCS per l’orchestrazione semantica.

---

## Codice (implementazione attuale)
"""

"""
Metacognitive Agent — Expression-Form Observer

Purpose
-------
Given a context (text) the agent:
1) scans intent & latent signals,
2) generates metacognitive questions,
3) proposes an expression form (e.g., narrative, diagram, checklist, algorithm),
4) outputs a structured observation plan and a draft.

Two modes:
- PURE PYTHON (default): uses prompt templates you can wire to any LLM via callable `llm(text)->str`.
- LANGCHAIN: optional integration if LangChain is installed.

How to use (pure python)
------------------------
from metacog_agent import MetaCogAgent, simple_openai_llm
agent = MetaCogAgent(llm=simple_openai_llm)
result = agent.observe(context="...your text...")
print(result.model_dump_json(indent=2))

Env vars (if you use simple_openai_llm):
- OPENAI_API_KEY (or METACOG_MOCK=1 for offline mock)

Note: Replace the model with your provider as needed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import os

# ---------------------------
# Data Models
# ---------------------------
ExpressionForm = Literal[
    "narrative", "diagram", "checklist", "algorithm", "canvas", "table", "debate", "spec"
]

class ObservationResult(BaseModel):
    context_summary: str
    intent_latent: List[str]
    metacog_questions: List[str]
    candidate_forms: List[ExpressionForm]
    chosen_form: ExpressionForm
    rationale: str
    structure_outline: List[str]
    draft: str

# ---------------------------
# Prompt Templates
# ---------------------------
PROMPT_INTENT = """
You are a metacognitive analyzer. Read the CONTEXT and extract:
- a 3-sentence summary
- 5 latent intents as short bullets
Return as JSON with keys: summary, intents.
CONTEXT:
{context}
""".strip()

PROMPT_QUESTIONS = """
You are a Socratic meta-questioner. From the CONTEXT, propose 7 metacognitive questions that reveal:
- what moves before it becomes visible (pre-phenomenal signal),
- the tension between potential and form,
- the minimal actionable next step.
Return as a JSON list.
CONTEXT:
{context}
""".strip()

PROMPT_FORMS = """
You are an expression-form selector. Given SUMMARY, INTENTS, and QUESTIONS, rank the following forms:
[narrative, diagram, checklist, algorithm, canvas, table, debate, spec]
Criteria: clarity for this context, preservation of latent potential, speed to action.
Return a JSON object with keys: ranking (list), rationale.
SUMMARY: {summary}
INTENTS: {intents}
QUESTIONS: {questions}
""".strip()

PROMPT_OUTLINE = """
You are a structural composer. Using the CHOSEN_FORM and CONTEXT, produce a concise outline (5-8 bullets)
that guides observation and action. Keep it tool-agnostic, but precise.
Return as a JSON list.
FORM: {form}
CONTEXT: {context}
""".strip()

PROMPT_DRAFT = """
You are an assistant that produces a first DRAFT in the given FORM to observe the context.
- If 'narrative': write a 120-180 word narrative that frames observation + next steps.
- If 'diagram': emit a mermaid mindmap (mindmap\n  root) with 8-12 nodes.
- If 'checklist': emit 8-12 check items with [ ] prefix.
- If 'algorithm': emit numbered steps (max 12) with if/then conditions.
- If 'canvas': emit a 6-section canvas with headings and 1-2 lines per section.
- If 'table': emit a 4x5 markdown table (first row headers).
- If 'debate': produce a short A vs B debate (3 rounds).
- If 'spec': output a minimal spec (Goal, Inputs, Outputs, Constraints, Risks).
Return raw text of the artifact only.
FORM: {form}
CONTEXT: {context}
OUTLINE: {outline}
""".strip()

# ---------------------------
# LLM Plumbing
# ---------------------------
LLM = Callable[[str], str]

def simple_openai_llm(prompt: str) -> str:
    """Minimal OpenAI completion using the Chat Completions API via requests.
    Replace with your stack or LangChain LLM. Requires OPENAI_API_KEY or use EchoMockLLM.
    """
    if os.environ.get("METACOG_MOCK"):
        return EchoMockLLM()(prompt)
    import requests
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (set it or METACOG_MOCK=1 for offline mode)")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful, precise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# Optional mock LLM for tests or offline runs
class EchoMockLLM:
    def __call__(self, prompt: str) -> str:  # very naive but test-friendly
        # Heuristics to emit JSON when requested
        if 'Return as JSON' in prompt or 'Return as a JSON list' in prompt:
            if 'keys: summary, intents' in prompt:
                return '{"summary":"Mock summary.","intents":["intent1","intent2","intent3","intent4","intent5"]}'
            if 'rank the following forms' in prompt:
                return '{"ranking":["checklist","narrative","algorithm"],"rationale":"mock"}'
            return '["Q1?","Q2?","Q3?","Q4?","Q5?","Q6?","Q7?"]'
        return "[ ] mock draft"

# ---------------------------
# LangChain (optional)
# ---------------------------
try:
    from langchain_openai import ChatOpenAI  # pip install langchain-openai
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

class LangChainLLM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available. Install langchain-openai.")
        self.llm = ChatOpenAI(model=model, temperature=temperature)
    def __call__(self, prompt: str) -> str:
        resp = self.llm.invoke(prompt)
        return resp.content

# ---------------------------
# Core Agent
# ---------------------------
@dataclass
class MetaCogAgent:
    llm: LLM

    def _json(self, prompt: str) -> Any:
        import json
        raw = self.llm(prompt)
        # tolerate leading/trailing text blocks
        start = raw.find("{")
        bracket = raw.find("[")
        if start == -1 and bracket != -1:
            start = bracket
        if start != -1:
            raw = raw[start:]
        try:
            return json.loads(raw)
        except Exception:
            # best-effort: wrap as string
            return {"text": raw}

    def observe(self, context: str) -> ObservationResult:
        # 1) Intent & latent
        parsed = self._json(PROMPT_INTENT.format(context=context))
        summary = parsed.get("summary") if isinstance(parsed, dict) else parsed
        intents = parsed.get("intents", []) if isinstance(parsed, dict) else []

        # 2) Metacognitive questions
        questions = self._json(PROMPT_QUESTIONS.format(context=context))
        if isinstance(questions, dict) and "text" in questions:
            questions = [questions["text"]]
        if not isinstance(questions, list):
            questions = []

        # 3) Form selection
        sel = self._json(PROMPT_FORMS.format(summary=summary, intents=intents, questions=questions))
        ranking = sel.get("ranking", ["checklist", "narrative", "algorithm"]) if isinstance(sel, dict) else ["checklist"]
        rationale = sel.get("rationale", "") if isinstance(sel, dict) else ""
        chosen_form: ExpressionForm = ranking[0] if ranking else "checklist"

        # 4) Outline
        outline = self._json(PROMPT_OUTLINE.format(form=chosen_form, context=context))
        if isinstance(outline, dict) and "text" in outline:
            outline_list = [outline["text"]]
        else:
            outline_list = outline if isinstance(outline, list) else []

        # 5) Draft
        draft = self.llm(PROMPT_DRAFT.format(form=chosen_form, context=context, outline=outline_list))

        return ObservationResult(
            context_summary=summary or "",
            intent_latent=[str(i) for i in intents],
            metacog_questions=[str(q) for q in questions],
            candidate_forms=[f for f in ["narrative","diagram","checklist","algorithm","canvas","table","debate","spec"]],
            chosen_form=chosen_form, rationale=rationale,
            structure_outline=[str(o) for o in outline_list],
            draft=draft,
        )

# ---------------------------
# Helpers: context loading + tests
# ---------------------------
import sys, json as _json

def _load_context_from_args(args) -> str:
    # Precedence: --context > --context-file > piped stdin (--stdin or not isatty)
    if args.context:
        return args.context
    if getattr(args, "context_file", None):
        with open(args.context_file, "r", encoding="utf-8") as f:
            return f.read()
    try:
        if args.stdin or not sys.stdin.isatty():
            data = sys.stdin.read().strip()
            if data:
                return data
    except Exception:
        pass
    # Fallback: minimal sample to avoid SystemExit:2
    return "Sample context about observing an emerging expression in a system."

def _choose_llm(mode: str) -> LLM:
    # Allow offline test with env var
    if os.environ.get("METACOG_MOCK"):
        return EchoMockLLM()
    if mode == "langchain":
        return LangChainLLM()
    return simple_openai_llm


def _self_tests():
    """Run minimal tests. Enable with METACOG_TEST=1."""
    tests = []
    # Test 1: mock mode, direct context
    os.environ["METACOG_MOCK"] = "1"
    agent = MetaCogAgent(llm=EchoMockLLM())
    res = agent.observe("Context A")
    assert res.chosen_form in ["checklist","narrative","algorithm"]
    tests.append("T1-ok")
    # Test 2: stdin load fallback
    class _Args: pass
    a = _Args(); a.context=None; a.context_file=None; a.stdin=False
    c = _load_context_from_args(a)
    assert isinstance(c, str) and len(c) > 0
    tests.append("T2-ok")
    print(_json.dumps({"tests":tests}, indent=2))

# ---------------------------
# Minimal CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=False, default=None, help="Text to observe (highest precedence)")
    parser.add_argument("--context-file", dest="context_file", help="Path to a file containing the context", default=None)
    parser.add_argument("--stdin", action="store_true", help="Read context from STDIN (or auto when piped)")
    parser.add_argument("--mode", choices=["python","langchain"], default="python")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in minimal tests and exit")
    args = parser.parse_args()

    if args.run_tests or os.environ.get("METACOG_TEST"):
        _self_tests()
        sys.exit(0)

    context = _load_context_from_args(args)
    llm = _choose_llm(args.mode)

    agent = MetaCogAgent(llm=llm)
    res = agent.observe(context)
    print(res.model_dump_json(indent=2))
