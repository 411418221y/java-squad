from backend.llm.client import LLMClient
from backend.agents.utils import safe_llm_json

COMPOSER_SCHEMA_HINT = """
{
  "checklist": [{"step": 1, "text": "...", "citations": ["ref"]}],
  "templates": [{"type": "email", "subject": "...", "body": "..."}],
  "risks": [{"risk": "...", "mitigation": "...", "citations": ["ref"]}],
  "citations": [{"ref": "...", "excerpt": "..."}]
}
"""

COMPOSER_SYSTEM = """You are ComposerAgent.
Write actionable outputs using ONLY the provided EVIDENCE snippets.
If evidence is missing, say "Not covered in provided docs" and propose what document would be needed.
Return ONLY valid JSON matching the schema. No markdown. No extra text.

Rules:
- Each checklist step MUST include citations (refs).
- Any factual claim MUST have citations.
"""

def format_evidence(evidence: list[dict], max_chars=1200) -> str:
    blocks = []
    for e in evidence:
        text = e["text"][:max_chars]
        blocks.append(f"REF: {e['ref']}\nTEXT:\n{text}\n")
    return "\n---\n".join(blocks)

class ComposerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, evidence: list[dict]) -> dict:
        user_prompt = f"""USER_QUESTION:
{question}

EVIDENCE:
{format_evidence(evidence)}
"""
        messages = [
            {"role": "system", "content": COMPOSER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        return safe_llm_json(
            self.llm,
            messages,
            schema_hint=COMPOSER_SCHEMA_HINT,
            required_keys=["checklist", "templates", "risks", "citations"]
        )
