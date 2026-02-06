from backend.llm.client import LLMClient
from backend.agents.utils import safe_llm_json

CRITIC_SCHEMA_HINT = """
{
  "checklist": [{"step": 1, "text": "...", "citations": ["ref"]}],
  "templates": [{"type": "email", "subject": "...", "body": "..."}],
  "risks": [{"risk": "...", "mitigation": "...", "citations": ["ref"]}],
  "citations": [{"ref": "...", "excerpt": "..."}]
}
"""

CRITIC_SYSTEM = """You are CriticAgent.
You will receive a draft JSON and a list of allowed EVIDENCE refs.

Rules:
1) Any factual claim must be supported by at least one citation ref.
2) Checklist: every step must have citations. If missing, rewrite the step into a "next action" that avoids factual claims, OR mark "Not covered in provided docs".
3) If draft includes unsupported content, remove or mark as "Not covered".

Return ONLY valid JSON matching the schema. No extra text.
"""

class CriticAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, draft: dict, allowed_refs: list[str]) -> dict:
        user_prompt = f"""ALLOWED_EVIDENCE_REFS:
{allowed_refs}

DRAFT_JSON:
{draft}
"""
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        return safe_llm_json(
            self.llm,
            messages,
            schema_hint=CRITIC_SCHEMA_HINT,
            required_keys=["checklist","templates","risks","citations"]
        )
