from backend.llm.client import LLMClient
from backend.agents.utils import safe_llm_json

ROUTER_SCHEMA_HINT = """
{
  "intent": "billing|course|housing|unknown",
  "retrieval_queries": ["2-5 queries"],
  "required_outputs": ["checklist","templates","risks"]
}
"""

ROUTER_SYSTEM = """You are RouterAgent.
Classify the user request into an intent: billing | course | housing | unknown.
Generate 2-5 retrieval_queries that would find relevant policy text in offline docs.
Return ONLY valid JSON matching the schema.
No extra text.
"""

class RouterAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
        return safe_llm_json(
            self.llm,
            messages,
            schema_hint=ROUTER_SCHEMA_HINT,
            required_keys=["intent", "retrieval_queries", "required_outputs"]
        )

if __name__ == "__main__":
    llm = LLMClient()
    agent = RouterAgent(llm)
    print(agent.run("I was charged a fee I don’t understand. What should I do?"))
