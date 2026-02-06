import json
import re

def extract_json(text: str) -> dict:
    """
    尝试从模型输出中提取第一个 JSON 对象 { ... }。
    """
    t = text.strip()

    # 情况1：本身就是 JSON
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    # 情况2：夹带其它文字，抓最外层 {...}
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def require_keys(obj: dict, keys: list[str]):
    for k in keys:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

def repair_to_json(llm, bad_text: str, schema_hint: str, timeout=60) -> dict:
    """
    如果模型输出不是 JSON，用二次提示修复成合法 JSON。
    """
    messages = [
        {"role": "system", "content": "You are a JSON repair bot. Return ONLY valid JSON. No extra text."},
        {"role": "user", "content": f"Fix the following into valid JSON matching this schema:\n{schema_hint}\n\nTEXT:\n{bad_text}"}
    ]
    fixed = llm.chat(messages, temperature=0.0, timeout=timeout)
    return extract_json(fixed)

def safe_llm_json(llm, messages, schema_hint: str, required_keys: list[str], max_attempts=2) -> dict:
    """
    统一封装：调用 LLM -> 解析 JSON -> 失败则修复重试
    """
    last = None
    for _ in range(max_attempts):
        last = llm.chat(messages, temperature=0.0)
        try:
            obj = extract_json(last)
            require_keys(obj, required_keys)
            return obj
        except Exception:
            try:
                obj = repair_to_json(llm, last, schema_hint=schema_hint)
                require_keys(obj, required_keys)
                return obj
            except Exception:
                continue
    raise ValueError(f"Failed to get valid JSON after retries. Last output:\n{last}")
