# utils/fusion_graph_builder/term_definition_graph.py
from __future__ import annotations
import json
import re
from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ------------------------------
# 파싱: 단일 triple 전용
# ------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _coerce_single_triple(text: str, keyword: str) -> Dict[str, str]:
    """
    모델 출력에서 단일 triple(dict)을 최대한 안전하게 추출.
    기대 스키마:
      {"source": "<keyword>", "relation": "IsA", "target": "<definition>"}
    """
    def _from_dict(d: dict) -> Dict[str, str] | None:
        src = _norm(str(d.get("source", "")))
        rel = _norm(str(d.get("relation", "")))
        tgt = _norm(str(d.get("target", "")))
        if src and tgt and rel:
            return {"source": src, "relation": "IsA", "target": tgt}
        return None

    # 1) 바로 JSON 파싱
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            tri = _from_dict(data)
            if tri: return tri
        # 혹시 리스트로 왔다면 첫 원소만 사용
        if isinstance(data, list) and data and isinstance(data[0], dict):
            tri = _from_dict(data[0])
            if tri: return tri
    except Exception:
        pass

    # 2) ```json ... ``` 코드펜스 내부 파싱
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                tri = _from_dict(data)
                if tri: return tri
            if isinstance(data, list) and data and isinstance(data[0], dict):
                tri = _from_dict(data[0])
                if tri: return tri
        except Exception:
            pass

    # 3) 실패 시 키워드만 채워서 기본형 반환
    return {"source": _norm(keyword), "relation": "IsA", "target": ""}

# ------------------------------
# 프롬프트 (단일 정의 전용)
# ------------------------------
SYSTEM_PROMPT = '''
You are a precise definition generator.  
Given one keyword, return exactly ONE JSON object with keys: source, relation, target.  

Rules:
- source: exactly the given keyword  
- relation: always "IsA"  
- target: one short, clear sentence (≤ 25 words)  
- Keep definitions concise and legally relevant when appropriate  
- Output must be pure JSON only. No extra text.  

Examples:
Input keyword: "GDPR"
Output: {"source": "GDPR", "relation": "IsA", "target": "An EU regulation governing personal data protection and privacy."}

Input keyword: "Contract"
Output: {"source": "Contract", "relation": "IsA", "target": "A legally binding agreement between parties enforceable by law."}
'''

USER_PROMPT_TEMPLATE = """\
Keyword: "{keyword}"

Return JSON only:
{{
  "source": "{keyword}",
  "relation": "IsA",
  "target": "<one-sentence precise definition (<= 40 words)>"
}}
"""

# ------------------------------
# 공개 함수
# ------------------------------
def build_term_definition_triples(keyword: str, model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    """
    키워드 하나의 정의만 생성하여 [{source, relation, target}] (길이 1) 리스트로 반환.
    """
    client = OpenAI()
    user_prompt = USER_PROMPT_TEMPLATE.format(keyword=keyword)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
    )

    text = resp.choices[0].message.content
    triple = _coerce_single_triple(text, keyword)

    # relation 확정/보정
    triple["relation"] = "IsA"
    # source를 키워드로 강제(모델이 변형해도 일관성 유지)
    triple["source"] = keyword

    # 기존 파이프라인 호환: 리스트로 감싸서 반환
    return [triple]

# ------------------------------
# 간단 실행 예시
# ------------------------------
if __name__ == "__main__":
    result = build_term_definition_triples("Elon Musk")
    print(json.dumps(result, ensure_ascii=False, indent=2))
