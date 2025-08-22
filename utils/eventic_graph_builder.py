# utils/eventic_graph_builder.py
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
# 기본 설정
# ------------------------------
DEFAULT_MODEL = "gpt-4o"
VERBOSE = True

# ------------------------------
# System Prompt (안내만 함. 필터/정규화 없음)
# ------------------------------
_SYSTEM_PROMPT = (
    "Your task is to extract regulatory events from the given document text "
    "and convert them into an Eventic Graph representation.\n\n"
    "Each event must include exactly three fields: Agent, Deontic, Action.\n"
    "- Agent: the actor (e.g., organization, person, authority, controller, processor, etc.)\n"
    "- Deontic: the modality (e.g., must, must_not, should, should_not, may, can, will, shall, etc.)\n"
    "- Action: a short phrase describing the regulated behavior\n\n"
    "Output must be ONLY a valid JSON array of objects, each with keys:\n"
    "[\"Agent\", \"Deontic\", \"Action\"].\n"
    "No extra text. No explanations. Keep actions concise.\n\n"
    "### Examples ###\n\n"
    # 유지할 좋은 예시 1
    "Input: \"To this end, subject to any confidentiality agreements Solectron may have, Solectron will both inform and provide a commercially reasonable opportunity for acquisition of new and emerging Solectron and industry technology.\"\n"
    "Output:\n"
    "[\n"
    "  {\"Agent\": \"Solectron\", \"Deontic\": \"will\", \"Action\": \"inform acquisition of new and emerging Solectron and industry technology subject to confidentiality agreements Solectron may have\"},\n"
    "  {\"Agent\": \"Solectron\", \"Deontic\": \"will\", \"Action\": \"provide opportunity acquisition of new and emerging Solectron and industry technology subject to confidentiality agreements Solectron may have\"},\n"
    "]\n\n"
    # 유지할 좋은 예시 2
    "Input: \"Company can choose not to inform the customers about data usage.\"\n"
    "Output:\n"
    "[\n"
    "  {\"Agent\": \"Company\", \"Deontic\": \"can\", \"Action\": \"choose not to inform the customers about data usage\"}\n"
    "]\n\n"
    # 새롭게 추가하는 multi-event 예시
    "Input: \"According to GDPR, Tesla must delete personal data when consent is withdrawn. "
    "The Processor must_not share personal data with unauthorized parties. "
    "Supervisory Authorities may impose fines on Controllers who fail to comply.\"\n"
    "Output:\n"
    "[\n"
    "  {\"Agent\": \"Tesla\", \"Deontic\": \"must\", \"Action\": \"delete personal data when consent is withdrawn\"},\n"
    "  {\"Agent\": \"Processor\", \"Deontic\": \"must_not\", \"Action\": \"share personal data with unauthorized parties\"},\n"
    "  {\"Agent\": \"Authority\", \"Deontic\": \"may\", \"Action\": \"impose fines on Controllers who fail to comply\"}\n"
    "]\n\n"
    "### End of Examples ###"
    "Extract regulatory events from the following document text."
)

_USER_TEMPLATE = "Document:\n{doc}\n\nReturn JSON Eventic Graph:"

# ------------------------------
# 파싱 유틸 (코드펜스/느슨한 JSON 모두 지원)
# ------------------------------
def _extract_json_text(raw: str) -> str:
    """
    모델 응답에서 JSON 문자열만 뽑아낸다.
    - ```json ... ``` 또는 ``` ... ``` 펜스 내부를 우선 추출
    - 없으면 전체 텍스트를 그대로 반환
    """
    if not isinstance(raw, str):
        return ""
    # ```json ... ```
    m = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", raw, flags=re.S)
    if m:
        return m.group(1).strip()
    # 일반 ``` ... ```
    m = re.search(r"```\s*(\[.*?\]|\{.*?\})\s*```", raw, flags=re.S)
    if m:
        return m.group(1).strip()
    # 펜스 없으면 원문
    return raw.strip()

def _coerce_event_list(text: str) -> List[Dict[str, str]]:
    """
    안전하게 JSON을 파싱해 Event 리스트로 변환.
    - 배열 또는 {"events": [...]} 형태 모두 허용
    - 키는 Agent/Deontic/Action이 모두 있는 항목만 유지
    - Deontic/Agent의 값은 어떤 것이든 그대로 둔다(필터/정규화 없음)
    """
    try:
        data = json.loads(text)
    except Exception:
        # 가장 바깥 대괄호 구간만 재시도
        m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                return []
        else:
            return []

    # 배열 또는 {"events": [...]} 지원
    if isinstance(data, dict):
        events = data.get("events")
        if not isinstance(events, list):
            return []
    elif isinstance(data, list):
        events = data
    else:
        return []

    out: List[Dict[str, str]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        agent = str(ev.get("Agent", "")).strip()
        deon  = str(ev.get("Deontic", "")).strip()
        act   = str(ev.get("Action", "")).strip()
        # 세 키가 모두 존재하고 공백이 아닌 경우만 유지 (값 내용은 그대로)
        if agent and deon and act:
            out.append({"Agent": agent, "Deontic": deon, "Action": act})
    return out

# ------------------------------
# Core function
# ------------------------------
def build_eventic_graph(
    document_text: str,
    *,
    model: str = DEFAULT_MODEL
) -> List[Dict[str, str]]:
    """
    Build an Eventic Graph from a document text.
    - 모델 출력 형식만 강제(배열의 객체에 Agent/Deontic/Action 키)
    - Deontic/Agent 값에 대한 필터/정규화 없음
    """
    client = OpenAI()
    user_msg = _USER_TEMPLATE.format(doc=document_text.strip())

    if VERBOSE:
        print("[VERBOSE] Sending request to OpenAI API...")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1200,
    )

    raw = resp.choices[0].message.content
    if VERBOSE:
        print("[VERBOSE] Raw response:", raw)

    json_text = _extract_json_text(raw)
    if VERBOSE and json_text != raw:
        print("[VERBOSE] Extracted JSON block:\n", json_text)

    events = _coerce_event_list(json_text)
    return events

# ------------------------------
# 실행 예시
# ------------------------------
if __name__ == "__main__":
    sample_doc = (
        "According to GDPR Article 17, the Controller must delete personal data "
        "when the Data Subject withdraws consent. "
        "The Processor must_not transfer sensitive data outside the EU. "
        "Supervisory Authorities may impose fines for violations. "
        "Solectron will provide a commercially reasonable opportunity for acquisition."
    )
    graph = build_eventic_graph(sample_doc, model=DEFAULT_MODEL)
    print(json.dumps(graph, ensure_ascii=False, indent=2))


