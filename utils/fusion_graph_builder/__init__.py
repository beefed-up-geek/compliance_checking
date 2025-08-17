# utils/fusion_graph_builder/__init__.py
from __future__ import annotations
import json
import re
from typing import Dict, List, Iterable, Optional, Tuple, Set
from openai import OpenAI

# 내부 모듈
from .concept_graph_search import fetch_conceptnet_triples
from .entity_graph_search import fetch_dbpedia_triples
from .term_definition_graph import build_term_definition_triples

# ==============================
# 하이퍼 파라미터 (전역 설정)
# ==============================
DEFAULT_MODEL = "gpt-4o-mini"

# 확장 강도/범위
CONCEPT_MIN_WEIGHT: float = 4
CONCEPT_MAX: Optional[int] = 3
ENTITY_MAX: Optional[int] = 3
ROUNDS: int = 2

# TDG 처리
INCLUDE_TDG_EDGES: bool = False   # TDG 엣지를 최종 그래프에 포함할지

# Ablation (기본 False = 꺼짐) — 필요 시 True로 켜기
USE_CONCEPT_GRAPH: bool = True
USE_ENTITY_GRAPH: bool = True
USE_TERM_DEFINITION_GRAPH: bool = True

# 로깅
VERBOSE: bool = True


# ==============================
# 유틸
# ==============================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _is_agent_key(k: str) -> bool:
    k_l = (k or "").lower()
    return k_l == "agent" or k_l.startswith("agent")

def _extract_agents_from_eventic(eventic_edges: List[Dict[str, str]]) -> List[str]:
    agents: List[str] = []
    for edge in eventic_edges:
        cand = None
        for k, v in edge.items():
            if _is_agent_key(k):
                cand = v
                break
        if cand:
            agents.append(_norm(str(cand)))
    uniq, seen = [], set()
    for a in agents:
        if a and a.lower() not in seen:
            uniq.append(a); seen.add(a.lower())
    return uniq

def _eventic_to_triples(eventic_edges: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    {Agent, Deontic, Action} → (source=Agent, relation=Deontic, target=Action)
    """
    out: List[Dict[str, str]] = []
    for row in eventic_edges:
        agent = row.get("Agent") or row.get("agent") or row.get("Agent1") or row.get("agent1") or ""
        deon  = row.get("Deontic") or row.get("deontic") or ""
        act   = row.get("Action") or row.get("action") or ""
        if agent and deon and act:
            out.append({
                "source": _norm(str(agent)),
                "relation": _norm(str(deon)),
                "target": _norm(str(act)),
                "source_graph": "eventic",
            })
    return out

def _dedup_edges(edges: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []
    for e in edges:
        src = _norm(e.get("source", "")); rel = _norm(e.get("relation", "")); tgt = _norm(e.get("target", ""))
        g   = e.get("source_graph", "")
        sig = (src.lower(), rel, tgt.lower(), g)
        if not src or not rel or not tgt: continue
        if sig in seen: continue
        out.append({"source": src, "relation": rel, "target": tgt, "source_graph": g})
        seen.add(sig)
    return out

def _print_edges(title: str, triples: List[Dict[str, str]], graph_tag: str, *, enqueued: bool):
    """VERBOSE 모드에서 그래프 확장 결과를 보기 좋게 출력."""
    if not VERBOSE:
        return
    print(f"[VERBOSE] {title} → {len(triples)} edge(s)")
    for t in triples:
        src = _norm(t.get("source", "")); rel = _norm(t.get("relation", "")); tgt = _norm(t.get("target", ""))
        print(f"  - [{graph_tag}] {src} -{rel}-> {tgt}")
    print(f"[VERBOSE]   ⤷ targets enqueued: {len(triples) if enqueued else 0}")


# ==============================
# GPT: 어려운/고유명사 Agent 선별 (few-shot 포함)
# ==============================
_SYSTEM_FILTER = (
    "You are a selective filter. From the given list of candidate agents, "
    "return ONLY proper nouns, named entities, or domain-specific difficult terms.\n"
    "Output pure JSON: {\"keep\": [\"...\", \"...\"]}. No extra text.\n\n"
    "Examples:\n"
    "Candidates:\n"
    "- apple\n"
    "- Contract\n"
    "- Elon Musk\n"
    "- regulation\n"
    "Return:\n"
    "{\"keep\": [ \"Elon Musk\"]}\n\n"
    "Candidates:\n"
    "- consent\n"
    "- Tesla\n"
    "- obligation\n"
    "- GDPR\n"
    "Return:\n"
    "{\"keep\": [\"Tesla\", \"GDPR\"]}"
    "Candidates:\n"
    "- Film producers\n"
    "- Seller\n"
    "- Moussa Bakayokon\n"
    "- 1931 establishments in New York City\n"
    "Return:\n"
    "{\"keep\": [\"Moussa Bakayoko\"]}"
    "== end of example =="
)

_USER_FILTER_TEMPLATE = "Candidates:\n{items}\n\nReturn:\n{{\n  \"keep\": [\"term1\", \"term2\", \"...\"]\n}}\n"

def _pick_hard_agents_via_gpt(cands: List[str], model: str = DEFAULT_MODEL) -> List[str]:
    if not cands:
        return []
    client = OpenAI()
    user_msg = _USER_FILTER_TEMPLATE.format(items="\n".join(f"- {c}" for c in cands))
    if VERBOSE:
        print(f"[VERBOSE] [Filter] candidates={cands}")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_FILTER},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=400,
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        data = json.loads(m.group(1)) if m else {"keep": []}
    keep = data.get("keep") or []
    if VERBOSE:
        print(f"[VERBOSE] [Filter] kept={keep}")
    # 원본 후보와 교집합으로 정제
    cand_map = {c.lower(): c for c in cands}
    out, seen = [], set()
    for t in keep:
        k = _norm(str(t)).lower()
        if k in cand_map and k not in seen:
            out.append(cand_map[k]); seen.add(k)
    return out


# ==============================
# 1라운드 확장 (요구 로직 반영 + 상세 로그)
# ==============================
def _expand_once(
    agent_list: List[str],
    *,
    model: str = DEFAULT_MODEL,
    concept_min_weight: float = CONCEPT_MIN_WEIGHT,
    concept_max: Optional[int] = CONCEPT_MAX,
    entity_max: Optional[int] = ENTITY_MAX,
    include_tdg_edges: bool = INCLUDE_TDG_EDGES,
    use_concept_graph: bool = USE_CONCEPT_GRAPH,
    use_entity_graph: bool = USE_ENTITY_GRAPH,
    use_term_definition_graph: bool = USE_TERM_DEFINITION_GRAPH,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    1) agent_list → GPT 필터(고유명사/어려운 용어)
    2) 필터된 리스트의 각 항목에 대해:
       - ConceptNet 1-hop (사용 시; target enqueue)
       - DBpedia 1-hop (사용 시; target enqueue)
       - TDG 단일 정의 (사용 시; enqueue 안 함)
    3) 결과 엣지와 다음 라운드 큐(next_queue) 반환
    """
    if VERBOSE:
        print(f"\n[VERBOSE] === Expand start === agents={agent_list}")

    edges: List[Dict[str, str]] = []
    next_queue: List[str] = []

    hard_agents = _pick_hard_agents_via_gpt(agent_list, model=model)
    # 폴백: GPT가 아무 것도 고르지 않으면 전체 후보를 사용
    if not hard_agents:
        if VERBOSE:
            print("[VERBOSE] [Filter] empty → fallback to all agents")
        hard_agents = list(agent_list)

    for agent in hard_agents:
        if VERBOSE:
            print(f"[VERBOSE] [Agent] '{agent}'")

        # ----- Concept Graph -----
        if use_concept_graph:
            if VERBOSE:
                print(f"[VERBOSE] >>> Concept Graph: expand '{agent}' (min_w={concept_min_weight}, max={concept_max})")
            c_tris = fetch_conceptnet_triples(agent, min_weight=concept_min_weight, max_num=concept_max) or []
            for t in c_tris:
                edges.append({**t, "source_graph": "concept"})
                next_queue.append(t["target"])
            _print_edges("Concept Graph result", c_tris, "concept", enqueued=True if c_tris else False)

        # ----- Entity Graph -----
        if use_entity_graph:
            if VERBOSE:
                print(f"[VERBOSE] >>> Entity Graph: expand '{agent}' (max={entity_max})")
            e_tris = fetch_dbpedia_triples(agent, max_num=entity_max, exclude_wiki=True) or []
            for t in e_tris:
                edges.append({**t, "source_graph": "entity"})
                next_queue.append(t["target"])
            _print_edges("Entity Graph result", e_tris, "entity", enqueued=True if e_tris else False)

        # ----- Term Definition Graph -----
        if use_term_definition_graph:
            if VERBOSE:
                print(f"[VERBOSE] >>> Term Definition Graph: expand '{agent}' (single IsA)")
            tdg_tris = build_term_definition_triples(agent, model=model) or []
            if include_tdg_edges:
                for t in tdg_tris:
                    edges.append({**t, "source_graph": "term_definition"})
            _print_edges("Term Definition Graph result", tdg_tris, "term_definition", enqueued=False)

    # 다음 라운드 큐 구성(CQ/EG target만)
    current_set = {a.lower() for a in agent_list}
    uniq, seen = [], set()
    for x in next_queue:
        xn = _norm(x)
        if not xn: continue
        k = xn.lower()
        if k in current_set or k in seen: continue
        uniq.append(xn); seen.add(k)

    edges = _dedup_edges(edges)
    if VERBOSE:
        print(f"[VERBOSE] === Expand end === added_edges={len(edges)}, next_queue={uniq}")
    return edges, uniq


# ==============================
# 공개 API
# ==============================
def build_fusion_graph(
    eventic_edges: List[Dict[str, str]],
    *,
    rounds: int = ROUNDS,
    model: str = DEFAULT_MODEL,
    concept_min_weight: float = CONCEPT_MIN_WEIGHT,
    concept_max: Optional[int] = CONCEPT_MAX,
    entity_max: Optional[int] = ENTITY_MAX,
    include_tdg_edges: bool = INCLUDE_TDG_EDGES,
    use_concept_graph: bool = USE_CONCEPT_GRAPH,
    use_entity_graph: bool = USE_ENTITY_GRAPH,
    use_term_definition_graph: bool = USE_TERM_DEFINITION_GRAPH,
) -> Dict[str, List[Dict[str, str]]]:
    """
    최종 그래프(JSON):
    {
      "edges": [
        {"source":"...", "relation":"...", "target":"...", "source_graph":"eventic|concept|entity|term_definition"},
        ...
      ]
    }
    """
    if VERBOSE:
        print("[VERBOSE] Build fusion graph start")
        print(f"[VERBOSE] Graphs enabled → CG={use_concept_graph}, EG={use_entity_graph}, "
              f"TDG={use_term_definition_graph} (include_tdg_edges={include_tdg_edges})")

    final_edges: List[Dict[str, str]] = _eventic_to_triples(eventic_edges)
    if VERBOSE:
        print(f"[VERBOSE] Seed from eventic: {len(final_edges)} edges")
    agents_queue: List[str] = _extract_agents_from_eventic(eventic_edges)
    if VERBOSE:
        print(f"[VERBOSE] Initial agent queue: {agents_queue}")

    for round_idx in range(max(0, int(rounds))):
        if not agents_queue:
            if VERBOSE:
                print(f"[VERBOSE] Queue empty → stop at round {round_idx+1}")
            break
        if VERBOSE:
            print(f"\n[VERBOSE] --- Round {round_idx+1} ---")

        step_edges, new_queue = _expand_once(
            agents_queue,
            model=model,
            concept_min_weight=concept_min_weight,
            concept_max=concept_max,
            entity_max=entity_max,
            include_tdg_edges=include_tdg_edges,
            use_concept_graph=use_concept_graph,
            use_entity_graph=use_entity_graph,
            use_term_definition_graph=use_term_definition_graph,
        )
        final_edges.extend(step_edges)
        agents_queue = new_queue
        if VERBOSE:
            print(f"[VERBOSE] Round {round_idx+1} end → total_edges={len(final_edges)}")

    final_edges = _dedup_edges(final_edges)
    if VERBOSE:
        print(f"[VERBOSE] Build fusion graph end → unique_edges={len(final_edges)}")
    return {"edges": final_edges}

