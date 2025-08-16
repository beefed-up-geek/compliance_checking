# utils/concept_graph_search.py
from __future__ import annotations
import requests
from urllib.parse import quote
from typing import Dict, List, Iterable, Optional

def _norm_term(term: str) -> str:
    """ConceptNet URI용 정규화: 소문자 + 공백을 '_'로."""
    return quote(term.strip().lower().replace(" ", "_"))

def _label_of(node: Dict) -> str:
    """사람이 읽기 좋은 라벨을 우선 사용, 없으면 ID의 말단 토큰."""
    return node.get("label") or node.get("@id", "").rsplit("/", 1)[-1]

def _rel_of(rel_obj: Dict) -> str:
    """관계 /r/RelatedTo → RelatedTo 로 변환."""
    rid = rel_obj.get("@id", "")
    return rid.split("/")[-1] if rid else ""

def fetch_conceptnet_triples(
    keyword: str,
    *,
    limit: int = 100,
    relations: Optional[Iterable[str]] = None,
    min_weight: float = 0.0,
    max_num: Optional[int] = None,
    timeout: int = 30,
) -> List[Dict[str, str]]:
    """
    ConceptNet(영어)에서 keyword 관련 1-hop 삼중항을 조회하여
    [{'source': ..., 'relation': ..., 'target': ...}, ...] 형태로 반환.

    Parameters
    ----------
    keyword : 조회할 키워드(영어)
    limit : API limit (최대 1000 권장 이하, 기본값 100)
    relations : 필터링할 관계명 집합 (예: {"IsA","CapableOf"})
    min_weight : weight의 최소값 (기본값 0.0)
        - ConceptNet weight 범위:
            * 0 ~ 1 : 거의 의미 없는 약한 연결
            * 1 ~ 2 : 낮은 신뢰도의 엣지
            * 2 ~ 3 : 중간 정도 신뢰도
            * 5 이상 : 강한 연결 (다수 출처 확인됨)
            * 20 이상 : 매우 확실한 대표적 관계 (예: dog IsA animal)
    max_num : 최종 반환할 triple 개수 (기본값 None → 무제한)
              weight 높은 순으로 정렬하여 잘라냄
    timeout : 요청 타임아웃(초)

    Returns
    -------
    List[Dict]: 삼중항 리스트 (중복 제거, weight 높은 순 정렬)
    """
    term = _norm_term(keyword)
    url = f"https://api.conceptnet.io/c/en/{term}"
    params = {"limit": max(10, min(int(limit), 1000))}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    triples: List[Dict[str, str]] = []
    seen = set()

    for edge in data.get("edges", []):
        w = edge.get("weight", 0)
        if w < min_weight:
            continue
        rel = _rel_of(edge.get("rel", {}))
        if relations and rel not in relations:
            continue

        src = _label_of(edge.get("start", {}))
        tgt = _label_of(edge.get("end", {}))
        sig = (src.lower(), rel, tgt.lower())
        if sig in seen:
            continue

        triples.append({
            "source": src,
            "relation": rel,
            "target": tgt,
            "weight": w  # weight도 같이 반환
        })
        seen.add(sig)

    # weight 높은 순으로 정렬
    triples.sort(key=lambda x: x["weight"], reverse=True)

    # max_num 제한 적용
    if max_num is not None:
        triples = triples[:max_num]

    # 🔑 weight 항목 제거
    for t in triples:
        t.pop("weight", None)

    return triples


# 간단 테스트
if __name__ == "__main__":
    triples = fetch_conceptnet_triples("dog", min_weight=2.0, max_num=5)
    for t in triples:
        print(t)
