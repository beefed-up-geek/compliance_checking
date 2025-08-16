# utils/entity_graph_search.py
from __future__ import annotations
import re
import json
import traceback
import requests
from typing import Dict, List, Iterable, Optional
from urllib.parse import quote

DBPEDIA_LOOKUP = "https://lookup.dbpedia.org/api/search"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"

# ----------------------------------------------------
# 내부 유틸 & 로깅
# ----------------------------------------------------
def _normalize_keyword_for_dbr(keyword: str) -> str:
    """
    Lookup 실패 시 폴백용: "new york city" -> "New_York_City"
    DBpedia 리소스 로컬명 관례에 맞춰 단어별 capitalize + '_' 연결
    """
    tokenized = re.findall(r"[A-Za-z0-9]+", keyword.strip().lower())
    return "_".join(t.capitalize() for t in tokenized) if tokenized else keyword.strip()

def _localname(uri: str) -> str:
    """URI의 말단 토큰: http://dbpedia.org/ontology/birthPlace -> birthPlace"""
    if not uri:
        return ""
    return uri.rstrip("/").rsplit("/", 1)[-1]

def _headers_json():
    # Lookup 등 일반 JSON 응답용
    return {"Accept": "application/json"}

def _headers_sparql_results_json():
    # SELECT / ASK 결과셋(JSON)용 (Virtuoso가 요구)
    # Ref: SPARQL 1.1 Query Results JSON Format
    return {"Accept": "application/sparql-results+json"}

def dbg(verbose: bool, *args):
    if verbose:
        print("[DBG]", *args)

# ----------------------------------------------------
# 0) 위키 메타 관계 식별 (기본 제외)
# ----------------------------------------------------
WIKI_REL_LOCAL_PREFIX = "wikiPage"
WIKI_REL_LOCAL_BLOCKLIST = {
    # 위키 링크/템플릿/리다이렉트/리비전/ID/인터랭크 등 (Reserved for DBpedia)
    "wikiPageWikiLink",
    "wikiPageExternalLink",
    "wikiPageRedirects",
    "wikiPageDisambiguates",
    "wikiPageInterLanguageLink",
    "wikiPageUsesTemplate",
    "wikiPageEditLink",
    "wikiPageHistoryLink",
    "wikiPageRevisionLink",
    "wikiPageWikiLinkText",
    "wikiPageID",
    "wikiPageRevisionID",
    "wikidataSplitIri",
    "22-rdf-syntax-ns#type",
    "rdf-schema#seeAlso",
}

def _is_wiki_relation(p_uri: str, rel_local: str) -> bool:
    """DBpedia의 위키 내비게이션/메타 성격의 관계면 True."""
    if not rel_local:
        return False
    if rel_local in WIKI_REL_LOCAL_BLOCKLIST:
        return True
    if rel_local.startswith(WIKI_REL_LOCAL_PREFIX):
        return True
    # URI 수준 보수적 체크
    return "dbpedia.org/ontology/wikiPage" in (p_uri or "")

# ----------------------------------------------------
# 1) 키워드 → DBpedia 리소스 URI 해석
# ----------------------------------------------------
def resolve_dbpedia_entity(keyword: str, *, timeout: int = 30, verbose: bool = False) -> Optional[str]:
    """
    DBpedia Lookup API로 키워드를 엔티티 URI로 해석.
    실패하면 dbr:{Normalized} 폴백을 시도 (존재 여부를 SPARQL로 검증).
    """
    dbg(verbose, f"resolve_dbpedia_entity() keyword={keyword!r}")
    try:
        params = {"query": keyword, "maxResults": 5, "format": "json"}
        dbg(verbose, "GET", DBPEDIA_LOOKUP, params)
        resp = requests.get(DBPEDIA_LOOKUP, params=params, headers=_headers_json(), timeout=timeout)
        dbg(verbose, "lookup status_code:", resp.status_code)
        if resp.ok:
            data = resp.json() or {}
            results = data.get("docs") or []
            dbg(verbose, f"lookup results len={len(results)}")
            for i, r in enumerate(results):
                uri = None
                for key in ("resource", "uri"):
                    val = r.get(key)
                    if isinstance(val, list) and val:
                        uri = val[0]
                        break
                    if isinstance(val, str):
                        uri = val
                        break
                dbg(verbose, f"candidate[{i}] uri={uri}")
                if uri and isinstance(uri, str) and uri.startswith("http"):
                    dbg(verbose, "-> chosen uri:", uri)
                    return uri
        else:
            dbg(verbose, "lookup not ok. text:", resp.text[:500])
    except Exception as e:
        dbg(verbose, "lookup exception:", repr(e))
        if verbose:
            traceback.print_exc()

    # 폴백: dbr:Normalized 로컬명 생성 후 존재 검증
    local = _normalize_keyword_for_dbr(keyword)
    candidate = f"http://dbpedia.org/resource/{quote(local)}"
    dbg(verbose, f"fallback candidate={candidate}")
    exists = _check_resource_exists(candidate, timeout=timeout, verbose=verbose)
    dbg(verbose, "fallback exists?", exists)
    return candidate if exists else None

def _check_resource_exists(uri: str, *, timeout: int = 30, verbose: bool = False) -> bool:
    """
    간단 존재 확인: ASK { <uri> ?p ?o } 를 SPARQL로 질의.
    """
    ask = f"ASK WHERE {{ <{uri}> ?p ?o }}"
    try:
        dbg(verbose, "ASK query:", ask)
        resp = requests.get(
            DBPEDIA_SPARQL,
            params={"query": ask, "format": "application/sparql-results+json"},
            timeout=timeout,
            headers=_headers_sparql_results_json(),  # 중요: 결과셋 JSON 헤더
        )
        dbg(verbose, "ASK status_code:", resp.status_code)
        if resp.ok:
            res = resp.json()
            dbg(verbose, "ASK result:", res)
            return bool(res.get("boolean"))
        else:
            dbg(verbose, "ASK not ok. text:", resp.text[:500])
    except Exception as e:
        dbg(verbose, "ASK exception:", repr(e))
        if verbose:
            traceback.print_exc()
        return False
    return False

# ----------------------------------------------------
# 2) 엔티티 → outgoing triples 조회
# ----------------------------------------------------
def fetch_dbpedia_triples(
    keyword: str,
    *,
    relations: Optional[Iterable[str]] = None,
    max_num: Optional[int] = None,
    limit: int = 200,
    timeout: int = 30,
    verbose: bool = False,
    exclude_wiki: bool = True,   # 기본: 위키 메타 관계 제외
) -> List[Dict[str, str]]:
    """
    DBpedia에서 keyword(영어)를 엔티티로 해석하고, 그 엔티티의 outgoing triple을
    [{'source': ..., 'relation': ..., 'target': ...}, ...] 로 반환.

    Parameters
    ----------
    keyword : 조회할 키워드(영어 자연어)
    relations : 허용할 관계명 집합 (로컬명 기준, 예: {"type","birthPlace","genre"})
                - 관계 URI의 마지막 토큰(localname)으로 매칭
                - 지정하지 않으면 모든 관계 허용
    max_num : 최종 반환할 최대 triple 개수 (기본 None → 제한 없음)
    limit : SPARQL LIMIT (기본 200; 상한 2000)
    timeout : 요청 타임아웃(초)
    verbose : True면 단계별 디버깅 로그 출력
    exclude_wiki : True면 wikiPage* 등 위키 메타 관계 제외

    Returns
    -------
    List[Dict[str,str]] : (source, relation, target) 리스트. 라벨은 영어 우선.
    """
    dbg(verbose, f"fetch_dbpedia_triples() keyword={keyword!r}, limit={limit}, max_num={max_num}, exclude_wiki={exclude_wiki}")
    uri = resolve_dbpedia_entity(keyword, timeout=timeout, verbose=verbose)
    dbg(verbose, "resolved uri:", uri)
    if not uri:
        dbg(verbose, "No URI resolved. Returning [].")
        return []

    sparql = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?p ?pLabel ?o ?oLabel WHERE {{
      <{uri}> ?p ?o .
      OPTIONAL {{ ?p rdfs:label ?pLabel FILTER (lang(?pLabel) = 'en') }}
      OPTIONAL {{ ?o rdfs:label ?oLabel FILTER (lang(?oLabel) = 'en') }}
    }} LIMIT {int(max(1, min(limit, 2000)))}
    """.strip()

    try:
        dbg(verbose, "SPARQL query:\n" + sparql)
        resp = requests.get(
            DBPEDIA_SPARQL,
            params={"query": sparql, "format": "application/sparql-results+json"},
            timeout=timeout,
            headers=_headers_sparql_results_json(),  # 중요: 결과셋 JSON 헤더
        )
        dbg(verbose, "SPARQL status_code:", resp.status_code)
        if not resp.ok:
            dbg(verbose, "SPARQL not ok. text:", resp.text[:800])
            return []
        data = resp.json()
        if verbose:
            print("[DBG] SPARQL raw keys:", list(data.keys()))
            print("[DBG] SPARQL head:", data.get("head"))
            print("[DBG] SPARQL sample binding:", (data.get("results", {}).get("bindings") or [None])[:1])
    except Exception as e:
        dbg(verbose, "SPARQL exception:", repr(e))
        if verbose:
            traceback.print_exc()
        return []

    triples: List[Dict[str, str]] = []
    source_label = _localname(uri).replace("_", " ")
    seen = set()
    bindings = data.get("results", {}).get("bindings", [])
    dbg(verbose, f"bindings len={len(bindings)}")

    for idx, b in enumerate(bindings):
        try:
            p_uri = b.get("p", {}).get("value", "")
            p_label = b.get("pLabel", {}).get("value")
            o_val = b.get("o", {}).get("value", "")
            o_type = b.get("o", {}).get("type", "")
            o_label = b.get("oLabel", {}).get("value")

            rel_local = _localname(p_uri)

            # 1) 위키 메타 관계 제외 (옵션)
            if exclude_wiki and _is_wiki_relation(p_uri, rel_local):
                if verbose:
                    print("[DBG] skip wiki-meta relation:", rel_local)
                continue

            # 2) 사용자 지정 관계 필터 (localname 기준)
            if relations and rel_local not in relations:
                continue

            relation_text = p_label or rel_local
            if o_type == "uri":
                target_text = (o_label or _localname(o_val)).replace("_", " ")
            else:
                target_text = o_label or o_val  # literal

            sig = (source_label.lower(), relation_text, target_text.lower())
            if sig in seen:
                continue

            triples.append({"source": source_label, "relation": relation_text, "target": target_text})
            seen.add(sig)

            if verbose and (idx < 5):  # 초반 5개 샘플 로그
                print("[DBG] triple:", triples[-1])

            if max_num is not None and len(triples) >= max_num:
                dbg(verbose, f"max_num reached: {max_num}")
                break

        except Exception as e:
            dbg(verbose, f"binding[{idx}] parse exception: {repr(e)}")
            if verbose:
                traceback.print_exc()
            continue

    dbg(verbose, f"triples count={len(triples)}")
    return triples

# ----------------------------------------------------
# 간단 실행 진단
# ----------------------------------------------------
if __name__ == "__main__":
    # 실행 시 기본 디버그 ON
    keyword = "Elon Musk"
    print(f"[RUN] fetch_dbpedia_triples('{keyword}') with verbose=True (exclude_wiki=True)")
    out = fetch_dbpedia_triples(keyword, relations=None, max_num=10, limit=2000, verbose=True, exclude_wiki=True)
    print("[RUN] result count:", len(out))
    for t in out:
        print(t)
