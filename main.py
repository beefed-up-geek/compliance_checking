from utils import build_fusion_graph
import utils.eventic_graph_builder as evg
import json

def main():
    # 예시 문서 (GDPR 제17조 삭제권 + 처리자 의무 + 감독기관 권한)
    document = (
        "According to GDPR Article 17, the Controller must delete personal data "
        "when the Data Subject withdraws consent. "
        "The Processor must_not transfer sensitive data outside the EU. "
        "Supervisory Authorities may impose fines for violations."
    )

    # 1) 문서 → Eventic Graph
    eventic = evg.build_eventic_graph(document)
    print("\n=== Eventic Graph ===")
    print(json.dumps(eventic, ensure_ascii=False, indent=2))

    # 2) Eventic Graph → Fusion Graph
    fusion = build_fusion_graph(
        eventic,
        rounds=2,
        use_concept_graph=True,
        use_entity_graph=True,
        use_term_definition_graph=True,
        include_tdg_edges=True
    )
    print("\n=== Fusion Graph ===")
    print(json.dumps(fusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
