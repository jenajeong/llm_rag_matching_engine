"""
NetworkX 기반 그래프 저장소
엔티티-관계 그래프를 저장하고 1-hop 확장 검색 지원
"""

import pickle
from pathlib import Path
from typing import List, Dict, Set, Optional

import networkx as nx

from ..config import RAG_STORE_DIR
from ..core.safe import as_text


class GraphStore:
    """NetworkX 기반 그래프 저장소"""

    def __init__(self, store_dir: str = None, doc_type: str = "patent"):
        """
        그래프 저장소 초기화

        Args:
            store_dir: 저장 디렉토리
            doc_type: 문서 타입 (patent/article/project)
        """
        self.store_dir = Path(store_dir or RAG_STORE_DIR) / "graphs"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.doc_type = doc_type

        # NetworkX 그래프 (방향 그래프)
        self.graph = nx.DiGraph()

        # 저장 파일 경로
        self.save_path = self.store_dir / f"graph_{doc_type}.pkl"

        # 기존 그래프 로드 시도
        self._load_if_exists()

    def _load_if_exists(self):
        """기존 그래프 파일이 있으면 로드"""
        if self.save_path.exists():
            self.load()
            print(f"Loaded existing graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        else:
            print(f"Created new graph for {self.doc_type}")

    def add_entity(
        self,
        name: str,
        entity_type: str,
        doc_id: str,
        description: str = ""
    ):
        """
        엔티티 노드 추가 (동일 엔티티명이면 sources에 추가)

        Args:
            name: 엔티티 이름
            entity_type: 엔티티 타입
            doc_id: 문서 ID
            description: 설명
        """
        name = as_text(name)
        entity_type = as_text(entity_type, "UNKNOWN")
        doc_id = as_text(doc_id)
        description = as_text(description)
        if not name:
            return

        if self.graph.has_node(name):
            # 기존 노드에 source 추가
            node_data = self.graph.nodes[name]
            sources = node_data.get("sources", [])

            # 중복 체크
            if doc_id and doc_id not in sources:
                sources.append(doc_id)
                self.graph.nodes[name]["sources"] = sources

            # description이 비어있으면 업데이트
            if not node_data.get("description") and description:
                self.graph.nodes[name]["description"] = description
        else:
            # 새 노드 추가
            self.graph.add_node(
                name,
                entity_type=entity_type,
                description=description,
                sources=[doc_id] if doc_id else []
            )

    def add_relation(
        self,
        source_entity: str,
        target_entity: str,
        keywords: str,
        doc_id: str,
        description: str = ""
    ):
        """
        관계 엣지 추가 (동일 관계면 sources, keywords에 추가, weight 증가)

        Args:
            source_entity: 소스 엔티티
            target_entity: 타겟 엔티티
            keywords: 관계 키워드 (LightRAG 스타일)
            doc_id: 문서 ID
            description: 설명
        """
        source_entity = as_text(source_entity)
        target_entity = as_text(target_entity)
        keywords = as_text(keywords)
        doc_id = as_text(doc_id)
        description = as_text(description)
        if not source_entity or not target_entity:
            return

        if self.graph.has_edge(source_entity, target_entity):
            # 기존 엣지에 source 추가
            edge_data = self.graph.edges[source_entity, target_entity]
            sources = edge_data.get("sources", [])
            keywords_list = edge_data.get("keywords", [])

            if doc_id and doc_id not in sources:
                sources.append(doc_id)
                self.graph.edges[source_entity, target_entity]["sources"] = sources
                # weight 증가 (관계 빈도)
                self.graph.edges[source_entity, target_entity]["weight"] = len(sources)

            # 새 키워드 추가 (중복 제거)
            if keywords and keywords not in keywords_list:
                keywords_list.append(keywords)
                self.graph.edges[source_entity, target_entity]["keywords"] = keywords_list
        else:
            # 새 엣지 추가
            self.graph.add_edge(
                source_entity,
                target_entity,
                keywords=[keywords] if keywords else [],
                description=description,
                sources=[doc_id] if doc_id else [],
                weight=1
            )

    def add_entities_batch(self, entities: List[Dict]):
        """
        엔티티 배치 추가

        Args:
            entities: 엔티티 리스트 (name, entity_type, source_doc_id, description)
        """
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            self.add_entity(
                name=entity["name"],
                entity_type=entity.get("entity_type", "UNKNOWN"),
                doc_id=entity.get("source_doc_id", ""),
                description=entity.get("description", "")
            )

    def add_relations_batch(self, relations: List[Dict]):
        """
        관계 배치 추가

        Args:
            relations: 관계 리스트 (source_entity, target_entity, keywords, source_doc_id)
        """
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            self.add_relation(
                source_entity=relation["source_entity"],
                target_entity=relation["target_entity"],
                keywords=relation.get("keywords", ""),
                doc_id=relation.get("source_doc_id", ""),
                description=relation.get("description", "")
            )

    def get_neighbors(
        self,
        entity_name: str,
        direction: str = "both",
        hop: int = 1
    ) -> List[Dict]:
        """
        엔티티의 이웃 노드 조회 (1-hop 확장) - 관계 정보 포함

        Args:
            entity_name: 엔티티 이름
            direction: 방향 (in/out/both)
            hop: 홉 수 (기본 1)

        Returns:
            이웃 엔티티 정보 리스트 (관계 description, keywords 포함)
        """
        if not self.graph.has_node(entity_name):
            return []

        neighbors = set()

        # 현재 홉의 노드들
        current_nodes = {entity_name}

        for _ in range(hop):
            next_nodes = set()
            for node in current_nodes:
                if direction in ("out", "both"):
                    next_nodes.update(self.graph.successors(node))
                if direction in ("in", "both"):
                    next_nodes.update(self.graph.predecessors(node))

            neighbors.update(next_nodes)
            current_nodes = next_nodes

        # 결과 구성 (관계 정보 포함)
        results = []
        for neighbor in neighbors:
            if neighbor == entity_name:
                continue

            node_data = self.graph.nodes[neighbor]

            # 엣지(관계) 정보 조회 - 양방향 확인
            edge_data = None
            if self.graph.has_edge(entity_name, neighbor):
                edge_data = self.graph.edges[entity_name, neighbor]
            elif self.graph.has_edge(neighbor, entity_name):
                edge_data = self.graph.edges[neighbor, entity_name]

            results.append({
                "name": neighbor,
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "sources": node_data.get("sources", []),
                "relation_description": edge_data.get("description", "") if edge_data else "",
                "relation_keywords": edge_data.get("keywords", []) if edge_data else []
            })

        return results

    def get_entity(self, entity_name: str) -> Optional[Dict]:
        """
        엔티티 정보 조회

        Args:
            entity_name: 엔티티 이름

        Returns:
            엔티티 정보 또는 None
        """
        if not self.graph.has_node(entity_name):
            return None

        node_data = self.graph.nodes[entity_name]
        return {
            "name": entity_name,
            "entity_type": node_data.get("entity_type", "UNKNOWN"),
            "description": node_data.get("description", ""),
            "sources": node_data.get("sources", [])
        }

    def get_relations_between(
        self,
        source_entity: str,
        target_entity: str
    ) -> Optional[Dict]:
        """
        두 엔티티 간 관계 조회

        Args:
            source_entity: 소스 엔티티
            target_entity: 타겟 엔티티

        Returns:
            관계 정보 또는 None
        """
        if not self.graph.has_edge(source_entity, target_entity):
            return None

        edge_data = self.graph.edges[source_entity, target_entity]
        return {
            "source": source_entity,
            "target": target_entity,
            "keywords": edge_data.get("keywords", []),
            "description": edge_data.get("description", ""),
            "sources": edge_data.get("sources", []),
            "weight": edge_data.get("weight", 1)
        }

    def get_subgraph(
        self,
        entity_names: List[str],
        include_neighbors: bool = True
    ) -> nx.DiGraph:
        """
        특정 엔티티들의 서브그래프 추출

        Args:
            entity_names: 엔티티 이름 리스트
            include_neighbors: 이웃 포함 여부

        Returns:
            서브그래프
        """
        nodes = set(entity_names)

        if include_neighbors:
            for name in entity_names:
                if self.graph.has_node(name):
                    nodes.update(self.graph.successors(name))
                    nodes.update(self.graph.predecessors(name))

        return self.graph.subgraph(nodes).copy()

    def get_stats(self) -> Dict:
        """그래프 통계"""
        return {
            "doc_type": self.doc_type,
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges)
        }

    def save(self):
        """그래프를 파일로 저장"""
        with open(self.save_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved: {self.save_path}")

    def load(self):
        """파일에서 그래프 로드"""
        with open(self.save_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded: {self.save_path}")

    def clear(self):
        """그래프 초기화"""
        self.graph = nx.DiGraph()
        print("Graph cleared")


if __name__ == "__main__":
    # 테스트
    print("Testing GraphStore...")

    store = GraphStore(doc_type="patent")

    # 엔티티 추가
    store.add_entity(
        name="딥러닝",
        entity_type="TECHNOLOGY",
        doc_id="patent_001",
        description="심층 신경망 기반 기계학습"
    )

    # 같은 엔티티, 다른 문서
    store.add_entity(
        name="딥러닝",
        entity_type="TECHNOLOGY",
        doc_id="patent_002",
        description=""
    )

    store.add_entity(
        name="의료영상분석",
        entity_type="DOMAIN",
        doc_id="patent_001"
    )

    # 관계 추가
    store.add_relation(
        source_entity="딥러닝",
        target_entity="의료영상분석",
        keywords="image processing, diagnosis, AI application",
        doc_id="patent_001"
    )

    # 통계 확인
    print(f"\nStats: {store.get_stats()}")

    # 엔티티 조회
    entity = store.get_entity("딥러닝")
    print(f"\n딥러닝 엔티티:")
    print(f"  - sources: {entity['sources']}")

    # 1-hop 이웃 조회
    neighbors = store.get_neighbors("딥러닝")
    print(f"\n딥러닝 이웃: {[n['name'] for n in neighbors]}")

    # 저장
    store.save()
    print("\nTest completed!")
