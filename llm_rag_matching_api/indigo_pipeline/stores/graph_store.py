import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from ..config import RAG_STORE_DIR
from ..core.safe import as_text, split_csv


def _source_set(value) -> Set[str]:
    if isinstance(value, (list, tuple, set)):
        sources: Set[str] = set()
        for item in value:
            sources.update(split_csv(item))
        return sources
    return split_csv(value)


def _node_key(name: str, doc_id: str) -> str:
    name = as_text(name)
    doc_id = as_text(doc_id)
    return f"{doc_id}::{name}" if doc_id else name


def _display_name(node_id: str, data: Dict) -> str:
    return as_text(data.get("name")) or as_text(node_id).split("::", 1)[-1]


class GraphStore:
    """NetworkX graph store with append-only, document-scoped keys."""

    def __init__(self, store_dir: str = None, doc_type: str = "patent"):
        self.store_dir = Path(store_dir or RAG_STORE_DIR) / "graphs"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.doc_type = doc_type
        self.graph = nx.DiGraph()
        self.save_path = self.store_dir / f"graph_{doc_type}.pkl"
        self._load_if_exists()

    def _load_if_exists(self):
        if self.save_path.exists():
            self.load()
            print(f"Loaded existing graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        else:
            print(f"Created new graph for {self.doc_type}")

    def add_entity(self, name: str, entity_type: str, doc_id: str, description: str = ""):
        name = as_text(name)
        entity_type = as_text(entity_type, "UNKNOWN")
        doc_id = as_text(doc_id)
        description = as_text(description)
        if not name or not doc_id:
            return

        node_id = _node_key(name, doc_id)
        if self.graph.has_node(node_id):
            return

        self.graph.add_node(
            node_id,
            name=name,
            entity_type=entity_type,
            description=description,
            sources=[doc_id],
            source_doc_id=doc_id,
        )

    def add_relation(
        self,
        source_entity: str,
        target_entity: str,
        keywords: str,
        doc_id: str,
        description: str = "",
    ):
        source_entity = as_text(source_entity)
        target_entity = as_text(target_entity)
        keywords = as_text(keywords)
        doc_id = as_text(doc_id)
        description = as_text(description)
        if not source_entity or not target_entity or not doc_id:
            return

        source_node = _node_key(source_entity, doc_id)
        target_node = _node_key(target_entity, doc_id)
        if not self.graph.has_node(source_node):
            self.add_entity(source_entity, "UNKNOWN", doc_id)
        if not self.graph.has_node(target_node):
            self.add_entity(target_entity, "UNKNOWN", doc_id)
        if self.graph.has_edge(source_node, target_node):
            return

        self.graph.add_edge(
            source_node,
            target_node,
            source_entity=source_entity,
            target_entity=target_entity,
            keywords=sorted(split_csv(keywords)),
            description=description,
            sources=[doc_id],
            source_doc_id=doc_id,
            weight=1,
        )

    def add_entities_batch(self, entities: List[Dict]):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            self.add_entity(
                name=entity.get("name", ""),
                entity_type=entity.get("entity_type", "UNKNOWN"),
                doc_id=entity.get("source_doc_id", ""),
                description=entity.get("description", ""),
            )

    def add_relations_batch(self, relations: List[Dict]):
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            self.add_relation(
                source_entity=relation.get("source_entity", ""),
                target_entity=relation.get("target_entity", ""),
                keywords=relation.get("keywords", ""),
                doc_id=relation.get("source_doc_id", ""),
                description=relation.get("description", ""),
            )

    def _matching_entity_nodes(self, entity_name: str) -> List[str]:
        entity_name = as_text(entity_name)
        return [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if _display_name(node_id, data) == entity_name
        ]

    def get_neighbors(self, entity_name: str, direction: str = "both", hop: int = 1) -> List[Dict]:
        matching_nodes = set(self._matching_entity_nodes(entity_name))
        if not matching_nodes:
            return []

        neighbors = set()
        current_nodes = set(matching_nodes)
        for _ in range(hop):
            next_nodes = set()
            for node in current_nodes:
                if direction in ("out", "both"):
                    next_nodes.update(self.graph.successors(node))
                if direction in ("in", "both"):
                    next_nodes.update(self.graph.predecessors(node))
            neighbors.update(next_nodes)
            current_nodes = next_nodes

        results = []
        for neighbor in neighbors:
            if neighbor in matching_nodes:
                continue
            node_data = self.graph.nodes[neighbor]
            edge_data = None
            for source_node in matching_nodes:
                if self.graph.has_edge(source_node, neighbor):
                    edge_data = self.graph.edges[source_node, neighbor]
                    break
                if self.graph.has_edge(neighbor, source_node):
                    edge_data = self.graph.edges[neighbor, source_node]
                    break
            results.append({
                "name": _display_name(neighbor, node_data),
                "entity_type": node_data.get("entity_type", "UNKNOWN"),
                "description": node_data.get("description", ""),
                "sources": node_data.get("sources", []),
                "relation_description": edge_data.get("description", "") if edge_data else "",
                "relation_keywords": edge_data.get("keywords", []) if edge_data else [],
            })
        return results

    def get_entity(self, entity_name: str) -> Optional[Dict]:
        matching_nodes = self._matching_entity_nodes(entity_name)
        if not matching_nodes:
            return None

        sources = set()
        descriptions = []
        entity_type = "UNKNOWN"
        for node_id in matching_nodes:
            node_data = self.graph.nodes[node_id]
            sources.update(_source_set(node_data.get("sources", [])))
            if node_data.get("description"):
                descriptions.append(as_text(node_data.get("description")))
            if entity_type == "UNKNOWN" and node_data.get("entity_type"):
                entity_type = node_data.get("entity_type", "UNKNOWN")

        return {
            "name": entity_name,
            "entity_type": entity_type,
            "description": descriptions[0] if descriptions else "",
            "sources": sorted(sources),
        }

    def get_relations_between(self, source_entity: str, target_entity: str) -> Optional[Dict]:
        source_nodes = self._matching_entity_nodes(source_entity)
        target_nodes = set(self._matching_entity_nodes(target_entity))
        edge_matches = [
            self.graph.edges[src, dst]
            for src in source_nodes
            for dst in self.graph.successors(src)
            if dst in target_nodes
        ]
        if not edge_matches:
            return None

        sources = set()
        keywords = set()
        descriptions = []
        for edge_data in edge_matches:
            sources.update(_source_set(edge_data.get("sources", [])))
            keywords.update(_source_set(edge_data.get("keywords", [])))
            if edge_data.get("description"):
                descriptions.append(as_text(edge_data.get("description")))

        return {
            "source": source_entity,
            "target": target_entity,
            "keywords": sorted(keywords),
            "description": descriptions[0] if descriptions else "",
            "sources": sorted(sources),
            "weight": len(sources) if sources else len(edge_matches),
        }

    def get_subgraph(self, entity_names: List[str], include_neighbors: bool = True) -> nx.DiGraph:
        nodes = {node for name in entity_names for node in self._matching_entity_nodes(name)}
        if include_neighbors:
            for node in list(nodes):
                if self.graph.has_node(node):
                    nodes.update(self.graph.successors(node))
                    nodes.update(self.graph.predecessors(node))
        return self.graph.subgraph(nodes).copy()

    def get_stats(self) -> Dict:
        return {
            "doc_type": self.doc_type,
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
        }

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved: {self.save_path}")

    def load(self):
        with open(self.save_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded: {self.save_path}")

    def clear(self):
        self.graph = nx.DiGraph()
        print("Graph cleared")
