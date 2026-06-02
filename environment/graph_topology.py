from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class IntersectionGraph:
    def __init__(self):
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.positions: Dict[str, Tuple[float, float]] = {}

    def add_node(self, node_id: str, x: float = 0.0, y: float = 0.0) -> None:
        self.adjacency[node_id]
        self.positions[node_id] = (x, y)

    def add_edge(self, node_a: str, node_b: str, bidirectional: bool = True) -> None:
        self.adjacency[node_a].add(node_b)
        if bidirectional:
            self.adjacency[node_b].add(node_a)

    def get_neighbors(self, node_id: str) -> List[str]:
        return sorted(self.adjacency.get(node_id, set()))

    def shortest_path_length(self, source: str, target: str) -> int:
        if source == target:
            return 0
        visited = {source}
        queue = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor in self.adjacency[node]:
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1

    def build_from_sumo(self, traci_module, max_distance: float = 500.0) -> None:
        tl_ids = traci_module.trafficlight.getIDList()
        for tl_id in tl_ids:
            x, y = traci_module.junction.getPosition(
                traci_module.trafficlight.getControlledJunctions(tl_id)[0]
            )
            self.add_node(tl_id, x, y)
        for i, tl_a in enumerate(tl_ids):
            pos_a = self.positions[tl_a]
            for tl_b in tl_ids[i + 1 :]:
                pos_b = self.positions[tl_b]
                dist = ((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2) ** 0.5
                if dist <= max_distance:
                    self.add_edge(tl_a, tl_b)

    def get_k_hop_neighbors(self, node_id: str, k: int = 2) -> List[str]:
        result = set()
        frontier = {node_id}
        for _ in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency[node]:
                    if neighbor != node_id:
                        result.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return sorted(result)

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: sorted(v) for k, v in self.adjacency.items()}
