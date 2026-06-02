from environment.graph_topology import IntersectionGraph


def test_add_nodes_and_edges():
    graph = IntersectionGraph()
    graph.add_node("A", 0, 0)
    graph.add_node("B", 100, 0)
    graph.add_edge("A", "B")
    assert "B" in graph.get_neighbors("A")


def test_shortest_path():
    graph = IntersectionGraph()
    for n in ["A", "B", "C"]:
        graph.add_node(n)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    assert graph.shortest_path_length("A", "C") == 2


def test_k_hop_neighbors():
    graph = IntersectionGraph()
    for n in ["A", "B", "C", "D"]:
        graph.add_node(n)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    neighbors = graph.get_k_hop_neighbors("A", k=2)
    assert "C" in neighbors
