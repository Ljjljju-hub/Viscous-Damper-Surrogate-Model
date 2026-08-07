import torch_geometric.transforms as T


def build_graph_transform():
    """Build dynamic edge geometry from the restored mesh positions."""
    return T.Compose(
        [
            T.FaceToEdge(remove_faces=False),
            T.Cartesian(norm=False, cat=True),
            T.Distance(norm=False, cat=True),
        ]
    )


def prepare_graph(graph, transform):
    graph = transform(graph)
    if graph.edge_attr is None or graph.edge_attr.shape[-1] != 3:
        raise ValueError(
            "Expected edge_attr=[delta_R, delta_Z, distance] with size 3, "
            f"got {None if graph.edge_attr is None else graph.edge_attr.shape}."
        )
    return graph
