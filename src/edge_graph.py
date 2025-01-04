import torch

def create_new_graph(edge_index, edge_attr):
    """
    Convert an original graph's edges into a new graph where:
      - Nodes represent edges of the original graph.
      - Edges exist between nodes if the corresponding edges in the original graph share a vertex.
    
    Parameters:
    - edge_index: torch.Tensor of shape [2, E], where E is the number of edges in the original graph.
    - edge_attr: torch.Tensor of shape [E], representing edge attributes in the original graph.
    
    Returns:
    - new_edge_index: torch.Tensor of shape [2, E'], where E' is the number of edges in the new graph.
    - node_attr: torch.Tensor of shape [E], representing node attributes in the new graph.
    """
    # Number of edges in the original graph
    num_edges = edge_index.size(1)
    num_edges=int(num_edges/2)
    # print(num_edges/2)
    edge_index = edge_index[:,:num_edges]

    # Create a mapping of nodes to edges
    node_to_edges = {}
    for edge_id in range(num_edges):
        u, v = edge_index[:, edge_id].tolist()
        if u not in node_to_edges:
            node_to_edges[u] = []
        if v not in node_to_edges:
            node_to_edges[v] = []
        node_to_edges[u].append(edge_id)
        node_to_edges[v].append(edge_id)

    # Generate new edges for the new graph
    new_edges = []
    for connected_edges in node_to_edges.values():
        # Create all pairs of connected edges
        for i in range(len(connected_edges)):
            for j in range(i + 1, len(connected_edges)):
                new_edges.append((connected_edges[i], connected_edges[j]))

    # Remove duplicate edges and ensure proper ordering
    new_edges = list(set(tuple(sorted(edge)) for edge in new_edges))
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).T.view(2,-1)  # Transpose for [2, E'] format
    # print(new_edge_index.shape)
    # assert 1==2

    # Node attributes of the new graph are the edge attributes of the original graph
    if edge_attr.shape[0]==0:
        node_attr=torch.zeros(1,edge_attr.shape[1])
    else:
        node_attr = edge_attr[:num_edges]

    return new_edge_index, node_attr
