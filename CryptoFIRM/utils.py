from se_community import compute_structural_entropy_mp
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.data import TemporalData
from collections import deque


def describe_tensor(name, tensor):
    print("Tensor name: ", name)
    print("Tensor details:")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Content: \n{tensor}")


def extract_roles(role_embedding):
    roles = torch.argmax(role_embedding, dim=1)
    return {i: value.item() for i, value in enumerate(roles)}


# TODO: future work
def extract_overlapping_roles(role_embedding, threshold=0.5):
    # indices = (role_embedding > 0.5).nonzero()
    # return indices
    result = []
    min_value = role_embedding[role_embedding > 0].min().item()
    if threshold > min_value:
        print(
            f"Threshold is larger than the minimum value in the role embedding. Setting threshold to {min_value}"
        )
        threshold = min_value
    for row in role_embedding >= threshold:
        result.append(row.nonzero(as_tuple=True)[0].tolist())

    return {i: value for i, value in enumerate(result)}


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_default_device_str():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # for macOS
        return "mps"
    else:
        return "cpu"


def calculate_lcc_size(G) -> int:
    """
    Calculate the size of the largest connected component in the graph.

    Args:
        G (networkx.Graph): The graph to analyze.

    Returns:
        int: Size of the largest connected component.
    """
    if G.is_directed():
        # largest_cc = max(nx.weakly_connected_components(G), key=len)
        largest_cc = max(
            nx.strongly_connected_components(G), key=len
        )
        # strongly connected components considering the direction of the edges
        # weakly connected components ignoring the direction of the edges
        # so, strongly connected components will ignore the impact of the peripheral nodes and pruned their outters
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc)


# for generic temporal data
def recover_graph_from_temporal_data(data: TemporalData) -> nx.MultiDiGraph:
    """
    Create a NetworkX graph from a temporal data.

    Args:
        edge_index (torch.Tensor): Edge index tensor.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        networkx.MultiDiGraph: The created graph.
    """
    G = nx.MultiDiGraph()
    t = data.t.cpu().numpy()
    src = data.src.cpu().numpy()
    dst = data.dst.cpu().numpy()
    msg = data.msg.cpu().numpy()

    for _t, _src, _dst, _msg in zip(t, src, dst, msg):
        for msg_id, val in enumerate(_msg):
            # if val != 0:
            attrs = {"time": _t, "message_type": msg_id, "message": val}
            G.add_edge(_src, _dst, **attrs)

    return G


# requires the G is recovered from the temporal data
# cannot use the original data, because the original data does not have the message, time, and message_type attributes
def dismantling_nodes(G, nodes):
    G_dismantled = G.copy()

    queue = deque(nodes)
    processed_nodes = set()

    for node in nodes:
        # add neighbors to the queue
        for neighbor in G.neighbors(node):
            if neighbor not in processed_nodes:
                queue.append(neighbor)
                processed_nodes.add(neighbor)

    peripheral_nodes = [
        node
        for node in G_dismantled.nodes()
        if G_dismantled.in_degree(node) == 0 or G_dismantled.out_degree(node) == 0
    ]

    # remove the edges of the role nodes
    edges_to_remove = list(G_dismantled.edges(nodes))
    # check whether the first in edge is removed, if removed, remove the all edges of the node
    G_dismantled.remove_edges_from(edges_to_remove)

    while queue:
        node = queue.popleft()
        if node in processed_nodes:
            continue

        if node in peripheral_nodes:
            # if the node is a peripheral node, means it may have external edges, so skip
            continue

        # get all out edges and in edges of the node
        out_edges = list(G_dismantled.out_edges(node, data=True, keys=True))
        in_edges = list(G_dismantled.in_edges(node, data=True, keys=True))

        # check whether the first in edge is removed, if removed, remove the all edges of the node
        for u, v, k, out_data in out_edges:
            message_type = out_data["message_type"]
            out_time = out_data["time"]
            out_val = out_data["message"]
            # check whether prev in_edges sum message is greater than the out_val
            is_valid = (
                sum(
                    [
                        in_data["message"]
                        for _, _, _, in_data in in_edges
                        if in_data["message_type"] == message_type
                        and in_data["time"] < out_time
                    ]
                )
                >= out_val
            )

            # if there is no enough balance, remove the out edge
            if not is_valid:
                # additional_edges_to_remove.append((u, v, k))
                G_dismantled.remove_edge(u, v, k)

                # add the destination node to the queue
                if v not in processed_nodes:
                    queue.append(v)
                    processed_nodes.add(v)

    return G_dismantled


def simple_temporal_network_dismantling_analysis(data, role_nodes, key_role):
    G = recover_graph_from_temporal_data(data)
    original_lcc_size = calculate_lcc_size(G)

    original_se = compute_structural_entropy_mp(G, {node: 0 for node in G.nodes()})

    nodes_to_remove = role_nodes[key_role]
    G_dismantled = dismantling_nodes(G, nodes_to_remove)
    lcc = calculate_lcc_size(G_dismantled)

    partition = {node: role for role, nodes in role_nodes.items() for node in nodes}
    se = compute_structural_entropy_mp(G, partition)

    result = {
        "node_count": len(nodes_to_remove),
        "role": key_role,
        "original_lcc": original_lcc_size,
        "dismantled_lcc": lcc,
        "lcc_impact": 1 - (lcc / original_lcc_size),
        "avg_lcc_impact": (1 - (lcc / original_lcc_size)) / len(role_nodes[key_role]),
        "original_se": original_se,
        "with_role_se": se,
        "dismantled_se": compute_structural_entropy_mp(
            G_dismantled, {node: 0 for node in G_dismantled.nodes()}
        ),
        "dismantled_with_role_se": compute_structural_entropy_mp(
            G_dismantled, {node: partition[node] for node in G_dismantled.nodes()}
        ),
    }

    return result


def temporal_network_dismantling_analysis(data, role_nodes, G=None):
    if G is None:
        G = recover_graph_from_temporal_data(data)
    else:
        G = G.copy()

    original_largest_cc_size = calculate_lcc_size(G)
    original_se = compute_structural_entropy_mp(G, {node: 0 for node in G.nodes()})

    dismantling_impact = {"by_roles": {}, "global_metrics": {}}
    partition = {node: role for role, nodes in role_nodes.items() for node in nodes}
    se = compute_structural_entropy_mp(G, partition)

    for role, nodes in tqdm(role_nodes.items(), desc="Analyzing roles"):
        G_dismantled = dismantling_nodes(G, nodes)

        largest_cc_size_after_removal = calculate_lcc_size(G_dismantled)
        lcc_impact = 1 - (largest_cc_size_after_removal / original_largest_cc_size)

        dismantling_impact["by_roles"][role] = {
            "node_count": len(nodes),
            "role": role,
            "original_lcc": original_largest_cc_size,
            "dismantled_lcc": largest_cc_size_after_removal,
            "lcc_impact": lcc_impact,
            "avg_lcc_impact": lcc_impact / len(role_nodes[role]),
            "original_se": original_se,
            "with_role_se": se,
            "dismantled_se": compute_structural_entropy_mp(
                G_dismantled, {node: 0 for node in G_dismantled.nodes()}
            ),
            "dismantled_with_role_se": compute_structural_entropy_mp(
                G_dismantled, {node: partition[node] for node in G_dismantled.nodes()}
            ),
        }

        print(
            f"Role {role} removed, node count {len(nodes)}, largest cc size: {largest_cc_size_after_removal}, impact: {lcc_impact}, avg impact: {lcc_impact / len(nodes)}"
        )

    dismantling_impact["global_metrics"]["structural_entropy"] = {
        "original": original_se,
        "with_role": se,
    }

    return dismantling_impact
