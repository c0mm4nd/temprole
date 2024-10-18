import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(*, data, model, neighbor_loader, device, batch_size):
    model.memory.eval()
    model.eval()

    importance_embedding_list = []
    role_embedding_list = []

    # Sequentially compute embeddings for all nodes.
    for i in tqdm(range(0, data.num_nodes, batch_size), desc="Evaluation"):
        batch_end = min(i + batch_size, data.num_nodes)
        batch_size = batch_end - i

        with torch.no_grad():
            n_id = torch.arange(i, batch_end, device=device)
            n_id, edge_index, e_id = neighbor_loader(n_id)
            model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, roles = model(
                edge_index.to(device),
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
                n_id.to(device),
            )

            batch_embeddings = z[model.assoc[torch.arange(i, batch_end, device=device)]]
            importance_embedding_list.append(batch_embeddings)

            batch_roles = roles[model.assoc[torch.arange(i, batch_end, device=device)]]
            role_embedding_list.append(batch_roles)

    importance_embeddings = torch.cat(importance_embedding_list, dim=0)
    role_embeddings = torch.cat(role_embedding_list, dim=0)

    return importance_embeddings, role_embeddings

