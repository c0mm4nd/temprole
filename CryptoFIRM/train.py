import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from CryptoFIRM.utils import describe_tensor


def train(*, model, data, data_loader, neighbor_loader, optimizer, device):
    model.train()

    model.memory.reset_state()
    neighbor_loader.reset_state()  # Start with an empty graph.

    all_losses = []
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, roles = model(
            edge_index.to(device),
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
            n_id.to(device),
        )

        loss = model.loss(
            z,
            roles,
            n_id,
            data.edge_index,
            data.t,
            data.msg,
        )

        # Update memory and neighbor loader with ground-truth state.
        model.memory.update_state(
            batch.src.to(device),
            batch.dst.to(device),
            batch.t.to(device),
            batch.msg.to(device),
        )
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.memory.detach()
        # total_loss += float(loss) * batch.num_events
        all_losses.append(float(loss))

    # return total_loss / data.num_event
    return float(np.mean(all_losses))
