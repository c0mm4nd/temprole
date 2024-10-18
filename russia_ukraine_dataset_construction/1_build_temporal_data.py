# %%
# convert Networkx to TemporalData
from torch_geometric.data import TemporalData
import torch.nn.functional as F
import networkx as nx
import pickle
import lzma
import torch
import pickle
import lzma
import os

for filename in ["russia_ukraine", "russia_ukraine_pruned"]:
    G: nx.MultiDiGraph = pickle.load(lzma.open(f"datasets/nx/{filename}.pkl.xz", "rb"))
    print(f"Loaded {filename}.pkl.xz")
    raw_data = {
        "src_list": [],
        "dst_list": [],
        "timestamp_list": [],
        "msg_list": [],
    }

    relabel_mapping = {}
    for index, addr in enumerate(G.nodes()):
        relabel_mapping[addr] = index

    H = nx.relabel_nodes(G, relabel_mapping)

    for from_node, to_node, data in H.edges(data=True):
        # print(data)
        # {'block_number': 14082491, 'block_timestamp': 1643216423, 'value': 0}
        raw_data["src_list"] += [from_node]
        raw_data["dst_list"] += [to_node]
        raw_data["timestamp_list"] += [data["block_number"]]
        raw_data["msg_list"] += [[data["value"]]]
        continue

    # %%
    # normalize timestamp
    t = torch.tensor(raw_data["timestamp_list"], dtype=torch.long)
    t = t - t.min()

    # normalize msg
    msg = torch.tensor(raw_data["msg_list"], dtype=torch.float)
    msg = F.normalize(msg, p=2, dim=1, eps=1e-12)

    temporal_data = TemporalData(
        src=torch.tensor(raw_data["src_list"], dtype=torch.long),
        dst=torch.tensor(raw_data["dst_list"], dtype=torch.long),
        t=t,
        msg=msg,
    )
    # %%

    os.makedirs("./datasets", exist_ok=True)
    with lzma.open(f"./datasets/{filename}_temporal.pkl.xz", "wb") as f:
        pickle.dump(temporal_data, f)

    # %%
    import json
    with open(f"./datasets/mapping/{filename}.json", "w") as f:
        json.dump(relabel_mapping, f)
