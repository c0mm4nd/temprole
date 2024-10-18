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

for filename in ["palestine_israel", "palestine_israel_pruned"]:
    G: nx.MultiDiGraph = pickle.load(lzma.open(f"datasets/nx/{filename}.pkl.xz", "rb"))
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
        # (0, 1, {'value': 39000000, 'blockNum': 51813403, 'coin': 'TRX', 'id': '0'})
        if data["coin"] == "TRX":
            raw_data["src_list"] += [from_node]
            raw_data["dst_list"] += [to_node]
            raw_data["timestamp_list"] += [data["blockNum"]]
            raw_data["msg_list"] += [[data["value"], 0]]
            continue
        if data["coin"] == "USDT":
            raw_data["src_list"] += [from_node]
            raw_data["dst_list"] += [to_node]
            raw_data["timestamp_list"] += [data["blockNum"]]
            raw_data["msg_list"] += [[0, data["value"]]]
            continue
        raise ValueError("unknown coin type", data)

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
