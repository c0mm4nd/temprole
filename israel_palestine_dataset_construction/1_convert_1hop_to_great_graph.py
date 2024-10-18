import os
from binascii import hexlify
import pickle
import networkx
import requests
import json
import lzma
from web3research.common import Address, ChainStyle

# Taddr = "TNT8WTuCoPwuYzScrHwbv5Wzw9XBwu9u3q"
# MAX_HOP = 2
# USDT_only = True


# Not usd due to not 2 or more hops
# IS_CONTRACT_RESULTS = {}
# def is_contract(Taddr):
#     if Taddr in IS_CONTRACT_RESULTS:
#         return IS_CONTRACT_RESULTS[Taddr]

#     url = "http://localhost:8090/wallet/getcontract"

#     payload = {"value": Taddr, "visible": True}
#     headers = {"accept": "application/json", "content-type": "application/json"}
#     response = requests.post(url, json=payload, headers=headers)

#     IS_CONTRACT_RESULTS[Taddr] = len(response.text) > 3
#     print(f"{Taddr} is contract?: {IS_CONTRACT_RESULTS[Taddr]}")

#     return IS_CONTRACT_RESULTS[Taddr]


G = networkx.MultiDiGraph()

for filename in os.listdir("rawdata/1hops"):
    print(f"Processing {filename}")
    Taddr = filename.split(".")[0]
    node_data = pickle.load(lzma.open(f"rawdata/1hops/{Taddr}.pkl.xz", "rb"))

    transfers = node_data["transfers"]
    tokenTransfers = node_data["usdtTransfers"]

    if len(transfers) == 0 and len(tokenTransfers) == 0:
        print(f"WARN: {Taddr} has no transfers")

    for tf in transfers:
        if tf["blockNum"] > 64000000:
            continue
        try:
            from_addr = Address(tf["ownerAddress"])
        except ValueError as e:
            print(f"Invalid address: {tf['ownerAddress']}")
            raise e

        G.add_edge(
            Address(tf["ownerAddress"]).string(chain_style=ChainStyle.TRON),
            Address(tf["toAddress"]).string(chain_style=ChainStyle.TRON),
            key=tf["transactionHash"],
            value=tf["amount"],
            blockNum=tf["blockNum"],
            coin="TRX",
        )

    for tf in tokenTransfers:
        if tf["blockNum"] > 64000000:
            continue

        G.add_edge(
            Address(tf["decoded"]["from"]).string(chain_style=ChainStyle.TRON),
            Address(tf["decoded"]["to"]).string(chain_style=ChainStyle.TRON),
            key=tf["transactionHash"] + "." + str(tf["logIndex"]),
            value=tf["decoded"]["value"],
            blockNum=tf["blockNum"],
            coin="USDT",
        )

    print("\tnodes: ", G.number_of_nodes(), "\tedges: ", G.number_of_edges())


print("nodes: ", G.number_of_nodes())
print("edges: ", G.number_of_edges())

# %%
import lzma

filename = "palestine_israel"

os.makedirs("./datasets/nx", exist_ok=True)
with lzma.open(f"./datasets/nx/{filename}.pkl.xz", "wb") as f:
    pickle.dump(G, f)

# with open(f"./graphs/gexf/{filename}.gexf", "wb") as f:
#     networkx.write_gexf(G, f)

# save nodes to a json
os.makedirs("./datasets/mapping", exist_ok=True)
with open(f"./datasets/mapping/{filename}.json", "w") as f:
    json.dump(list(G.nodes), f)


prune = True
if prune:
    # remove 
    # Sink Node: in degree = 1, out degree = 0
    # Source Node: in degree = 0, out degree = 1
    # isolated nodes: in degree = 0, out degree = 0

    sink_nodes = [node for node in G.nodes() if G.in_degree(node) == 1 and G.out_degree(node) == 0]
    source_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 1]
    isolated_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]

    G.remove_nodes_from(sink_nodes)
    G.remove_nodes_from(source_nodes)
    G.remove_nodes_from(isolated_nodes)

    print("nodes: ", G.number_of_nodes())
    print("edges: ", G.number_of_edges())

# %%
import lzma

filename = "palestine_israel_pruned"

os.makedirs("./datasets/nx", exist_ok=True)
with lzma.open(f"./datasets/nx/{filename}.pkl.xz", "wb") as f:
    pickle.dump(G, f)

# with open(f"./graphs/gexf/{filename}.gexf", "wb") as f:
#     networkx.write_gexf(G, f)

