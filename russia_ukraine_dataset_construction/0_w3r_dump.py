# %%

from web3 import Web3
from web3research import Web3Research
from web3research.evm import SingleEventDecoder
from web3research.common.types import Address, ChainStyle

w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

w3r = Web3Research(api_token="default")
w3r_eth = w3r.eth(
    backend="http://localhost:8123",
    send_receive_timeout=6000,
)

addr = Address("0x165CD37b4C644C2921454429E7F9358d18A45e14")  # Ukraine Crypto Donation
# %%
import networkx as nx


# %%
from binascii import hexlify
import json
import os

if not os.path.exists("all_contract_addresses.json"):
    all_create_address_result = w3r_eth.query(
        "SELECT distinct resultCreateAddress from ethereum.traces  where resultCreateAddress != ''",
        column_formats={"resultCreateAddress": "bytes"},
    )
    LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES = [
        hexlify(result[0]).decode().lower()
        for result in all_create_address_result.result_rows
    ]
    with open("all_contract_addresses.json", "w") as f:
        json.dump(LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES, f)
else:
    with open("all_contract_addresses.json", "r") as f:
        LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES = json.load(f)

LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES = set(LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES)

print(f"all contract addresses: {len(LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES)}")


def is_contract_address(addr):
    _addr = addr.removeprefix("0x").lower()
    return _addr in LOWER_NOPREFIX_ALL_CONTRACT_ADDRESSES


# %%
import os
import json

os.makedirs("tx_caches", exist_ok=True)


# 加载或获取一跳交易数据
def get_1hop_transactions(addr):
    if os.path.exists("normal_txs.json"):
        with open("normal_txs.json", "r") as f:
            normal_txs = json.load(f)
    else:
        normal_txs = w3r_eth.transactions(
            f"""
            `from` = {addr} or `to` = {addr} and `blockNumber` < 19100000
        """,
            limit=None,
        )  # 获取交易
        normal_txs = list(normal_txs)
        with open("normal_txs.json", "w") as f:
            json.dump(normal_txs, f, indent=4)
    return normal_txs


# 获取并添加1-hop交易到图中
def add_1hop_edges(g, transactions):
    for tx in transactions:
        g.add_edge(
            tx["from"],
            tx["to"],
            key=tx["hash"],
            tx_hash=tx["hash"],
            block_number=tx["blockNumber"],
            block_timestamp=tx["blockTimestamp"],
            value=tx["value"],
        )


# 新增函数：获取2-hop交易数据
def ready_2hop_transactions(one_hop_g):
    all_addresses = set(G.nodes)

    for node in all_addresses:
        node_addr = Address("0x" + node)

        if not os.path.exists(f"tx_caches/{node}.json"):
            if is_contract_address(node):
                # USDT is tooooo large
                continue

            print(f"tx_caches/{node}.json not exists")
            transactions = w3r_eth.transactions(
                f"""
                `from` = {node_addr} or `to` = {node_addr} and `blockNumber` < 19100000
            """,
                limit=None,
            )
            transactions = list(transactions)
            with open(f"tx_caches/{node}.json", "w") as f:
                json.dump(transactions, f, indent=4)
        else:
            print(f"tx_caches/{node}.json exists")


from tqdm import tqdm


# 添加2-hop交易到图中
def add_2hop_edges(g):
    all_addresses = set(g.nodes)
    two_hop_txs = []

    tx_counts = []
    tx_counts_less_than_1e5 = []
    tx_count_more_than_1e5 = []

    bot_addresses = []
    contract_addresses = []
    for node in tqdm(all_addresses):
        if is_contract_address(node):
            # USDT is tooooo large
            print(f"contract address: {node}")
            contract_addresses.append(Web3.to_checksum_address("0x"+node))
            continue
        node_transactions = json.load(open(f"tx_caches/{node}.json", "r"))

        tx_count = len(node_transactions)
        tx_counts.append(tx_count)

        if len(node_transactions) >= 100_000:
            tx_count_more_than_1e5.append(tx_count)
            print(f"bot address: {node}")
            bot_addresses.append(Web3.to_checksum_address("0x"+node))
            continue
        else:
            tx_counts_less_than_1e5.append(tx_count)

        for tx in node_transactions:
            if tx["to"] is None:
                assert tx["contractAddress"] is not None
                to = tx["contractAddress"]
                continue # skip contract creation tx, because it doesn't transfer value
            else:
                to = tx["to"]
            # if not g.has_edge(tx["from"], tx["to"], key=tx["hash"]):  # 防止重复添加
            g.add_edge(
                tx["from"],
                to,
                key=tx["hash"],
                # tx_hash=tx["hash"],
                block_number=tx["blockNumber"],
                block_timestamp=tx["blockTimestamp"],
                value=tx["value"],
            )
    # # plot a tx_count histogram
    # import matplotlib.pyplot as plt

    # plt.hist(tx_counts, bins=100, log=True)
    # plt.savefig("tx_counts.png")
    # plt.clf()

    # plt.hist(tx_count_more_than_2e4, bins=100, log=True)
    # plt.savefig("tx_counts_more_than_2e4.png")
    # plt.clf()

    # plt.hist(tx_counts_less_than_2e4, bins=100, log=True)
    # plt.savefig("tx_counts_less_than_2e4.png")
    # plt.clf()

    # with open("bot_addresses.json", "w") as f:
    #     json.dump(bot_addresses, f)
    
    # with open("contract_addresses.json", "w") as f:
    #     json.dump(contract_addresses, f)

def check_or_remove_neighbors(G, G_no_multi_no_di, node, N):
    
    single_connected_nodes = [n for n in G_no_multi_no_di.neighbors(node) if G_no_multi_no_di.degree(n) == 1]
    # if too many single connected nodes, remove them
    if len(single_connected_nodes) > N:
        G.remove_nodes_from(single_connected_nodes)

def clean_bot_network(G):
    G_no_multi_no_di = nx.Graph(G)
    sorted_nodes = sorted(G_no_multi_no_di.nodes(), key=lambda x: G_no_multi_no_di.degree(x), reverse=True)

    N = 18_000

    for node in sorted_nodes:
        if G.has_node(node):
            check_or_remove_neighbors(G, G_no_multi_no_di, node, N)

if __name__ == "__main__":
    import pickle
    import lzma

    G = nx.MultiDiGraph()

    normal_txs = get_1hop_transactions(addr)
    add_1hop_edges(G, normal_txs)
    ready_2hop_transactions(G)

    print("ready")
    print(G.number_of_nodes(), G.number_of_edges())
    add_2hop_edges(G)

    print(G.number_of_nodes(), G.number_of_edges())

    clean_bot_network(G)

    with lzma.open("russia_ukraine.nx.pkl.xz", "wb") as f:
        pickle.dump(G, f)

    prune = True
    if prune:
        sink_nodes = [
            node
            for node in G.nodes()
            if G.in_degree(node) == 1 and G.out_degree(node) == 0
        ]
        source_nodes = [
            node
            for node in G.nodes()
            if G.in_degree(node) == 0 and G.out_degree(node) == 1
        ]
        isolated_nodes = [
            node
            for node in G.nodes()
            if G.in_degree(node) == 0 and G.out_degree(node) == 0
        ]

        G.remove_nodes_from(sink_nodes)
        G.remove_nodes_from(source_nodes)
        G.remove_nodes_from(isolated_nodes)

        print(
            "pruned",
            G.number_of_nodes(),
            G.number_of_edges(),
            len(sink_nodes),
            len(source_nodes),
            len(isolated_nodes),
        )
        with lzma.open("russia_ukraine_pruned.nx.pkl.xz", "wb") as f:
            pickle.dump(G, f)

# %%
# internal_txs = w3r.eth.traces(f"""
#     actionCallFrom = {addr} or actionCallTo = {addr}
# """, limit=None) # internal transactions

# list(internal_txs)

# %%
# USDT_txs = w3r.eth.events(
#     f"""
#     address = 0xdac17f958d2ee523a2206206994597c13d831ec7 and
#     topics[0] = 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef and
#     (topics[1] = {addr} or topics[2] = {addr})
# """,
#     limit=1,
# )  # USDT
