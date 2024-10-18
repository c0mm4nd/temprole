from web3research import Web3Research
from web3research.common import Address, Hash
from web3research.evm import ContractDecoder, ERC20_ABI
from web3 import Web3
import os
import dotenv

dotenv.load_dotenv()

TRANSFER_TOPIC = Hash("ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
USDT_ADDR = Address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t") # dont change me

w3r = Web3Research(api_token=os.getenv("W3R_API"))
w3r_tron = w3r.tron(backend=os.getenv("W3R_BACKEND"))

ERC20Decoder = ContractDecoder(Web3(), ERC20_ABI)

# up to 64_000_000 -> 2024-08-03 00:42:18 (UTC)
def dump_1hops_from_w3r_backend(Taddr: str):
    try:
        addr = Address(Taddr)
    except ValueError as e:
        print(f"Invalid address: {Taddr}")
        raise e
    addr_hash = Hash("0" * 24 + addr.addr_hex)

    # get all trx transfers from/to Taddr
    transfers = w3r_tron.transfer_contracts(f"""
        (toAddress = {addr} or ownerAddress = {addr}) and blockNum <= 64000000
    """, limit=None)
    transfers = list(transfers)
    # print(list(transfers))

    # get all usdt transfers from/to Taddr
    usdtTransferLogs = w3r_tron.events(f"""
        address = {USDT_ADDR} and topic0 = {TRANSFER_TOPIC} and (topic1 = {addr_hash} or topic2 = {addr_hash}) and blockNum <= 64000000
    """, limit=None)
    usdtTransfers = []
    for log in usdtTransferLogs:
        decoded_log = ERC20Decoder.decode_event_log("Transfer", log)
        log["decoded"] = decoded_log
        usdtTransfers.append(log)
    # print(usdtTransfers)
    return transfers, usdtTransfers

if __name__ == "__main__":
    import json
    import lzma
    import pickle
    from tqdm import tqdm

    addrs = json.load(open("./labels/israel_labelled.json"))
    os.makedirs("./dataset/1hops", exist_ok=True)
    for addr in tqdm(addrs):
        if os.path.exists(f"./dataset/1hops/{addr}.pkl.xz"):
            print(f"Skipping {addr}")
            continue
        transfers, usdtTransfers = dump_1hops_from_w3r_backend(addr)
        if len(transfers) == 0 and len(usdtTransfers) == 0:
            print(f"WARN: {addr} has no transfers")
            continue
        data = {
            "addr": addr,
            "transfers": transfers,
            "usdtTransfers": usdtTransfers
        }
        with lzma.open(f"./dataset/1hops/{addr}.pkl.xz", "w") as f:
            pickle.dump(data, f)
        # save to dataset hop1_compresses
