import os
import time
import json
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
import pickle
import lzma
import sys
import networkx as nx
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader,
)
from torch.optim.lr_scheduler import ExponentialLR
import logging
import argparse

from CryptoFIRM.train import train
from CryptoFIRM.eval import evaluate
from CryptoFIRM.model import CryptoFIRM
from CryptoFIRM.utils import (
    describe_tensor,
    dismantling_nodes,
    extract_roles,
    get_default_device_str,
    set_seed,
    simple_temporal_network_dismantling_analysis,
    temporal_network_dismantling_analysis,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--dataset",
    type=str,
    default="palestine_israel_pruned",
    choices=[
        "russia_ukraine",
        "russia_ukraine_pruned",
        "palestine_israel",
        "palestine_israel_pruned",
    ],
)
argparser.add_argument(
    "--model",
    type=str,
    default="CryptoFIRM",
)
argparser.add_argument("--lr", type=float, default=0.0001)
argparser.add_argument("--max-epochs", type=int, default=50)
argparser.add_argument("--train-batch-size", type=int, default=128)
argparser.add_argument("--eval-batch-size", type=int, default=1024)
argparser.add_argument("--num-heads", type=int, default=2)
argparser.add_argument("--max-roles", type=int, default=8)
argparser.add_argument("--embedding-dim", type=int, default=100)
argparser.add_argument("--memory-dim", type=int, default=100)
argparser.add_argument("--time-dim", type=int, default=100)
argparser.add_argument("--loss-lambda", type=float, default=0.5)
argparser.add_argument("--device", type=str, default=get_default_device_str())
argparser.add_argument("--disable-temporal", action="store_true")
argparser.add_argument("--disable-rolx", action="store_true")
argparser.add_argument("--disable-train", action="store_true")
argparser.add_argument("--enable-scheduler", action="store_true")
argparser.add_argument("--config", type=str, default="")
argparser.add_argument("--log-level", type=str, default="INFO")
argparser.add_argument("--output", type=str, default="")


def save_best_model_result(
    logger, data, model, model_name, device, neighbor_loader, args
):
    # load the best model
    importance_embeddings, role_embeddings = evaluate(
        data=data,
        model=model,
        device=device,
        neighbor_loader=neighbor_loader,
        batch_size=args.eval_batch_size,
    )

    roles = extract_roles(role_embeddings)

    # reverse roles
    role_nodes = {}
    for node, role in roles.items():
        if role not in role_nodes:
            role_nodes[role] = []
        role_nodes[role].append(node)

    for role, nodes in role_nodes.items():
        logger.info(f"Role {role}: {len(nodes)} nodes")

    node_importance = importance_embeddings.norm(dim=1)
    logger.info(f"Node importance: {node_importance}")

    for role, nodes in role_nodes.items():
        logger.info(
            f"Role {role}: {len(nodes)} nodes, mean importance: {node_importance[nodes].mean()}, "
            + f"total importance: {node_importance[nodes].sum()}"
        )

    most_important_role = max(
        role_nodes.keys(), key=lambda x: node_importance[role_nodes[x]].mean()
    )
    logger.warning(f"Most important role: {most_important_role}")

    result = temporal_network_dismantling_analysis(data, role_nodes)
    logger.warning(result)

    if not args.disable_rolx and not args.disable_temporal:
        # add the role info into the graph and save it
        G: nx.MultiDiGraph = pickle.load(
            lzma.open(f"./datasets/nx/{args.dataset}.pkl.xz", "rb")
        )
        mapping = json.load(open(f"./datasets/mapping/{args.dataset}.json", "r"))
        rev_mapping = {v: k for k, v in mapping.items()}
        role_addresses = {role: [] for role in role_nodes}
        for node, role in roles.items():
            addr = rev_mapping[node]
            nx.set_node_attributes(G, {addr: {"role": role}})
            role_addresses[role].append(addr)
        nx.write_gexf(G, f"./{args.dataset}_with_roles_by_{model_name}.gexf")

        G_dismantled = dismantling_nodes(G, role_addresses[most_important_role])
        nx.write_gexf(G_dismantled, f"./{args.dataset}_dismantled_by_{model_name}.gexf")


if __name__ == "__main__":
    args = argparser.parse_args()
    if args.config:
        import yaml

        conf = yaml.safe_load(open(args.config, "r"), Loader=yaml.FullLoader)
        for key, value in conf.items():
            setattr(args, key, value)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    if not args.output:
        default_folder = "./outputs/"
        os.makedirs(default_folder, exist_ok=True)
        default_log_details = [
            args.dataset,
            args.model,
        ]
        if args.disable_temporal:
            default_log_details.append("noTemporal")
        if args.disable_rolx:
            default_log_details.append("noRolx")
        default_log_details.append(time.strftime("%Y%m%d-%H%M%S"))
        args.output = default_folder + "_".join(default_log_details) + ".log"

    file_handler = logging.FileHandler(args.output)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    stdout_handler.setFormatter(formatter)

    set_seed(2024)
    # fix as https://github.com/pytorch/pytorch/issues/21819
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        if args.device == "cuda":
            args.device = "cuda:0"
        torch.cuda.set_device(args.device)
    device = torch.device(args.device)

    # describe each arg
    logger.info("Arguments:")
    logger.info(f"dataset: {args.dataset}")
    logger.info(f"model: {args.model}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"max_epochs: {args.max_epochs}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"eval_batch_size: {args.eval_batch_size}")
    logger.info(f"num_heads: {args.num_heads}")
    logger.info(f"embedding_dim: {args.embedding_dim}")
    logger.info(f"memory_dim: {args.memory_dim}")
    logger.info(f"time_dim: {args.time_dim}")
    logger.info(f"loss_lambda: {args.loss_lambda}")
    logger.info(f"device: {args.device}")
    logger.info(f"disable_temporal: {args.disable_temporal}")
    logger.info(f"disable_rolx: {args.disable_rolx}")
    logger.info(f"disable_train: {args.disable_train}")
    logger.info(f"config: {args.config}")
    logger.info(f"output: {args.output}")

    data: TemporalData = pickle.load(
        lzma.open(f"./datasets/{args.dataset}_temporal.pkl.xz")
    )
    data.to(device)

    embedding_dim = args.embedding_dim
    memory_dim = args.memory_dim
    time_dim = args.time_dim

    data_loader = TemporalDataLoader(
        data,
        batch_size=args.train_batch_size,
        neg_sampling_ratio=1.0,
    )

    neighbor_loader = LastNeighborLoader(
        data.num_nodes, size=args.train_batch_size, device=device
    )

    msg_dim = data.msg.size(-1)

    if args.model == "CryptoFIRM":
        model = CryptoFIRM(
            num_nodes=data.num_nodes,
            memory_dim=memory_dim,
            embedding_dim=embedding_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            edge_index=data.edge_index.to(device),
            num_headers=args.num_heads,
            max_roles=args.max_roles,
            loss_lambda=args.loss_lambda,
            device=device,
            disable_temporal=args.disable_temporal,
            disable_role_extraction=args.disable_rolx,
        ).to(device)
        # model = torch.compile(model)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-9,
    )
    if args.enable_scheduler:
        scheduler = ExponentialLR(optimizer, gamma=0.9)

    model_name = (
        "CryptoFIRM"
        + f"_{args.dataset}"
        + ("_noTemporal" if args.disable_temporal else "")
        + ("_noRolx" if args.disable_rolx else "")
    )

    if args.disable_train:
        logger.debug(f"Loading model from ./models/{model_name}.pt")
        saved_model = torch.load(f"./models/{model_name}.pt")
        model.load_state_dict(saved_model["model"])
        optimizer.load_state_dict(saved_model["optimizer"])
    else:
        steps = []
        all_losses = []
        # lowest_loss = float("inf")
        best_avg_lcc_impact = float(0)
        for epoch in range(1, args.max_epochs + 1):
            loss = train(
                model=model,
                neighbor_loader=neighbor_loader,
                optimizer=optimizer,
                data_loader=data_loader,
                data=data,
                device=device,
            )

            # early stopping
            if epoch > 10 and loss > max(all_losses):
                logger.info(
                    f"Early stopping at epoch {epoch}: {loss:.4f} > {max(all_losses[-5:])}"
                )
                logger.info(all_losses)
                break

            logger.info(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

            # if loss < lowest_loss:
            #     lowest_loss = loss
            #     best_model = model.state_dict()

            all_losses.append(loss)

            importance_embeddings, role_embeddings = evaluate(
                data=data,
                model=model,
                device=device,
                neighbor_loader=neighbor_loader,
                batch_size=args.eval_batch_size,
            )
            if args.enable_scheduler:
                scheduler.step()

            roles = extract_roles(role_embeddings)

            # reverse roles
            role_nodes = {}
            for node, role in roles.items():
                if role not in role_nodes:
                    role_nodes[role] = []
                role_nodes[role].append(node)

            for role, nodes in role_nodes.items():
                logger.info(f"Role {role}: {len(nodes)} nodes")

            logger.info(f"importance_embeddings: \n{importance_embeddings}")
            node_importance = importance_embeddings.norm(dim=1)
            logger.info(f"Node importance: {node_importance}")

            for role, nodes in role_nodes.items():
                logger.info(
                    f"Role {role}: {len(nodes)} nodes, mean importance: {node_importance[nodes].mean()}, "
                    + f"total importance: {node_importance[nodes].sum()}"
                )

            most_important_role = max(
                role_nodes.keys(), key=lambda x: node_importance[role_nodes[x]].mean()
            )
            logger.warning(f"Most important role: {most_important_role}")
            most_important_role_nodes = role_nodes[most_important_role]

            result = simple_temporal_network_dismantling_analysis(
                data, role_nodes, most_important_role
            )
            result["epoch"] = epoch
            result["loss"] = loss
            logger.warning(result)
            steps.append(result)

            if result["avg_lcc_impact"] > best_avg_lcc_impact:
                logger.warning(f"New best model found at epoch {epoch}")
                best_avg_lcc_impact = result["avg_lcc_impact"]
                if epoch > 1:
                    save_best_model_result(
                        logger=logger,
                        data=data,
                        model=model,
                        model_name=model_name,
                        device=device,
                        neighbor_loader=neighbor_loader,
                        args=args,
                    )

        with open(f"./{model_name}_step_result.json", "w") as f:
            json.dump(steps, f)
