import torch
import torch_scatter
from torch.nn import Linear, Parameter, LayerNorm
from torch_geometric.nn import TransformerConv, TGNMemory, MLP
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)
import torch.nn.functional as F
import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)

# for debugging
from CryptoFIRM.utils import describe_tensor


class RoleExtractor(torch.nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        num_roles=None,
        max_roles=16,
        min_roles=4,
        device="cuda",
    ):
        super(RoleExtractor, self).__init__()
        self.num_nodes = num_nodes

        self.graph_structure_features = 2  # in_degrees, out_degrees, degrees
        self.resizer = MLP([embedding_dim, self.graph_structure_features])
        self.importance_features = self.graph_structure_features  # embedding_dim

        self.num_features = self.graph_structure_features + self.importance_features

        self.max_roles = max_roles
        self.min_roles = min_roles
        # self.feature_matrix = torch.Tensor(num_nodes, self.num_features).to(device) # is not a parameter
        self.device = device
        self.behavior_matrix = None
        if num_roles is not None:
            self.num_roles = num_roles
            self.behavior_matrix = Parameter(torch.Tensor(num_roles, self.num_features))
            torch.nn.init.xavier_uniform_(self.behavior_matrix)
        else:
            self.num_roles = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.behavior_matrix:
            torch.nn.init.xavier_uniform_(self.behavior_matrix)

    def determine_num_roles(self, edge_index, simlified=False):
        if simlified:
            suggested_roles = int(np.log(self.num_nodes))
            self.num_roles = max(self.min_roles, min(suggested_roles, self.max_roles))
        else:
            # Compute average degree
            degree = torch.bincount(edge_index[0], minlength=self.num_nodes)
            avg_degree = torch.mean(degree.float())

            # Compute an approximation of the clustering coefficient
            # This is a simplified version and may not be accurate for all graphs
            A = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.size(1)).to(self.device),
                (self.num_nodes, self.num_nodes),
            )
            A_squared = torch.sparse.mm(A, A)
            triangles = torch.sum(A * A_squared) / 2
            possible_triangles = torch.sum(degree * (degree - 1)) / 2
            clustering_coeff = (
                triangles / possible_triangles if possible_triangles > 0 else 0
            )

            # Use these metrics to determine the number of roles
            # This is a heuristic and may need to be adjusted based on your specific use case
            suggested_roles = int(
                torch.sqrt(avg_degree)
                * (1 + clustering_coeff).clone().detach()
                * torch.log(torch.tensor(self.num_nodes))
            )
            self.num_roles = max(self.min_roles, min(suggested_roles, self.max_roles))
            # print(f"Network stats - Avg degree: {avg_degree:.2f}, Clustering coefficient: {clustering_coeff:.2f}")

        logger.warning(f"Determined number of roles: {self.num_roles}")

        self.behavior_matrix = Parameter(
            torch.Tensor(self.num_roles, self.num_features).to(self.device)
        )

        torch.nn.init.xavier_uniform_(self.behavior_matrix)

    def approximate_pagerank(self, edge_index, num_iterations=10, alpha=0.15):
        num_nodes = self.num_nodes

        # 计算出度
        out_degrees = torch.bincount(edge_index[0], minlength=num_nodes)

        pagerank = torch.ones(num_nodes, device=edge_index.device) / num_nodes

        for _ in range(num_iterations):
            contrib = (
                (1 - alpha)
                * pagerank[edge_index[0]]
                / out_degrees[edge_index[0]].clamp(min=1)
            )
            new_pagerank = torch.zeros_like(pagerank).scatter_add_(
                0, edge_index[1], contrib
            )
            new_pagerank += alpha / num_nodes
            pagerank = new_pagerank

        return pagerank

    def approximate_clustering(self, edge_index, num_samples=1000):
        sampled_nodes = torch.randint(
            0, self.num_nodes, (num_samples,), device=edge_index.device
        )
        clustering = torch.zeros(self.num_nodes, device=edge_index.device)

        # 将边集存入集合，方便查询
        edges_set = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))

        for node in sampled_nodes:
            neighbors = edge_index[1][edge_index[0] == node]
            if len(neighbors) < 2:
                continue
            neighbor_pairs = torch.combinations(neighbors, r=2)

            # 检查这些邻居对是否也是边
            connected_pairs = [
                (pair[0].item(), pair[1].item()) in edges_set for pair in neighbor_pairs
            ]
            clustering[node] = torch.tensor(
                connected_pairs, dtype=torch.float32, device=edge_index.device
            ).mean()

        return clustering

    def forward(self, edge_index, importance_emb, batch_nodes):
        if self.num_nodes is None:
            self.determine_num_roles(edge_index)

        edge_index = edge_index.to(self.device)

        # in_degrees = torch.bincount(edge_index[1], minlength=self.num_nodes).float()
        out_degrees = torch.bincount(edge_index[0], minlength=self.num_nodes).float()
        # degrees = in_degrees + out_degrees

        approx_pagerank = self.approximate_pagerank(edge_index)
        batch_importance_emb = self.resizer(importance_emb)

        # learnt node feature embedding
        # batch_features = self.feature_matrix[batch_nodes]
        # set metrics to the feature matrix
        graph_structure_features = torch.cat(
            [
                # in_degrees[batch_nodes].unsqueeze(1),
                out_degrees[batch_nodes].unsqueeze(1),
                approx_pagerank[batch_nodes].unsqueeze(1),
                batch_importance_emb,
                # page_rank,
                # local_entropy,
                # approx_pagerank[batch_nodes],
                # approx_clustering[batch_nodes],
            ],
            dim=1,
        )
        # update the feature matrix
        # self.feature_matrix[batch_nodes] = graph_structure_features

        # X = W * H
        H = self.behavior_matrix
        X = graph_structure_features #self.feature_matrix

        W = torch.linalg.lstsq(H.T, X.T).solution.T
        roles = F.normalize(W, p=2, dim=1)
        
        return roles


class FallbackRoleExtractor(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, num_roles=None, max_roles=32, min_roles=4):
        super(FallbackRoleExtractor, self).__init__()
        self.num_nodes = num_nodes
        self.max_roles = max_roles
        self.min_roles = min_roles
        if num_roles is None:
            # suggested_roles = int(np.log(self.num_nodes))
            suggested_roles = int(
                torch.log(torch.tensor(self.num_nodes, dtype=torch.float32))
            )
            self.num_roles = max(self.min_roles, min(suggested_roles, self.max_roles))
        else:
            self.num_roles = num_roles
        self.input_dim = input_dim
        self.linear = Linear(self.input_dim, self.num_roles)

    def forward(self, x):
        return F.normalize(self.linear(x), p=2, dim=1)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_headers):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // num_headers,
            heads=num_headers,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class CryptoFIRM(torch.nn.Module):
    def __init__(
        self,
        *,
        num_nodes,
        memory_dim,
        embedding_dim,
        msg_dim,
        time_dim,
        edge_index,
        num_headers,
        max_roles=16,
        loss_lambda=0.5,  # the weight of the importance loss
        device="cuda",
        disable_temporal=False,
        disable_role_extraction=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.loss_lambda = loss_lambda

        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)

        self.disable_temporal = disable_temporal
        self.disable_role_extraction = disable_role_extraction

        if disable_temporal:
            # initialize a x feature matrix
            self.x = Parameter(torch.zeros(num_nodes, embedding_dim))
            self.gae = Linear(embedding_dim, embedding_dim)
        else:
            self.gae = GraphAttentionEmbedding(
                memory_dim,
                embedding_dim,
                msg_dim,
                self.memory.time_enc,
                num_headers,
            )

        if disable_role_extraction:
            self.role_extractor = FallbackRoleExtractor(num_nodes, memory_dim)
        else:
            self.role_extractor = RoleExtractor(
                num_nodes,
                embedding_dim=embedding_dim,
                max_roles=max_roles,
                device=device,
            )
            self.role_extractor.determine_num_roles(edge_index)

        # Helper vector to map global node indices to local ones.
        # self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.register_buffer(
            "assoc", torch.empty(num_nodes, dtype=torch.long, device=device)
        )
        self.dummy_node_importances = None
        # self.register_buffer(
        #     "dummy_node_importances", torch.empty(num_nodes, dtype=torch.float, device=device)
        # )

    def forward(self, edge_index, t, msg, batch_nodes):
        if self.disable_temporal:
            importance_emb = self.gae(self.x)
            importance_emb = importance_emb[batch_nodes]
        else:
            z, last_update = self.memory(batch_nodes)
            assert z.isnan().sum() == 0, f"z has NaN values: {z}"

            importance_emb = self.gae(z, last_update, edge_index, t, msg)
        assert (
            importance_emb.isnan().sum() == 0
        ), f"importance_emb has NaN values: {importance_emb}"

        if self.disable_role_extraction:
            role_emb = self.role_extractor(z)
        else:
            role_emb = self.role_extractor(
                edge_index, importance_emb, batch_nodes
            )  # simple role embedding
        assert role_emb.isnan().sum() == 0, f"role_emb has NaN values: {role_emb}"

        # print("importance_emb", importance_emb, importance_emb.size())
        return importance_emb, role_emb

    @torch.no_grad()
    def dummy_node_importance(self, full_edge_index, full_t, full_msg):
        # 计算节点重要性
        # print("node_ids", node_ids, node_ids.size())
        # print("edge_index", edge_index, edge_index.size())
        # print("t", t, t.size())
        # print("msg", msg, msg.size())
        if self.dummy_node_importances is not None:
            return self.dummy_node_importances

        structural_rank = torch.ones(self.num_nodes, device=full_edge_index.device) / self.num_nodes
        damping_factor = 0.85
        # node is not match the index, but match the node_ids
        out_degrees = torch_scatter.scatter_add(
            torch.ones(full_edge_index.size(1), device=full_edge_index.device),
            full_edge_index[0],
            dim=0,
            dim_size=self.num_nodes,
        )

        for _ in range(10):
            # Step 3.1: Compute contributions from neighbors
            neighbor_contributions = structural_rank[full_edge_index[0]] / out_degrees[full_edge_index[0]]
            
            # Step 3.2: Aggregate contributions to each node (scatter_add based on target nodes)
            aggregated_contributions = torch_scatter.scatter_add(
                neighbor_contributions,
                full_edge_index[1],
                dim=0,
                dim_size=self.num_nodes
            )
            
            # Step 3.3: Apply the PageRank formula
            structural_rank = (1 - damping_factor) / self.num_nodes + damping_factor * aggregated_contributions
        

        if full_t.numel() == 0:
            logger.warning("t is empty")
            max_t = torch.tensor(0.0, device=full_t.device)
        else:
            max_t = full_t.max()

        node_times = torch.full(
            (self.num_nodes,), max_t, device=full_t.device, dtype=full_t.dtype
        )
        # node_times = torch.scatter_reduce(
        #     node_times, 0, edge_index[0], t, reduce='amin', include_self=False
        # )
        node_times = torch_scatter.scatter_min(
            full_t, full_edge_index[0], dim=0, dim_size=self.num_nodes
        )[0]
        # for i in range(edge_index[0].size(0)):
        #     node_times[edge_index[0][i]] = min(node_times[edge_index[0][i]], t[i])
        # print("node_times", node_times, node_times.size())

        # calculate the sum of messages for each node
        if full_msg.numel() == 0:
            logger.warning("msg is empty")
            msg_sum = torch.zeros(self.num_nodes, device=full_msg.device)
        else:
            # use scatter_add to sum the messages for each node
            msg_sum = torch.zeros(self.num_nodes, full_msg.size(-1), device=full_msg.device)
            msg_sum.scatter_add_(
                0, full_edge_index[0].unsqueeze(-1).expand(-1, full_msg.size(-1)), full_msg
            )
            msg_sum = msg_sum

        # print("msg_sum", msg_sum, msg_sum.size())
        node_time_importance = 1 / (node_times + 1)

        node_structural_importance = structural_rank#.norm(dim=1)
        node_structural_importance = node_structural_importance / node_structural_importance.max()

        node_msg_importance = msg_sum.norm(dim=1) if msg_sum.dim() > 1 else msg_sum

        # print("node_time_importance", node_time_importance, node_time_importance.size())
        # print("node_structural_importance", node_structural_importance, node_structural_importance.size())
        # print("node_msg_importance", node_msg_importance, node_msg_importance.size())

        # importance = norm(time * msg)
        importance = node_time_importance * node_msg_importance
        importance = importance + node_structural_importance
        importance = importance / importance.max()
        self.dummy_node_importances = importance

        # print most important nodes
        _, topk = importance.topk(10)
        print(f"Top 10 most important nodes: {topk}")

        return self.dummy_node_importances

    def contrastive_importance_loss(self, node_ids, z, importance):
        importance_true = importance[node_ids]
        importance_true = importance_true / importance_true.max()

        importance_pred = z.norm(dim=1)
        importance_pred = importance_pred / importance_pred.max()

        return F.mse_loss(importance_pred, importance_true)
        # return F.huber_loss(importance_pred, importance_true)

    def contrastive_role_loss(self, roles, similarity_threshold=0.9):
        # Contrastive learning for roles
        role_sim = torch.mm(roles, roles.t())  # Similarity between roles
        # role_sim /= temperature

        logger.debug(f"role_sim {role_sim} {role_sim.size()}")
        # split the similarity matrix into positive and negative samples by thresholding
        pos_mask = role_sim > similarity_threshold
        neg_mask = role_sim <= similarity_threshold

        # avoid self-similarity
        neg_mask.fill_diagonal_(0)

        max_sim = role_sim.max()
        exp_pos_sim = torch.exp(role_sim - max_sim) * pos_mask.float()
        pos_sim_sum = exp_pos_sim.sum(dim=1)

        exp_neg_sim = torch.exp(role_sim - max_sim) * neg_mask.float()
        neg_sim_sum = exp_neg_sim.sum(dim=1)

        eps = 1e-9
        role_loss = -torch.log(
            (pos_sim_sum + eps) / (pos_sim_sum + neg_sim_sum + eps)
        ).mean()

        return role_loss

    def loss(
        self, z, roles, node_ids, full_edge_index, full_t, full_msg, 
    ):
        role_loss = self.contrastive_role_loss(roles) * (1 - self.loss_lambda)
        importance = self.dummy_node_importance(full_edge_index, full_t, full_msg)
        importance_loss = (
            self.contrastive_importance_loss(node_ids, z, importance) * self.loss_lambda
        )

        # # punish if the most important node is too many roles
        # node_roles = torch.argmax(roles, dim=1)
        # node_importance = importance[node_ids]
        # unique_roles = torch.unique(node_roles)

        # # calculate the average importance for each role
        # role_importance_dict = {}
        # for role in unique_roles:
        #     role_indices = (node_roles == role).nonzero(as_tuple=True)[0]
        #     average_importance = node_importance[role_indices].mean().item()
        #     role_importance_dict[role.item()] = average_importance

        # # find the most important role
        # most_important_role = max(role_importance_dict, key=role_importance_dict.get)

        # # get the count of the most important role
        # most_important_node_count = (node_roles == most_important_role).sum().item()
        # most_important_node_proportion = most_important_node_count / len(node_ids)

        assert not torch.isnan(importance_loss), f"importance_loss is NaN"
        assert not torch.isnan(role_loss), f"role_loss is NaN"

        # warn if the loss values are too different
        # if role_loss.is_nonzero() and importance_loss.is_nonzero() and importance_loss > 100 * role_loss or role_loss > 100 * importance_loss:
        #         logger.warning(
        #             f"importance_loss {importance_loss}, role_loss {role_loss}, if this warning is frequent, consider adjusting the loss_lambda parameter"
        #         )

        # TODO: 32 = max role count, set to a variable
        total_loss = importance_loss + role_loss
        # * (most_important_node_proportion * 32)
        # print("total_loss", total_loss)
        return total_loss
