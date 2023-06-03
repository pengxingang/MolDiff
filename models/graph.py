import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph
from models.common import GaussianSmearing, MLP, NONLINEARITIES


class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim+node_dim+1, hidden_dim, hidden_dim) # add 1 for time

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, node_time):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(h_edge * h_node[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col], node_time[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class NodeEncoder(Module):
    
    def __init__(self, node_dim=256, edge_dim=64, key_dim=128, num_heads=4, 
                    num_blocks=6, k=48, cutoff=10.0, use_atten=True, use_gate=True,
                    dist_version='new'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
            self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim-self.additional_edge_feat)
            self.edge_emb = Linear(edge_dim, edge_dim)
        else:
            raise NotImplementedError('dist_version notimplemented')
        self.node_blocks = ModuleList()
        for _ in range(num_blocks):
            block = NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                use_atten=use_atten,
                use_gate=use_gate,
            )
            self.node_blocks.append(block)

    @property
    def out_channels(self):
        return self.node_dim

    def forward(self, h, pos, edge_index, is_mol):
        #NOTE in the encoder, the edge dose not change since the position of mol and protein is fixed
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_attr = self._add_edge_features(pos, edge_index, is_mol)
        for interaction in self.node_blocks:
            h = h + interaction(h, edge_index, edge_attr)
        return h

    @property
    def additional_edge_feat(self,):
        return 2

    def _add_edge_features(self, pos, edge_index, is_mol):
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        # 2-vector represent the two node types (atoms of protein or mol)
        edge_src_feat = is_mol[edge_index[0]].float().view(-1, 1)
        edge_dst_feat = is_mol[edge_index[1]].float().view(-1, 1)
        edge_attr = torch.cat([edge_attr, edge_src_feat, edge_dst_feat], dim=1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim+node_dim+1, out_dim, 32)  # +1 for time

    def forward(self, bond_feat_input, node_feat_input, time):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, time], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class QKVLin(Module):
    def __init__(self, h_dim, key_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_lin = Linear(h_dim, key_dim)
        self.k_lin = Linear(h_dim, key_dim)
        self.v_lin = Linear(h_dim, h_dim)

    def forward(self, inputs):
        n = inputs.size(0)
        return [
            self.q_lin(inputs).view(n, self.num_heads, -1),
            self.k_lin(inputs).view(n, self.num_heads, -1),
            self.v_lin(inputs).view(n, self.num_heads, -1),
        ]


class BondBlock(Module):
    def __init__(self, bond_dim, node_dim, use_gate=True, use_atten=False, num_heads=2, key_dim=128):
        super().__init__()
        self.use_atten = use_atten
        self.use_gate = use_gate
        inter_dim = bond_dim * 2

        self.bond_ffn_left = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        if self.use_atten:
            # key_dim = bond_dim // 2
            assert bond_dim % num_heads == 0
            assert key_dim % num_heads == 0
            # linear transformation for attention 
            self.qkv_left = QKVLin(bond_dim, key_dim, num_heads)
            self.qkv_right = QKVLin(bond_dim, key_dim, num_heads)
            # self.q_lin = Linear(bond_dim, key_dim)
            # self.k_lin = Linear(bond_dim, key_dim)
            # self.v_lin = Linear(bond_dim, bond_dim)
            self.layer_norm_atten1 = nn.LayerNorm(bond_dim)
            self.layer_norm_atten2 = nn.LayerNorm(bond_dim)
        
        self.node_ffn_left = Linear(node_dim, bond_dim)
        self.node_ffn_right = Linear(node_dim, bond_dim)

        self.self_ffn = Linear(bond_dim, bond_dim)
        self.layer_norm = nn.LayerNorm(bond_dim)
        self.out_transform = Linear(bond_dim, bond_dim)
        self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node, atten_index=None):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        pos_node: (n, 3)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node])
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node])
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond = self.layer_norm(h_bond)

        if self.use_atten:
            index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right = atten_index

            # left node
            h_queries, h_keys, h_values = self.qkv_left(h_bond)
            queries_i = h_queries[index_query_bond_left]
            keys_j = h_keys[index_key_bond_left]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_left, dim=0)
            values_j = h_values[index_key_bond_left]
            num_attns = len(index_key_bond_left)
            h_bond = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_left, dim=0, dim_size=h_bond.size(0))
            h_bond = self.layer_norm_atten1(h_bond)

            # right node
            h_queries, h_keys, h_values = self.qkv_right(h_bond)
            queries_i = h_queries[index_query_bond_right]
            keys_j = h_keys[index_key_bond_right]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_right, dim=0)
            values_j = h_values[index_key_bond_right]
            num_attns = len(index_key_bond_right)
            h_bond = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_right, dim=0, dim_size=h_bond.size(0))
            h_bond = self.layer_norm_atten2(h_bond)

        h_bond = self.out_transform(self.act(h_bond))
        return h_bond




class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node, bond_time):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_time)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_time)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond = self.layer_norm(h_bond)

        h_bond = self.out_transform(self.act(h_bond))
        return h_bond


class NodeEdgeNet(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))

    def forward(self, h_node, pos_node, h_edge, edge_index, node_time, edge_time):
        for i in range(self.num_blocks):
            # edge fetures before each block
            if self.update_pos or (i==0):
                h_edge_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if self.update_edge:
                h_edge = torch.cat([h_edge, h_edge_dist], dim=-1)
            else:
                h_edge = h_edge_dist
            h_edge = self.edge_embs[i](h_edge)
                
            # node and edge feature updates
            h_node_with_edge = self.node_blocks_with_edge[i](h_node, edge_index, h_edge, node_time)
            if self.update_edge:
                h_edge = h_edge + self.edge_blocks[i](h_edge, edge_index, h_node, edge_time)
            h_node = h_node + h_node_with_edge
            # pos updates
            if self.update_pos:
                pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, edge_time)
        return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, edge_time):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        right_feat = self.right_lin_edge(h_node[edge_index_right])
        weight_edge = self.edge_lin(h_edge, left_feat * right_feat, edge_time)
        
        # relative_vec = pos_node[edge_index_left] - pos_node[edge_index_right]
        # distance = torch.norm(relative_vec, dim=-1, keepdim=True)
        force_edge = weight_edge * relative_vec / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])

        return delta_pos

class NodeBondNet(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, key_dim, num_heads, num_blocks, k, cutoff, use_atten, use_gate):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.bond_dim = bond_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
        self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        # node network
        self.lin_node = Linear(node_dim, node_dim)
        self.node_blocks_with_edge = ModuleList()
        self.node_blocks_with_bond = ModuleList()
        self.bond_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,  # never use atten for edges message becused too many edges
            ))
            self.node_blocks_with_bond.append(NodeBlock(
                node_dim=node_dim, edge_dim=bond_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,
            ))
            if bond_dim > 0:
                self.bond_blocks.append(BondBlock(
                    bond_dim=bond_dim, node_dim=node_dim, use_gate=use_gate,
                    use_atten=use_atten, key_dim=key_dim, num_heads=num_heads,
                ))

    def forward(self, h_node, pos_node, h_bond, bond_index, batch, is_mol, is_frag, return_edge=False):

        edge_attr, edge_index = self._build_edges(pos_node, batch, is_mol, is_frag)
        for i in range(self.num_blocks):
            # node updates with edges
            h_node_with_edge = self.node_blocks_with_edge[i](h_node, edge_index, edge_attr)
            if self.bond_dim > 0:
                # node updates with bonds
                h_node_with_bond = self.node_blocks_with_bond[i](h_node, bond_index, h_bond)
                # bond updates
                h_bond = h_bond + self.bond_blocks[i](h_bond, bond_index, h_node)
            else:
                h_node_with_bond = 0
            h_node = h_node + self.lin_node(h_node_with_edge + h_node_with_bond)
        if return_edge:
            return {
                'h_node': h_node,
                'h_bond': h_bond,
                'edge_attr': edge_attr,
                'edge_index': edge_index,
            }
        else:
            return {
                'h_node': h_node,
                'h_bond': h_bond,
            }

    @property
    def additional_edge_feat(self):
        return 6

    def _build_edges(self, pos, batch, is_mol, is_frag):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source') 
        # distance
        distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
        edge_attr = self.distance_expansion(distance)

        # 6-vector represent the two node types (atoms of protein or mol or frag)
        edge_src_feat = is_mol[edge_index[0]].long()
        edge_src_feat = edge_src_feat + 2 * is_frag[edge_index[0]].long()
        edge_dst_feat = is_mol[edge_index[1]].long()
        edge_dst_feat = edge_dst_feat + 2 * is_frag[edge_index[1]].long()
        edge_type_feat = torch.cat([
            F.one_hot(edge_src_feat, num_classes=3),
            F.one_hot(edge_dst_feat, num_classes=3),
        ], axis=-1)

        edge_attr = torch.cat([edge_attr, edge_type_feat], axis=-1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr, edge_index

    def _build_bond_atten(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        for node in torch.unique(left_node):
            ind_connect_left = (left_node == node)
            idx_connect_left = torch.nonzero(ind_connect_left)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            ind_connect_right = (right_node == node)
            idx_connect_right = torch.nonzero(ind_connect_right)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right



    def _build_bond_atten2(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        left_node_unique = torch.unique(left_node).cpu().numpy()
        right_node_unique = torch.unique(right_node).cpu().numpy()
        group2node_dict_left = {l:[] for l in left_node_unique}
        group2node_dict_right = {l:[] for l in right_node_unique}
        for i, node in enumerate(left_node.cpu().numpy()):
            group2node_dict_left[node] += [i]
        for i, node in enumerate(right_node.cpu().numpy()):
            group2node_dict_right[node] += [i]
        for node in left_node_unique:
            # ind_connect_left = (left_node == node)
            # idx_connect_left = torch.nonzero(ind_connect_left)[:, 0]
            idx_connect_left = torch.LongTensor(group2node_dict_left[node]).to(bond_index.device)
            # assert (idx_connect_left == idx_connect_left2).all()
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            # ind_connect_right = (right_node == node)
            # idx_connect_right = torch.nonzero(ind_connect_right)[:, 0]
            idx_connect_right = torch.LongTensor(group2node_dict_right[node]).to(bond_index.device)
            # assert (idx_connect_right == idx_connect_right2).all()
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right

class PosPredictor(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

        self.bond_dim = bond_dim
        if bond_dim > 0:
            self.left_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.right_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.bond_lin = BondFFN(bond_dim, bond_dim, node_dim, use_gate, out_dim=1)

    def forward(self, h_node, pos_node, h_bond, bond_index, h_edge, edge_index, is_frag):
        # 1 pos update through edges
        is_left_frag = is_frag[edge_index[0]]
        edge_index_left, edge_index_right = edge_index[:, is_left_frag]
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        right_feat = self.right_lin_edge(h_node[edge_index_right])
        weight_edge = self.edge_lin(h_edge[is_left_frag], left_feat * right_feat)
        force_edge = weight_edge * (pos_node[edge_index_left] - pos_node[edge_index_right])
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])

        # 2 pos update through bonds
        if self.bond_dim > 0:
            is_left_frag = is_frag[bond_index[0]]
            bond_index_left, bond_index_right = bond_index[:, is_left_frag]

            left_feat = self.left_lin_bond(h_node[bond_index_left])
            right_feat = self.right_lin_bond(h_node[bond_index_right])
            weight_bond = self.bond_lin(h_bond[is_left_frag], left_feat * right_feat)
            force_bond = weight_bond * (pos_node[bond_index_left] - pos_node[bond_index_right])
            delta_pos = delta_pos + scatter_sum(force_bond, bond_index_left, dim=0, dim_size=h_node.shape[0])
        
        pos_update = pos_node + delta_pos / 10.
        return pos_update #TODO: use only frag pos instead of all pos to save memory
