import torch
import numpy as np
from torch import nn
from einops import rearrange, repeat, reduce
from flashdiv.flows.flow_net_torchdiffeq import FlowNet
import torch.nn.functional as F



class MessagePassing(FlowNet):
    """
    A class for message passing in a graph neural network.
    """
    def __init__(self,
                 n_particles,
                 dim,
                 device,
                 node_feature_d=2,
                 edge_feature_d=2,
                 message_feature_d=2,
                 fused_message_feature_d=2,
                 out_node_d=10,
                 potential_d=2,
                 layers=3,
                 activation=nn.SiLU(),
                 attention=False):

        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.device = device
        self.edges = self._create_edges(self.n_particles)
        self._edges_dict = {}
        self.node_feature_d = node_feature_d  # output node feature size
        self.edge_feature_d = edge_feature_d
        self.message_feature_d = message_feature_d
        self.fused_message_feature_d = fused_message_feature_d
        self.out_node_d = out_node_d  # output node feature size
        self.potential_d = potential_d  # output potential size
        self.layers = layers  # number of message passing layers
        self.attention = attention  # whether to use attention in the message passing
        self.activation = activation  # activation function

        # take time away in embeddings
        self.edge_embedding = nn.Sequential(
        nn.Linear(1, self.edge_feature_d),
        # nn.ReLU(),
        # nn.Linear(self.edge_feature_d, self.edge_feature_d)
        ).to(device)

        # take time away in embeddings
        self.node_embedding = nn.Sequential(
        nn.Linear(1, self.node_feature_d),
        # nn.ReLU(),
        # nn.Linear(self.node_feature_d, self.node_feature_d)
        ).to(device)  # output node feature size

        # probably don't want time here either
        self.node_decoding = nn.Sequential(
            nn.Linear(self.node_feature_d, self.out_node_d)).to(device)

        self.message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.node_feature_d + self.edge_feature_d + 1, self.message_feature_d),
                self.activation,
                nn.Linear(self.message_feature_d, self.message_feature_d),
            ).to(device) for _ in range(self.layers)
        ])

        self.fused_message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.message_feature_d + 1, self.fused_message_feature_d),
                self.activation,
                nn.Linear(self.fused_message_feature_d, self.fused_message_feature_d)
            ).to(device) for _ in range(self.layers)
        ])

        self.node_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_feature_d + self.fused_message_feature_d + 1, self.node_feature_d),
                self.activation,
                nn.Linear(self.node_feature_d, self.node_feature_d)
            ).to(device) for _ in range(self.layers)
        ])

        # this is the final potential --> could be any sort of potential with fixed parameters.
        # We make it more complicated here to allow for more complex potentials.
        self.potential_model = nn.Sequential(
            nn.Linear(1 + self.out_node_d, self.potential_d),
            self.activation,
            nn.Linear(self.potential_d, self.potential_d),
            self.activation,
            nn.Linear(self.potential_d, 1)
        ).to(device)  # output potential size

    def _create_edges(self, nb_nodes):
        rows, cols = [], []
        for i in range(nb_nodes):
            for j in range(i + 1, nb_nodes):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    # for aggregation of messages
    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self, x, t):
        B, P, D = x.shape

        # create all subgraphs
        r_x = repeat(x, 'b p d -> b p1 p d', p1 = P)
        mask = ~(torch.eye(P, dtype=torch.bool).reshape(1, P, P, 1).expand(B, P, P, D )) # repeat x to create subgraphs
        subgraph_x = r_x[mask].reshape(B, P, P-1, D)  # subgraphs for each particle in the batch
        subgraph_x_flat = subgraph_x.view(B * P *( P-1), D)  # flatten the batch and particle dimensions
        subgraph_edges = self._create_edges(P-1)
        E = subgraph_edges[0].shape[0]  # number of edges in the subgraph
        subgraph_edges_tobatch = self._cast_edges2batch(subgraph_edges, B * P, P-1)  # cast edges to batch
        ei, ej = subgraph_edges_tobatch

        diff_ij = subgraph_x_flat[ei] - subgraph_x_flat[ej]
        rij = torch.norm(diff_ij, dim=-1, keepdim=True)  # compute distances

        # expand times
        tedges = t.reshape(B, 1,1, 1).expand(B, P, E, 1).reshape(B * P * E, 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape
        tnodes = t.reshape(B, 1,1, 1).expand(B, P, P-1, 1).reshape(B * P * (P-1), 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape

        # we should start with node features all equivalent
        node_features = self.node_embedding(torch.ones_like(tnodes))  # identical at first
        edge_features = self.edge_embedding(rij)  # for now

        for k in range(self.layers):
            # print(f"Layer {k+1}/{self.layers}")

            messages = self.message_models[k](torch.cat((node_features[ei], node_features[ej], edge_features, tedges), dim = -1)) #mij

            # aggregate
            # fused_messages = self.fused_message_models[k](torch.cat((messages, tedges), dim=-1))  # fuse messages with time feature

            agg = self.unsorted_segment_sum(messages, ei, num_segments=subgraph_x_flat.size(0)) # simple sum in the aggregation for now
            # print(f"agg shape: {agg.shape}, node_features shape: {node_features.shape}, tnodes shape: {tnodes.shape}")
            node_features = self.node_models[k](torch.cat((node_features, agg, tnodes), dim=-1))  # update node features

        out = self.node_decoding(node_features)  # decode node features to output
        potential_params = out.reshape(B, P, P-1, -1)  # reshape to match subgraph shape

        diffx = (x.unsqueeze(2).expand(-1, -1, P-1, -1) - subgraph_x) #(B, P, P-1, 3)
        rdiffx = diffx.norm(dim=-1, keepdim=True)  #(B, P, P-1, 1) god rij



        potentials = self.potential_model(torch.cat((rdiffx.reshape(-1,1), potential_params.reshape(-1, self.out_node_d )), dim=-1)).reshape(B, P, P-1, -1)  # compute potentials

        # just the same as we would do in the original eqtf idea
        if self.attention:
            potentials = potentials * F.softmax(potentials, dim=-2)  # apply attention to potentials

        vel = (potentials * diffx).sum(dim=(-2))  # compute the final output, which is the sum of the potentials times the distance vector

        # we actually don't want to take the mean away beacause this introduces bad behavior in the trace
        # vel = vel - vel.mean(dim=1, keepdim=True)

        return vel

class MessagePassingDotProduct(FlowNet):
    """
    A class for message passing in a graph neural network.
    Same as above but we add a dot product as an input featue to the edges
    """
    def __init__(self,
                 n_particles,
                 dim,
                 device,
                 node_feature_d=2,
                 edge_feature_d=2,
                 message_feature_d=2,
                 fused_message_feature_d=2,
                 out_node_d=10,
                 potential_d=2,
                 layers=3,
                 activation=nn.SiLU(),
                 attention=False):

        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.device = device
        self.edges = self._create_edges(self.n_particles)
        self._edges_dict = {}
        self.node_feature_d = node_feature_d  # output node feature size
        self.edge_feature_d = edge_feature_d
        self.message_feature_d = message_feature_d
        self.fused_message_feature_d = fused_message_feature_d
        self.out_node_d = out_node_d  # output node feature size
        self.potential_d = potential_d  # output potential size
        self.layers = layers  # number of message passing layers
        self.attention = attention  # whether to use attention in the message passing
        self.activation = activation  # activation function

        # take time away in embeddings
        self.edge_embedding = nn.Sequential(
        nn.Linear(1 + 1, self.edge_feature_d),
        # nn.ReLU(),
        # nn.Linear(self.edge_feature_d, self.edge_feature_d)
        ).to(device)

        # take time away in embeddings
        self.node_embedding = nn.Sequential(
        nn.Linear(1, self.node_feature_d),
        # nn.ReLU(),
        # nn.Linear(self.node_feature_d, self.node_feature_d)
        ).to(device)  # output node feature size

        # probably don't want time here either
        self.node_decoding = nn.Sequential(
            nn.Linear(self.node_feature_d, self.out_node_d)).to(device)

        self.message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.node_feature_d + self.edge_feature_d + 1, self.message_feature_d),
                self.activation,
                nn.Linear(self.message_feature_d, self.message_feature_d),
            ).to(device) for _ in range(self.layers)
        ])

        self.fused_message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.message_feature_d + 1, self.fused_message_feature_d),
                self.activation,
                nn.Linear(self.fused_message_feature_d, self.fused_message_feature_d)
            ).to(device) for _ in range(self.layers)
        ])

        self.node_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_feature_d + self.fused_message_feature_d + 1, self.node_feature_d),
                self.activation,
                nn.Linear(self.node_feature_d, self.node_feature_d)
            ).to(device) for _ in range(self.layers)
        ])

        # this is the final potential --> could be any sort of potential with fixed parameters.
        # We make it more complicated here to allow for more complex potentials.
        self.potential_model = nn.Sequential(
            nn.Linear(1 + self.out_node_d, self.potential_d),
            self.activation,
            nn.Linear(self.potential_d, self.potential_d),
            self.activation,
            nn.Linear(self.potential_d, 1)
        ).to(device)  # output potential size

    def _create_edges(self, nb_nodes):
        rows, cols = [], []
        for i in range(nb_nodes):
            for j in range(i + 1, nb_nodes):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    # for aggregation of messages
    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self, x, t):
        B, P, D = x.shape

        # create all subgraphs
        r_x = repeat(x, 'b p d -> b p1 p d', p1 = P)
        mask = ~(torch.eye(P, dtype=torch.bool).reshape(1, P, P, 1).expand(B, P, P, D )) # repeat x to create subgraphs
        subgraph_x_nocentered = r_x[mask].reshape(B, P, P-1, D)  # subgraphs for each particle in the batch

        # substract com, it should be done before working on the subgraph
        subgraph_x = subgraph_x_nocentered - subgraph_x_nocentered.mean(dim=-2, keepdim=True)  # center the subgraph around the center of mass of subgraph
        subgraph_x_flat = subgraph_x.view(B * P *( P-1), D)  # flatten the batch and particle dimensions
        subgraph_edges = self._create_edges(P-1)
        E = subgraph_edges[0].shape[0]  # number of edges in the subgraph
        subgraph_edges_tobatch = self._cast_edges2batch(subgraph_edges, B * P, P-1)  # cast edges to batch
        ei, ej = subgraph_edges_tobatch

        diff_ij = subgraph_x_flat[ei] - subgraph_x_flat[ej]
        rij = torch.norm(diff_ij, dim=-1, keepdim=True)  # compute distances
        vdotij = (subgraph_x_flat[ei] * subgraph_x_flat[ej]).sum(dim=-1, keepdim=True)  # compute dot product between the two nodes (they have been com centered prior)

        # expand times
        tedges = t.reshape(B, 1,1, 1).expand(B, P, E, 1).reshape(B * P * E, 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape
        tnodes = t.reshape(B, 1,1, 1).expand(B, P, P-1, 1).reshape(B * P * (P-1), 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape

        # we should start with node features all equivalent
        node_features = self.node_embedding(torch.ones_like(tnodes))  # identical at first
        edge_features = self.edge_embedding(torch.cat((rij, vdotij), dim=-1))  # we cat the radial and dot product info

        for k in range(self.layers):
            # print(f"Layer {k+1}/{self.layers}")

            messages = self.message_models[k](torch.cat((node_features[ei], node_features[ej], edge_features, tedges), dim = -1)) #mij

            # aggregate
            # fused_messages = self.fused_message_models[k](torch.cat((messages, tedges), dim=-1))  # fuse messages with time feature

            agg = self.unsorted_segment_sum(messages, ei, num_segments=subgraph_x_flat.size(0)) # simple sum in the aggregation for now
            # print(f"agg shape: {agg.shape}, node_features shape: {node_features.shape}, tnodes shape: {tnodes.shape}")
            node_features = self.node_models[k](torch.cat((node_features, agg, tnodes), dim=-1))  # update node features

        out = self.node_decoding(node_features)  # decode node features to output
        potential_params = out.reshape(B, P, P-1, -1)  # reshape to match subgraph shape

        diffx = (x.unsqueeze(2).expand(-1, -1, P-1, -1) - subgraph_x_nocentered) #(B, P, P-1, 3) here we use the non-centered subgraph to compute the distance vector
        rdiffx = diffx.norm(dim=-1, keepdim=True)  #(B, P, P-1, 1) god rij



        potentials = self.potential_model(torch.cat((rdiffx.reshape(-1,1), potential_params.reshape(-1, self.out_node_d )), dim=-1)).reshape(B, P, P-1, -1)  # compute potentials

        # just the same as we would do in the original eqtf idea
        if self.attention:
            potentials = potentials * F.softmax(potentials, dim=-2)  # apply attention to potentials

        vel = (potentials * diffx).sum(dim=(-2))  # compute the final output, which is the sum of the potentials times the distance vector

        # we actually don't want to take the mean away beacause this introduces bad behavior in the trace
        # vel = vel - vel.mean(dim=1, keepdim=True)

        return vel

# for some reason this one is not permutation invariant, I need to fix it eventually.
class MessagePassingDotProductPolynomial(FlowNet):
    """
    A class for message passing in a graph neural network.
    Same as above but we add a dot product as an input featue to the edges
    We restrict the output potential to be a polynomial of order [-m, m]
    """
    def __init__(self,
                 n_particles,
                 dim,
                 device,
                 node_feature_d=2,
                 edge_feature_d=2,
                 message_feature_d=2,
                 fused_message_feature_d=2,
                 out_node_d=10,
                 polynomial_order = 2, #[-m, m]
                 layers=3,
                 activation=nn.SiLU(),
                 attention=False):

        super().__init__()
        self.n_particles = n_particles
        self.dim = dim
        self.device = device
        self.edges = self._create_edges(self.n_particles)
        self._edges_dict = {}
        self.node_feature_d = node_feature_d  # output node feature size
        self.edge_feature_d = edge_feature_d
        self.message_feature_d = message_feature_d
        self.fused_message_feature_d = fused_message_feature_d
        self.out_node_d = out_node_d  # output node feature size
        self.polynomial_order = polynomial_order  # output potential size
        self.layers = layers  # number of message passing layers
        self.attention = attention  # whether to use attention in the message passing
        self.activation = activation  # activation function

        # take time away in embeddings
        self.edge_embedding = nn.Sequential(
        nn.Linear(1 + 1, self.edge_feature_d),
        ).to(device)

        # take time away in embeddings
        self.node_embedding = nn.Sequential(
        nn.Linear(1, self.node_feature_d),
        ).to(device)  # output node feature size

        # probably don't want time here either
        self.node_decoding = nn.Sequential(
            nn.Linear(self.node_feature_d, 2 * self.polynomial_order + 1)).to(device)

        self.message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.node_feature_d + self.edge_feature_d + 1, self.message_feature_d),
                self.activation,
                nn.Linear(self.message_feature_d, self.message_feature_d),
            ).to(device) for _ in range(self.layers)
        ])

        self.fused_message_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.message_feature_d + 1, self.fused_message_feature_d),
                self.activation,
                nn.Linear(self.fused_message_feature_d, self.fused_message_feature_d)
            ).to(device) for _ in range(self.layers)
        ])

        self.node_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_feature_d + self.fused_message_feature_d + 1, self.node_feature_d),
                self.activation,
                nn.Linear(self.node_feature_d, self.node_feature_d)
            ).to(device) for _ in range(self.layers)
        ])


    def _create_edges(self, nb_nodes):
        rows, cols = [], []
        for i in range(nb_nodes):
            for j in range(i + 1, nb_nodes):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    # for aggregation of messages
    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    def forward(self, x, t):
        B, P, D = x.shape

        # create all subgraphs
        r_x = repeat(x, 'b p d -> b p1 p d', p1 = P)
        mask = ~(torch.eye(P, dtype=torch.bool).reshape(1, P, P, 1).expand(B, P, P, D )) # repeat x to create subgraphs
        subgraph_x_nocentered = r_x[mask].reshape(B, P, P-1, D)  # subgraphs for each particle in the batch

        # substract com, it should be done before working on the subgraph
        subgraph_x = subgraph_x_nocentered - subgraph_x_nocentered.mean(dim=-2, keepdim=True)  # center the subgraph around the center of mass of subgraph
        subgraph_x_flat = subgraph_x.view(B * P *( P-1), D)  # flatten the batch and particle dimensions
        subgraph_edges = self._create_edges(P-1)
        E = subgraph_edges[0].shape[0]  # number of edges in the subgraph
        subgraph_edges_tobatch = self._cast_edges2batch(subgraph_edges, B * P, P-1)  # cast edges to batch
        ei, ej = subgraph_edges_tobatch

        diff_ij = subgraph_x_flat[ei] - subgraph_x_flat[ej]
        rij = torch.norm(diff_ij, dim=-1, keepdim=True)  # compute distances
        # print(rij.min())
        vdotij = (subgraph_x_flat[ei] * subgraph_x_flat[ej]).sum(dim=-1, keepdim=True)  # compute dot product between the two nodes (they have been com centered prior)

        # expand times
        tedges = t.reshape(B, 1,1, 1).expand(B, P, E, 1).reshape(B * P * E, 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape
        tnodes = t.reshape(B, 1,1, 1).expand(B, P, P-1, 1).reshape(B * P * (P-1), 1) #(B * P * (P-1), 1)  # time feature, broadcasted to match subgraph shape

        # we should start with node features all equivalent
        node_features = self.node_embedding(torch.ones_like(tnodes))  # identical at first
        edge_features = self.edge_embedding(torch.cat((rij, vdotij), dim=-1))  # we cat the radial and dot product info

        for k in range(self.layers):
            # print(f"Layer {k+1}/{self.layers}")

            messages = self.message_models[k](torch.cat((node_features[ei], node_features[ej], edge_features, tedges), dim = -1)) #mij

            # aggregate
            # fused_messages = self.fused_message_models[k](torch.cat((messages, tedges), dim=-1))  # fuse messages with time feature

            agg = self.unsorted_segment_sum(messages, ei, num_segments=subgraph_x_flat.size(0)) # simple sum in the aggregation for now
            # print(f"agg shape: {agg.shape}, node_features shape: {node_features.shape}, tnodes shape: {tnodes.shape}")
            node_features = self.node_models[k](torch.cat((node_features, agg, tnodes), dim=-1))  # update node features

        out = self.node_decoding(node_features)  # decode node features to output
        potential_params = out.reshape(B, P, P-1, -1)  # reshape to match subgraph shape

        diffx = (x.unsqueeze(2).expand(-1, -1, P-1, -1) - subgraph_x_nocentered) #(B, P, P-1, 3) here we use the non-centered subgraph to compute the distance vector
        rdiffx = diffx.norm(dim=-1, keepdim=True)  #(B, P, P-1, 1) god rij

        potentials = (rdiffx.reshape(-1,1) ** torch.arange(-self.polynomial_order, self.polynomial_order + 1).to(self.device)).reshape(B, P, P-1, -1)  # compute polynomial potentials

        # print('r powers')
        # print(potentials)
        # print('params')
        # print(potential_params)
        potentials = (potentials * potential_params).sum(-1).unsqueeze(-1)  #(B, P, P-1, 1)
        # print('potential value')
        # print(potentials)
        # print(potentials.shape)
        # potentials = self.potential_model(torch.cat((rdiffx.reshape(-1,1), potential_params.reshape(-1, self.out_node_d )), dim=-1)).reshape(B, P, P-1, -1)  # compute potentials
        # print(potentials)
        # just the same as we would do in the original eqtf idea
        if self.attention:
            potentials = potentials * F.softmax(potentials, dim=-2)  # apply attention to potentials

        vel = (potentials * diffx).sum(dim=(-2))  # compute the final output, which is the sum of the potentials times the distance vector

        # we actually don't want to take the mean away beacause this introduces bad behavior in the trace
        # vel = vel - vel.mean(dim=1, keepdim=True)

        return vel