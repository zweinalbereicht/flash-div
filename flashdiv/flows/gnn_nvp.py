from flashdiv.flows.gcl import  E_GCL
# from flashdiv.flows.gcl import  E_GCL, GCLv2
import torch
import torch.nn as nn

class EasyTrace_EGNN(nn.Module):
    def __init__(self,
                 n_particles, n_dimension,  hidden_nf=64,
                 act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=False, tanh=False, agg='sum', *, remove_com=True):
        super().__init__()
        self.n_particles, self.n_dimension = n_particles, n_dimension
        # self.ignn = IGNNv2(in_node_nf=2, out_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf,  act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        self.ignn = IGNN(in_node_nf=2, out_node_nf=1, in_edge_nf=1, hidden_nf=hidden_nf,  act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg)
        self.remove_com = remove_com
        self.edges = self._create_edges()
        self._edges_dict = {}

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self.n_particles-1):
            for j in range(i + 1, self.n_particles-1):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes, device):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(device)
            cols_total = torch.cat(cols_total).to(device)
            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


    def _compute_gnn(self, x, t, *, return_xdiff=False):
        """
        Memory-friendly version: loop over excluded index `i`
        instead of materialising every (i,j) graph at once.
        """
        B, P, D = x.shape
        dev     = x.device
        V       = P - 1                         # nodes per sub-graph
        # --- build table of “all but i” once --------------------------------
        full   = torch.arange(P, device=dev).repeat(P, 1)     # (P,P)
        idx_excl = full[~torch.eye(P, dtype=torch.bool, device=dev)].view(P, V)                              # (P,V)
        edges = self._cast_edges2batch(self.edges, B*(P-1), P-1, dev)  # cast edges to batch
        W_rows   = []      # to collect (B , V) blocks per i
        if return_xdiff:
            dx_rows = []

        for i in range(P):
            # ----------------------------------------------------------------
            # (1)  coordinates & differences  (B , V , D)
            # ----------------------------------------------------------------
            x_other = x[:, idx_excl[i]]                       # exclude i
            if return_xdiff:
                dx = x[:, i:i+1] - x_other                   # (B,1,D) - (B,V,D)
            # ----------------------------------------------------------------
            # (2)  node features  (B , V , 2)
            #     channel-0 = marker j (one-hot via IGNN, so just j index)
            #     channel-1 = time t
            # ----------------------------------------------------------------
            h = torch.zeros(B, V, V, 2, device=dev)             # (B,V,2)
            eye = torch.eye(P-1, device=dev)                              # (P-1, P-1)
            h[..., 0] = eye                                             # broadcast to B,P
            t_expanded = t.view(B, 1, 1).expand(-1, P-1, P-1)       # (B,P,P-1,P-1)
            h[..., 1] = t_expanded
            x_flat = x_other.unsqueeze(1).expand(-1, V, -1, -1).reshape(-1,D)
            # flatten nodes for IGNN
            h_flat = h.reshape(-1, 2)
            edge_attr = (x_flat[edges[0]] - x_flat[edges[1]]).pow(2).sum(-1, keepdim=True)
            # IGNN forward for this i
            w_flat = self.ignn(h_flat, x_flat, edges, edge_attr=edge_attr)
            # w_flat = self.ignn(h_flat,  edges, edge_attr=edge_attr)
            W_rows.append(w_flat.view(B, V, V).sum(dim=-1))            # append (B,V)
            if return_xdiff:
                dx_rows.append(dx)                      # (B,1,V,D) after unsqueeze later
        # --------------------------------------------------------------------
        # (3)  stack results into original layout
        # --------------------------------------------------------------------
        W = torch.stack(W_rows, dim=1)                  # (B , P , V)
        if return_xdiff:
            xdiff = torch.stack(dx_rows, dim=1)         # (B , P , V , D)
            return W, xdiff, f
        else:
            return W,f

    def _compute_gnn_full(self, x, t, return_xdiff):
        dev     = x.device
        B, P, D = x.shape
        edges = self._cast_edges2batch(self.edges, B*P*(P-1), P-1, dev)  # cast edges to batch
        edges = [edges[0], edges[1]]
        v = torch.zeros_like(x)               # (B,P,D)
                # ------------------------------------------------------------------
        # (1)  x_broadcast  -------------------------------------------------
        # ------------------------------------------------------------------
        # indices of “all but i”
        full   = torch.arange(P, device=dev).repeat(P, 1)             # (P,P)
        other  = full[~torch.eye(P, dtype=torch.bool, device=dev)]    # drop diag
        other  = other.view(P, P-1)                                   # (P, P-1)
        # gather → (B, P, P-1, D)
        x_other = x[:, other]                                         # advanced indexing
        # duplicate along new j-axis → (B, P, P-1, P-1, D)
        x_broadcast = x_other.unsqueeze(2).expand(-1, -1, P-1, -1, -1)
        # ------------------------------------------------------------------
        # (2)  h tensor  ----------------------------------------------------
        # ------------------------------------------------------------------
        h = torch.zeros(B, P, P-1, P-1, 2, device=dev)
        # identity matrix on channel-0
        eye = torch.eye(P-1, device=dev)                              # (P-1, P-1)
        # h[..., 0] = eye                                               # broadcast to B,P
        # time channel-1
        t_expanded = t.view(B, 1, 1, 1).expand(-1, P, P-1, P-1)       # (B,P,P-1,P-1)
        h[..., 1] = t_expanded
        x_flat = x_broadcast.reshape(B*P*(P-1)*(P-1), D)
        h_flat = h.reshape(B*P*(P-1)*(P-1), 2)
        # edges & distances for P-1 nodes
        edge_attr = torch.sum((x_flat[edges[0]] - x_flat[edges[1]])**2, dim=1, keepdim=True)
        # -------- 2) IGNN -> weights for node i -------------------
        f = self.ignn(h_flat, x_flat, edges, edge_attr=edge_attr)
        # W = f.view(B, P, P-1, P-1).sum(dim=-1)          # (b, i, j) -> (B, P, P-1)
        W = f.view(B, P, P-1, P-1)[...,0,:]         # (b, i, j) -> (B, P, P-1)
        if return_xdiff:
            xdiff = x.unsqueeze(2) - x_other.view(B, P, P-1, D)
            return W, xdiff, f
        else:
            return W,f

    def forward(self, x, t):
        W, xdiff, f = self._compute_gnn_full(x, t, return_xdiff=True)  # (B, P, P-1), (B, P, P-1, D)
        #W_full, xdiff_full = self._compute_gnn_full(x, t, return_xdiff=True)  # (B, P, P-1)
        # print(“error:“,(W-W_full).abs().max())
        # print(“xdiff error:“, (xdiff-xdiff_full).abs().max())

        v  = (xdiff * W.unsqueeze(-1)).sum(dim=2)          # (B , P , D)
        if self.remove_com:
            v = v - v.mean(dim=1, keepdim=True)
        return v, f

    def divergence(self, x, t):
        W = self._compute_gnn(x, t, return_xdiff=False)
        return W.sum(dim=(1,2))*x.shape[-1]

class IGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(IGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))
    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)
        return h

class IGNNv2(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(IGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCLv2(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, agg=agg))

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        if edge_attr is None:
            raise ValueError("Edge attributes must be provided for IGNNv2.")

        h = self.embedding(h)

        for i in range(0, self.n_layers):
            h, edge_attr = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)
        return h