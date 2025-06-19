import torch
import torch.nn as nn
#from flashdiv.flows.flow_net import FlowNet
from flashdiv.flows.flow_net_torchdiffeq import FlowNet
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

class ParallelEqTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim=128, activation=nn.ReLU(),auto_scale=True, nb_units=1, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.activation = activation
        self.nb_units = nb_units
        self.auto_scale = auto_scale
        self.units = nn.ModuleList([EqTransformerFlowLJ(self.input_dim, self.embed_dim, self.activation, auto_scale=self.auto_scale).to(device) for _ in range(self.nb_units)])

    def forward(self, x, t):
        out = torch.zeros_like(x)
        for unit in self.units:
            out = out + unit(x, t)
        return out

    def divergence(self, x, t):
        out = torch.zeros(x.shape[0]).to(x)
        for unit in self.units:
            out = out + unit.divergence(x, t)
        return out

class EqTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim=128, activation=nn.ReLU(), auto_scale=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.activation = activation
        self.auto_scale = auto_scale

        # we have multipe encoders to avoid the symmetry issue.
        self.encoder =  nn.Sequential(
            nn.Linear(2 + 1, self.embed_dim),
            self.activation,
            nn.Linear(self.embed_dim, 1)
        )

        self.scaler = nn.Sequential(
            nn.Linear(2 + 1, embed_dim),
            self.activation,
            nn.Linear(embed_dim, 1)
        )

        # the "self" scaler function --> unclear if it's necessary
        if self.auto_scale:
            self.auto_scaler = nn.Sequential(
                nn.Linear(1 + 1, embed_dim),
                self.activation,
                nn.Linear(embed_dim, 1)
            )


        self.edges = None
        self._edges_dict = {}


    def forward(self, x, t):
        """
        This is a rot and perm equivariant flow field, that uses the "softmax attention mechanism" on featured radial distances.
        In the Noe paper they compose multiple similar layers, but our bane, or advantage, is that we want only one
        for quick tract computations.

        Not that we can always put them in parallel. to have potentially more expressivity.

        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size)

        Outputs
        v :  (batch_size, nbpart, dim)
        """

        n_batch = x.shape[0]
        n_particles = x.shape[1]
        # steal MW's trick --> this assigns self.eges the first time.
        # Doesn't work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)

        com = reduce(
            x,
            'b p d -> b 1 d',
            'mean'
        )

        # shift everything by com
        flat_coord = rearrange(
            x - com,
            'b p d -> (b p) d'
        )

        # compute pairwaise distances and direction information
        radial, dotproduct, coord_diff = self.coord2raddotproduct(edges, flat_coord) # one big matrix of size ((b p p) 1)
        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms

        flat_features = torch.cat((radial, dotproduct, flat_t), dim=-1)
        scale = self.scaler(flat_features) #((b p p ) 1)
        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)

        # now we have to reshape to do a soft max operation.
        sm = torch.zeros(
            (n_batch, n_particles, n_particles,1),
            device=x.device,
            dtype=flat_features.dtype
        )
        diffs = torch.zeros(
            (n_batch, n_particles, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype
        )

        scales = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)

        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles

        sm[batches, i_mod, j_mod] = flat_features
        diffs[batches, i_mod, j_mod] = coord_diff
        scales[batches, i_mod, j_mod] = scale

        #softmax
        sm = F.softmax(sm, dim=-2)

        sm = sm * scales
        #multiply and sum
        out = reduce(
            sm * diffs, # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum')

        if self.auto_scale:
            # do the single partticle contribution
            auto_norms = (flat_coord ** 2).sum(-1, keepdim=True) + 1e-8

            flat_t = repeat(t, 'b -> (b p ) 1', p=n_particles) # no cross terms
            # print(auto_norms.shape, flat_t.shape)
            # flat_t = repeat(t, 'b -> (b p) 1', p=(batches==0).sum()) # no cross terms
            auto_scale = self.auto_scaler(
                torch.cat((auto_norms, flat_t), dim=-1)
            )
            # print(auto_norms.shape, auto_scale.shape, flat_coord.shape)

            out = out + rearrange(
                flat_coord * auto_scale,
                '(b p) d -> b p d',
                b=n_batch, p=n_particles
            )


        return out

    # This is a standard trick to have more parrallelizable computations between independent batches.
    def _create_edges(self, n_particles):
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
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
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)
            batches = rows_total // (n_nodes)

            self._edges_dict[n_batch] = [batches, rows_total, cols_total]
        return self._edges_dict[n_batch]

    # returns the dij, (xi-xj)
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        return radial, coord_diff

    # returns the xi . xj , (xi-xj)
    def coord2raddotproduct(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
        dotproduct = (coord[row] * coord[col]).sum(dim=-1, keepdim=True)

        return radial, dotproduct, coord_diff

    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)
        Outputs
        v :  (batch_size)
        """

        # start with essential compute
        n_batch = x.shape[0]
        n_particles = x.shape[1]
        # steal MW's trick --> this assigns self.eges the first time.
        # Doesn't work for batches bcs batch size might very.
        if self.edges is None:
            self.edges = self._create_edges(n_particles)
        batches,edgesi,edgesj = self._cast_edges2batch(self.edges, n_batch, n_particles)
        edges = [edgesi.to(x.device), edgesj.to(x.device)]
        batches = batches.to(x.device)

        com = repeat(
            reduce(
                x,
                'b p d -> b d',
                'mean'),
            'b d -> b p d',
            p=n_particles  # number of particles in the batch
        )

        # shift everything by com
        flat_coord = rearrange(
            x - com,
            'b p d -> (b p) d'
        )

        radial, dotproduct, coord_diff = self.coord2raddotproduct(edges, flat_coord) # one big matrix of size ((b p p) 1)
        radial.requires_grad_(True)
        dotproduct.requires_grad_(True)
        flat_t = repeat(t, 'b -> (b p ) 1', p=(batches==0).sum()) # no cross terms

        flat_features = torch.cat((radial, dotproduct, flat_t), dim=-1)

        scale = self.scaler(flat_features) #((b p p ) 1)
        # pass features throught the encoder
        flat_features = self.encoder(flat_features) #((b p p ) 1)

        #compute derivatives
        # compute gradient of scale.sum() w.r.t. radial
        (dscalerad,) = torch.autograd.grad(
            outputs=scale.sum(),
            inputs=radial,
            create_graph=False,
            retain_graph=True
        )


        (dscaledot,) = torch.autograd.grad(
            outputs=scale.sum(),
            inputs=dotproduct,
            create_graph=False,
            retain_graph=True
        )

        # radial.detach().requires_grad_(True)
        # dotproduct.detach().requires_grad_(True)

        dfeatrad, = torch.autograd.grad(
            outputs=flat_features.sum(),
            inputs=radial,
            create_graph=False,
            retain_graph=True
        )

        dfeatdot, = torch.autograd.grad(
            outputs=flat_features.sum(),
            inputs=dotproduct,
            create_graph=False,
            retain_graph=True
        )





        # these might have an extra particle, but it can be a problem for later.
        sm = torch.zeros(
            (n_batch, n_particles, n_particles,1),
            device=x.device,
            dtype=flat_features.dtype
        )
        diffs = torch.zeros(
            (n_batch, n_particles, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype
        )
        scales = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        dscalesdot = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        dscalesrad = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        dfeaturesdot = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)
        dfeaturesrad = torch.zeros_like(sm, device=x.device, dtype=flat_features.dtype)

        # fill in alll the arrays needed
        i_mod = edges[0] % n_particles
        j_mod = edges[1] % n_particles
        sm[batches, i_mod, j_mod] = flat_features
        diffs[batches, i_mod, j_mod] = coord_diff
        scales[batches, i_mod, j_mod] = scale
        dscalesdot[batches, i_mod, j_mod] = dscaledot
        dscalesrad[batches, i_mod, j_mod] = dscalerad
        dfeaturesdot[batches, i_mod, j_mod] = dfeatdot
        dfeaturesrad[batches, i_mod, j_mod] = dfeatrad

        ## we also need to construct the drad and dtot vectors
        drad = 2 * diffs # easy
        ddot_ = flat_coord[edges[1]] - 1 / n_particles *  (flat_coord[edges[0]] + flat_coord[edges[1]]) # com contributions
        ddot = torch.zeros_like(drad, device=x.device, dtype=ddot_.dtype)
        ddot[batches, i_mod, j_mod] = ddot_

        #softmax
        sm = F.softmax(sm, dim=-2)


        # divergence vector to be filled
        divergence = torch.zeros(
            (n_batch, n_particles, x.shape[-1]),
            device=x.device,
            dtype=flat_features.dtype,
            requires_grad=False
        )

        #linear term

        divergence = divergence +  reduce(
            sm * scales,  # (b, p, p, d),
            'b p1 p2 d -> b p1 d',
            'sum'
        )

        #scales term
        divergence = divergence +  reduce(
            diffs * sm * (dscalesrad * drad + dscalesdot * ddot),
            # 2 * diffs ** 2 * dscales * sm,
            'b p1 p2 d -> b p1 d',
            'sum'
        )

        #sm term

        # non cross terms
        sm_ = repeat(
            (- sm * (dfeaturesrad * drad + dfeaturesdot * ddot)).sum(-2) ,
            'b p1 d -> b p1 p2 d',
            p2=n_particles
        )

        divergence = divergence +  reduce(
            (sm_ * sm * scales * diffs),
            'b p1 p2 d -> b p1 d',
            'sum'
            )

        #cross terms
        divergence = divergence + reduce(
            diffs * scales * sm  * (dfeaturesrad * drad + dfeaturesdot * ddot),
            'b p1 p2 d -> b p1 d',
            'sum'
            )

        # need to do the autoparticle contribution
        if self.auto_scale:

            auto_norms = (flat_coord ** 2).sum(-1, keepdim=True) + 1e-8
            auto_norms.requires_grad_(True)

            flat_t = repeat(t, 'b -> (b p ) 1', p=n_particles) # no cross terms
            # print(auto_norms.shape, flat_t.shape)
            # flat_t = repeat(t, 'b -> (b p) 1', p=(batches==0).sum()) # no cross terms
            auto_scale = self.auto_scaler(
                torch.cat((auto_norms, flat_t), dim=-1)
            )

            dautoscalednorm,= torch.autograd.grad(
                outputs=auto_scale.sum(),
                inputs=auto_norms,
                create_graph=False,
                retain_graph=True
            )

            dnorm = 2 * flat_coord * (1 - 1 / n_particles)

            auto_div = rearrange(
                torch.ones_like(flat_coord) * (1 - 1 / n_particles) * auto_scale + flat_coord * dautoscalednorm * dnorm,
                '(b p) d -> b p d',
                b=n_batch, p=n_particles
            )

            divergence = divergence + auto_div

        divergence = reduce(
            divergence,
            'b p d -> b',
            'sum'
        )

        return divergence
