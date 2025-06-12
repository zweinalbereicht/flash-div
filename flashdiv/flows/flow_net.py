import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
from torch.func import jvp, vmap, jacrev

# base class
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")

    @torch.no_grad()
    def divergence(self, x,t, div_samples=int(1e3)):
        """
        hutchison trace estimator
        """
        x0 = repeat(x, 'b p d -> (b r) p d', r=div_samples)
        t = repeat(t, 'b -> (b r)', r=div_samples)
        v = torch.randn_like(x0)

        ouptut, jacvp = jvp(lambda x : self.forward(x, t),
        (x0,),
        (v,),
        )
        # print(jacvp.shape)

        tr = reduce(
            reduce(
                rearrange(
                    v * jacvp,
                    '(b r) p d -> b r p d',
                    r=div_samples),
                'b r p d -> b r',
                'sum'),
            'b r  -> b ',
            'mean')

        # takes care of residual
        torch.cuda.empty_cache()

        return tr

    @torch.no_grad()
    def divergence_full_jacobian(self, x,t, div_samples=int(1e3)):
        """
        Computes the full jacobian and then selects the diagonal
        """

        jac = jacrev(
            lambda x, t : self.forward(x.unsqueeze(0), t.unsqueeze(0)).squeeze(0),
            argnums=0
        )

        vmapped_jac = vmap(jac, in_dims=(0, 0))

        batched_jacobian = vmapped_jac(x, t) #(b p d p d)

        return torch.einsum(
            'b p d p d -> b',
            batched_jacobian
        )

    @torch.no_grad()
    def sample(self, xs, n_steps: int=100):
        """
        ODE integration returning only the final state
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            vt = self.forward(xs, t)
            xs = xs.detach().clone() + dt * vt
        return xs


    @torch.no_grad()
    def sample_traj(self, xs, n_steps: int=100):
        """
        ODE integration returning the trajectory
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        all_xs = [xs]
        all_ts = [0.0]
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            vt = self.forward(xs, t)
            xs = xs.detach().clone() + dt * vt
            all_xs.append(xs)
            all_ts.append((i+1) * dt)
        return torch.tensor(all_ts).to(xs), torch.stack(all_xs).to(xs)

    @torch.no_grad()
    def sample_logprob(self, x0, logprob0, times,**kwargs):
        """
        ODE integration returning the trajectory and logprob
        """
        batch_size = x0.shape[0]
        npart = x0.shape[-2]
        dim = x0.shape[-1]

        if 'method' not in  kwargs:
            kwargs['method'] = 'euler'
        if 'options' not in kwargs:
            kwargs['options'] = {'step_size': 1 / 100}

        state0 = torch.cat(
            (x0,
            repeat(
                logprob0,
                'b -> b p d',
                p=npart, d=dim
            )),
            dim=0
        )

        # little reshaping here
        def integration_func(t, state):
            xs = state[:batch_size]
            t_ = torch.ones(batch_size).to(xs) * t.item()
            v = self.forward(xs, t_).detach()
            div = self.divergence(xs, t_).detach()
            return torch.cat(
                (v,
                repeat(
                    - div,
                    'b -> b p d',
                    p=npart, d=dim
                )),
                dim=0
            ).detach()

        integrated_state = odeint(integration_func, state0, times, **kwargs)
        all_xs = integrated_state[:, :batch_size]
        all_logprobs = integrated_state[:, batch_size:, 0, 0]

        return all_xs, all_logprobs

    def sample_logprob(self, xs, n_steps: int=100, **kwargs):
        """
        ODE integration returning the last position and associated logprob
        """
        dt = 1. / n_steps
        xs = xs.detach().clone()
        curr_trace = torch.zeros((xs.shape[0])).to(xs)
        curr_trace.requires_grad_(False)
        batch_size = xs.shape[0]
        for i in range(n_steps):
            t = torch.ones(batch_size).to(xs) * i * dt
            xs = xs.detach().clone()
            xs.requires_grad_(True)
            with torch.enable_grad():
                div = self.divergence(xs, t, **kwargs).detach()
                curr_trace = curr_trace + div * dt
            vt = self.forward(xs, t)
            xs = xs + dt * vt
        return xs, curr_trace
