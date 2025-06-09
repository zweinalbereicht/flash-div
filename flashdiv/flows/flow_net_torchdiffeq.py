import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
from torch.func import jvp
# import ode solver class
from torchdiffeq import odeint

# base class
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = lambda x, t: torch.cat([x, t], dim=1)

    def forward(self, x, t):
        raise NotImplementedError("Override this method in subclasses")

    @torch.no_grad()
    def divergence2(self, x,t, div_samples=int(1e3)):
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
    def sample(self, x0, times,**kwargs):
        """
        input : x0 (batch_size, napart, dim)
        times : (n_steps, ) evaluations times

        the kwargs should corresponf to those of the odeint function
        """
        batch_size = x0.shape[0]
        npart = x0.shape[-2]
        dim = x0.shape[-1]

        if 'method' not in  kwargs:
            kwargs['method'] = 'euler'
        # print(kwargs)

        # little reshaping here
        def integration_func(t, xs):
            t_ = torch.ones(batch_size).to(xs) * t.item()
            return self.forward(xs, t_).detach()

        return odeint(integration_func, x0, times, **kwargs)


    # @torch.no_grad()
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
