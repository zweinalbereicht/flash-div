import torch
import math
import numpy as np
from flashdiv.lj.sampling import even_spacing
from mpmath import jtheta # wrapped gaussian computations
import mpmath as mp
mp.dps = 25

class BaseDistribution(torch.nn.Module):
    def __init__(self, nparticles=4, dim=2, batch_size=10, device='cuda'):
        super(BaseDistribution, self).__init__()
        self.batch_size = batch_size
        self.nparticles = nparticles
        self.dim = dim
        self.param = torch.nn.Parameter(torch.randn(batch_size,nparticles,dim).to(device))

    @property
    def state(self):
        return self.param

    def potential(self, x):
        pass

    def forward(self, x=None):
        return self.log_prob(x)

    def log_prob(self, x=None):
        if x is None:
            x = self.param
        return -self.potential(x)

    def grad_log_prob(self, x=None):
        if x is None:
            x = self.param
        x.requires_grad_(True)
        with torch.enable_grad():
            return  -torch.autograd.grad(-self.potential(x), x,
                                         torch.ones(x.shape[0]).to(x.device), create_graph=True)[0]

    def neg_force_clipped(self, x=None, max_val=80):
        if x is None:
            x = self.param
        return torch.clip(self.grad_log_prob(x),-max_val,max_val)

    def reset_parameters(self):
        self.param.data = torch.randn(self.batch_size,self.nparticles,self.dim).to(self.param.device)

class LJ(BaseDistribution):
    def __init__(self, nparticles=10, dim=3, batch_size=10,
                  device="cuda", boxlength=None, kT=1,
                  epsilon=1., sigma=1., cutoff=None, shift=True,
                  periodic=False, spring_constant=0.5):
        super(LJ, self).__init__(nparticles=nparticles, dim=dim, batch_size=batch_size, device=device)
        self.kT = kT
        self.epsilon=epsilon
        self.sigma=sigma
        self.cutoff=cutoff
        self.shift=shift
        self.boxlength=boxlength
        self.periodic=periodic
        self.spring_constant=spring_constant


    def potential(self,particle_pos, min_dist=None, turn_off_harmonic=False):
        """
        Calculates Lennard_Jones potential between particles
        Arguments:
        particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        boxlength: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        Returns:
        total_potential: A tensor of shape (n_particles, n_dimensions)
        representing the total potential of the system
        """
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        rem_dims = distances.shape[:-2]
        n = distances.shape[-1]
        distances = distances.flatten(start_dim=-2)[...,1:].view(*rem_dims,n-1, n+1)[...,:-1].reshape(*rem_dims,n, n-1)
        scaled_distances = distances
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            if self.shift:
                pow_6_shift = (self.sigma/self.cutoff)**6
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6 - pow_6_shift**2+pow_6_shift)
            else:
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        else:
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        pair_potential = pair_potential *distances_inverse*distances
        total_potential = torch.sum(pair_potential,axis=(-1,-2))/2
        if not self.periodic and not turn_off_harmonic:
            total_potential += self.harmonic_potential(particle_pos)
        return total_potential


    def log_likelihood(self,particle_pos, min_dist=None):
        """
        heat version of logprob
        """
        return - self.potential(particle_pos, min_dist=min_dist) / self.kT


    def harmonic_force(self,particle_pos):
        com = torch.mean(particle_pos,axis=-2,keepdim=True)
        rel_pos = particle_pos - com
        harm_f = -self.spring_constant*rel_pos
        return harm_f

    def harmonic_potential(self,particle_pos):
        com = torch.mean(particle_pos,axis=-2,keepdim=True)
        rel_pos = particle_pos - com
        harm_pot = 0.5*self.spring_constant*(rel_pos)**2
        return harm_pot.sum(axis=(-1,-2))

    def grad_log_prob(self, x=None):
        if x is None:
            x = self.param
        return self.force(x)

    def force(self,particle_pos,min_dist=None):
        """
        Calculates Lennard_Jones force between particles
        Arguments:
            particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        box_length: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ

        Returns:
            total_force_on_particle: A tensor of shape (n_particles, n_dimensions)
        representing the total force on a particle
         """
        eps = self.epsilon
        sig = self.sigma
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        scaled_distances = distances + (distances == 0)
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(sig*distances_inverse, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        else:
            pow_6 = torch.pow(sig/scaled_distances, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        force_mag = force_mag * distances_inverse
        force = -force_mag.unsqueeze(-1) * pair_vec
        total_force = torch.sum(force, dim=1)
        if not self.periodic:
            total_force += self.harmonic_force(particle_pos)
        return total_force

    # unfortunately this one breaks when particles start to be too far away.
    def pair_vec(self,particle_pos):
        # start by sending them all inside the box
        if self.periodic:
            # print('in here')
            # first bring them in the same box
            particle_pos = particle_pos % self.boxlength
            pair_vec = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))

            to_subtract = ((torch.abs(pair_vec)> 0.5 * self.boxlength)
                        * torch.sign(pair_vec) * self.boxlength ) # we need to account for rogue particles. Putting the COM is not enough
            pair_vec -= to_subtract
        else:
            pair_vec = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        return pair_vec

    def g_r(self,particle_pos, bins=100):
        dim = particle_pos.shape[-1]
        pair_vec = self.pair_vec(particle_pos)
        nsamples = len(pair_vec)
        nparticles = pair_vec.shape[1]
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        # remove diagonal zeros
        rem_dims = distances.shape[:-2]
        distances = distances.flatten(start_dim=-2)[...,1:].view(
        *rem_dims,nparticles-1, nparticles+1)[...,:-1].reshape(*rem_dims,nparticles, nparticles-1)
        if self.periodic:
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            if dim == 2:
                bulk_density = nparticles/(self.boxlength**2)
                areas = math.pi*(bins[1:]**2-bins[:-1]**2)
            elif dim == 3:
                bulk_density = nparticles/(self.boxlength**3)
                areas = 4*math.pi*bins[1:]**2
            g_r = counts/(nparticles-1)**2/nsamples/areas/bulk_density
        else:
            com = torch.mean(particle_pos,axis=1)
            dist_from_com = torch.linalg.norm(particle_pos-com.unsqueeze(1),axis=-1)
            com_atom = torch.min(dist_from_com,axis=1)[1]
            distances = distances[torch.arange(distances.shape[0]),com_atom].flatten()
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            bulk_density = nparticles/(4/3*math.pi*(self.boxlength/2)**3)
            areas = 4*math.pi*((bins[:-1]+bins[1:])/2)**2*(bins[1:]-bins[:-1])
            g_r = counts/nsamples/areas/bulk_density
        return bins, g_r

    def sample_wrapped_gaussian(self, std, size=1, device=None, logprobs = False):
        """
        Sample from an N-D Gaussian and wrap the result into a box of size `wrap`.
        Args:
            mean: array-like, shape (dim,)
            cov: array-like, shape (dim, dim)
            wrap: float, the box size to wrap into (default 2*pi)
            size: int, number of samples
            device: torch.device or None
        Returns:
            samples: torch.Tensor, shape (size, particles, dim)
        """

        # this is a wrapped gaussian pdf
        def f(x, mu):
            q = np.exp(- (std / self.boxlength * 2 * np.pi) ** 2 / 2)
            z = np.pi * (x-mu) / self.boxlength
            return float(jtheta(3, z, q) / (self.boxlength))

        mean = even_spacing(self.nparticles, self.boxlength, self.dim).flatten()
        # print(mean)
        cov = np.eye(mean.shape[0]) * std ** 2
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        # dim = mean.shape[0]
        samples = np.random.multivariate_normal(mean, cov, size=size)
        # print(samples - )
        samples_wrapped = np.mod(samples, self.boxlength)
        samples_wrapped = torch.tensor(samples_wrapped, dtype=torch.float32)
        # print(samples_wrapped.shape, torch.tensor(mean).unsqueeze(0).expand(size, -1).shape)
        # compute sample log prob

        if logprobs:
            logprobs_ = torch.log(torch.tensor([f(s.item(), m.item()) for s, m in zip(samples_wrapped.flatten(), torch.tensor(mean).unsqueeze(0).expand(size, -1).flatten())]))
            logprobs_ = logprobs_.reshape(size, self.nparticles, self.dim).sum(dim=(-1,-2))
        else:
            logprobs_ = torch.zeros(size, device=samples_wrapped.device)

        if device is not None:
            samples_wrapped = samples_wrapped.to(device)
            logprobs_ = logprobs_.to(device)
        return  samples_wrapped.reshape(-1, self.nparticles, self.dim), logprobs_
