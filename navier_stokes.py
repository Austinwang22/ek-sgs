import math
import torch
from .solvers import NavierStokes2d


class ForwardNavierStokes2d(object):
    def __init__(self, 
                 resolution=128, L=2 * math.pi,
                 forward_time=1.0,
                 Re=200.0, 
                 downsample_factor=2,
                 unnormalize_factor=10.0, 
                 sigma_noise=0.0,
                 device=torch.device('cuda'), 
                 dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.unnormalize_factor = unnormalize_factor
        self.solver = NavierStokes2d(resolution, resolution, L, L, device=device, dtype=dtype)
        self.force = self.get_forcing(resolution, L)

        self.downsample_factor = downsample_factor
        self.forward_time = forward_time
        self.Re = Re
        self.sigma_noise = sigma_noise
        

    def get_forcing(self, resolution, L):
        t = torch.linspace(0, L, resolution+1, 
                           device=self.device, dtype=self.dtype)[0:-1]
        _, y = torch.meshgrid(t, t, indexing='ij')
        return - 4 * torch.cos(4.0 * y)

    @torch.no_grad()
    def __call__(self, u, unnormalize=True):
        '''
        Args:
            - u: velocity field of shape (batch_size, 1, resolution, resolution)

        Returns:
            - u: solution velocity field of shape (batch_size, resolution, resolution), torch.float32
        '''
        # Solve for the velocity field
        sol = self.forward(u, unnormalize)
        # Add noise
        sol += self.sigma_noise * torch.randn_like(sol)
        return sol
    
    @torch.no_grad()
    def forward(self, u, unnormalize=True):
        '''
        Args:
            - u: velocity field of shape (batch_size, 1, resolution, resolution)

        Returns:
            - u: solution velocity field of shape (batch_size, resolution, resolution), torch.float32
        '''
        # Solve for the velocity field
        if unnormalize:
            raw_u = u * self.unnormalize_factor
        else:
            raw_u = u

        sol = self.solver.solve(raw_u.squeeze(1), self.force, self.forward_time, self.Re, adaptive=True)
        # Downsample the velocity field
        sol = sol[..., ::self.downsample_factor, ::self.downsample_factor]
        return sol.unsqueeze(1).to(torch.float32)
    
    def loglikelihood(self, inputs, observation, exact=True, post_forward=False, unnormalize=True):
        '''
        Compute the log-likehood
        Args:
            - inputs (torch.Tensor): if post_forward is False, inputs is the initial velocity field
                                     if post_forward is True, inputs is the measurements
            - observation (torch.Tensor): the observation of shape (1, 1, resolution, resolution)
            - exact (bool): whether to use the exact log-likelihood
            - post_forward (bool): whether the inputs are measurements
            - unnormalize (bool): whether to unnormalize the inputs
        '''
        if post_forward:
            measurement = inputs
        else:
            measurement = self.forward(inputs, unnormalize=unnormalize)
        
        y_diff = (measurement - observation).reshape(measurement.shape[0], -1)
        if self.sigma_noise > 0.0 and exact:
            return -0.5 * torch.sum(y_diff ** 2, dim=1) / self.sigma_noise ** 2
        else:
            return -0.5 * torch.sum(y_diff ** 2, dim=1)
