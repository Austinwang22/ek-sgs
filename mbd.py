import torch
from sampling import ode_sampler


def mbd_mfd_sampling(target_pdf, model, y, forward, N, x_initial, num_queries=150000, sigma_max=80., sigma_min=0.002, 
                     rho=7, scale=5.0, batch_size=50000):
    model.eval()
    
    step_indices = torch.arange(N, dtype=torch.float64, device=x_initial.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (N - 1) * \
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    num_batches = num_queries // batch_size
        
    x_next = x_initial.to(torch.float64)
    
    for i in range(N):
        
        print(f'Iteration {i}: LL: {target_pdf(x_next)}')
        
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        
        numerator = torch.zeros_like(x_next)
        denominator = torch.tensor(0., device=x_next.device).to(torch.float64)
        
        for b in range(num_batches):
            # queries = torch.randn([num_queries, model.img_channels, model.img_resolution, model.img_resolution], 
            #                     device=x_initial.device) * t_cur + x_next   
            queries = torch.randn([batch_size, model.img_channels, model.img_resolution, model.img_resolution], 
                                device=x_initial.device) * t_cur + x_next
            evals = target_pdf(queries)
            numerator += (evals.view(batch_size, 1, 1, 1) * queries).sum(dim=0)
            denominator += evals.sum()
            
        # integral_est = (evals.view(num_queries, 1, 1, 1) * queries).sum(dim=0) / evals.sum()
        integral_est = numerator / denominator
        score_est = -x_next / (t_cur ** 2) + integral_est / (t_cur ** 2)
        
        dx_ll = t_cur * score_est
        
        denoised = model(x_next, t_cur).to(torch.float64)
        dx_pf = (x_next - denoised) / t_cur
        
        # err = torch.norm(y - forward(denoised))
        
        x_next += scale * (t_next - t_cur) * dx_ll + (t_next - t_cur) * dx_pf
        
    return x_next