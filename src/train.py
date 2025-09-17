import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def predict_x0_from_eps(xt, t, eps, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod):
    sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * eps


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class UNet(nn.Module):
    def __init__(self, model_size='XS'):
        super().__init__()
        if model_size == 'XS':
            dims = [64, 128, 256]
        else: # M
            dims = [128, 256, 512, 1024]
        
        time_dim = dims[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dims[0]),
            nn.Linear(dims[0], time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        in_out = [(3, dims[0])] + [(dims[i], dims[i+1]) for i in range(len(dims)-1)]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.GroupNorm(min(8, dim_out), dim_out),
                nn.SiLU(),
                nn.Conv2d(dim_out, dim_out, 3, padding=1),
                nn.GroupNorm(min(8, dim_out), dim_out),
                nn.SiLU(),
                nn.Conv2d(dim_out, dim_out, 2, stride=2) if ind < len(in_out) -1 else nn.Identity()
            ))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_first = ind == 0
            # For other layers: skip connection doubles input, so dim_out * 2
            input_channels = dim_out if is_first else dim_out * 2
            self.ups.append(nn.Sequential(
                nn.Conv2d(input_channels, dim_in, 3, padding=1),
                nn.GroupNorm(min(8, dim_in), dim_in),
                nn.SiLU(),
                nn.Conv2d(dim_in, dim_in, 3, padding=1),
                nn.GroupNorm(min(8, dim_in), dim_in),
                nn.SiLU(),
                nn.ConvTranspose2d(dim_in, dim_in, 2, stride=2) if not is_first else nn.Identity()
            ))
        self.final_conv = nn.Conv2d(3, 3, 1)

    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        residuals = []
        
        for down in self.downs:
            residuals.append(x)
            x = down(x)
        
        for i, up in enumerate(self.ups):
            if i > 0:  # Skip connection for all but first upsampling layer
                res = residuals.pop()
                x = torch.cat((x, res), dim=1)
            x = up(x)
            
        return self.final_conv(x)


def jasaic_loss(model, x0, t, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, num_timesteps, max_d=20, eps=1e-8):
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    xt.requires_grad_()

    eps_t = model(xt, t)
    x0_hat_t = predict_x0_from_eps(xt, t, eps_t, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)

    Δ = torch.randint(1, max_d + 1, t.shape, device=t.device)
    s = torch.clamp(t + Δ, max=num_timesteps - 1)
    r = torch.clamp(t - Δ, min=1)
    j = torch.randint(1, num_timesteps, t.shape, device=t.device)
    d = torch.clamp(t + ((Δ + 1) // 2), max=num_timesteps - 1)

    def re_noise(pred, t_new):
        return q_sample(pred, t_new, torch.randn_like(x0), sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

    with torch.no_grad():
        xs, xr, xj, xd = [re_noise(x0_hat_t.detach(), k) for k in (s, r, j, d)]

    x0_hat_s = predict_x0_from_eps(xs, s, model(xs, s), sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)
    x0_hat_r = predict_x0_from_eps(xr, r, model(xr, r), sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)
    x0_hat_j = predict_x0_from_eps(xj, j, model(xj, j), sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)
    x0_hat_d = predict_x0_from_eps(xd, d, model(xd, d), sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod)

    v = torch.randn_like(x0)

    def jvp(y, x_in):
        return torch.autograd.grad(y, (x_in,), v, retain_graph=True, create_graph=False)[0]

    Jt = jvp(x0_hat_t, xt)
    
    xs.requires_grad_()
    xr.requires_grad_()
    xj.requires_grad_()
    xd.requires_grad_()
    
    Jk_proxies = [predict_x0_from_eps(x_k, t_k, model(x_k, t_k), sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod) for x_k, t_k in [(xs,s), (xr,r), (xj,j), (xd,d)]]
    Jk_grads = [jvp(proxy, x_proxy) for proxy, x_proxy in zip(Jk_proxies, [xs, xr, xj, xd])]

    jac_gap = sum(((Jt - j_grad)**2).mean(dim=[1, 2, 3]) for j_grad in Jk_grads)

    vb = ((x0_hat_t - xt)**2).mean(dim=[1, 2, 3])
    vc = ((x0_hat_s - x0_hat_r)**2 + (x0_hat_s - x0_hat_j)**2 + (x0_hat_r - x0_hat_j)**2) / 3.0
    vc = vc.mean(dim=[1, 2, 3])
    cov = ((x0_hat_t - xt) * (x0_hat_s - x0_hat_r)).mean(dim=[1, 2, 3])
    rho = cov / (vc + eps)

    lamJ = vb / (jac_gap + eps)

    w = torch.sqrt(alphas_cumprod[s] / alphas_cumprod[t])
    val_gap = ((x0_hat_s.detach() - x0_hat_t)**2 + (x0_hat_r.detach() - x0_hat_t)**2 + (x0_hat_j.detach() - x0_hat_t)**2 + (x0_hat_d.detach() - x0_hat_t)**2).mean(dim=[1, 2, 3])

    L_base = F.mse_loss(eps_t, noise, reduction='none').view(x0.size(0), -1).mean(1)
    L_cons = val_gap + lamJ * jac_gap
    
    loss = (L_base + rho.detach() * w.detach() * L_cons).mean()
    return loss
