from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

import functools
from torchdiffeq import odeint

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, fake_speaker=None, fake_content=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
#        z = torch.randn_like(mu) * temperature
        z = torch.randn(mu.size(0), 80, mu.size(-1), device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
#        print (t_span)


#        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=0.0, fake_speaker=fake_speaker, fake_content=fake_content)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=0.5, fake_speaker=fake_speaker, fake_content=fake_content)
#        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=1.0, fake_speaker=fake_speaker, fake_content=fake_content)
#        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=1.5, fake_speaker=fake_speaker, fake_content=fake_content)
#        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=2.0, fake_speaker=fake_speaker, fake_content=fake_content)
#        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=3.0, fake_speaker=fake_speaker, fake_content=fake_content)
#        return self.solve_heun(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, n_steps=n_timesteps, guidance_scale=0.5, fake_speaker=fake_speaker, fake_content=fake_content)

#        estimator = functools.partial(self.func_dphi_dt_1, mask=mask, mu=mu, spks=spks, guidance_scale=0.5, fake_speaker=fake_speaker, fake_content=fake_content)
#        trajectory = odeint(estimator, z, t_span, method='dopri5', rtol=1e-4, atol=1e-4)
        return trajectory[-1]



    def solve_euler(self, x, t_span, mu, mask, spks, cond, n_steps, training=False, guidance_scale=0.0, fake_speaker=None, fake_content=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        steps = 1
        while steps <= n_steps:
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, spks, cond, training=training, guidance_scale=guidance_scale, fake_speaker=fake_speaker, fake_content=fake_content)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < n_steps:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def solve_heun(self, x, t_span, mu, mask, spks, cond, n_steps, training=False, guidance_scale=0.0, fake_speaker=None, fake_content=None):
        """
        Fixed heun solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        #-! : reserved space for debugger
        sol = []
        steps = 1

        while steps <= n_steps:
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, spks, cond, training=training, guidance_scale=guidance_scale, fake_speaker=fake_speaker, fake_content=fake_content)
            dphi_dt_2 = self.func_dphi_dt(x + dt * dphi_dt, mask, mu, t+dt, spks, cond, training=training, guidance_scale=guidance_scale, fake_speaker=fake_speaker, fake_content=fake_content)
            
            #- Euler's -> Y'n+1' = Y'n' + h * F(X'n', Y'n')
            # x = x + dt * dphi_dt
            
            #- Heun's -> Y'n+1' = Y'n' + h * 0.5( F(X'n', Y'n') + F(X'n' + h, Y'n' + h * F(X'n', Y'n') ) )
            x = x + dt * 0.5 * (dphi_dt + dphi_dt_2)
            t = t + dt

            sol.append(x)
            if steps < n_steps:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]





    def solve_midpoint(self, x, t_span, mu, mask, cond, training=False, guidance_scale=0.0):
        """
        Fixed midpoint solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # -! : reserved space for debugger
        sol = []
        steps = 1

        while steps <= len(t_span) - 1:
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, spks, cond, training=training, guidance_scale=guidance_scale)
            dphi_dt_2 = self.func_dphi_dt(x + dt * 0.5 * dphi_dt, mask, mu, t + dt * 0.5, spks, cond, training=training, guidance_scale=guidance_scale)
            
            # - Euler's -> Y'n+1' = Y'n' + h * F(X'n', Y'n')
            # x = x + dt * dphi_dt
            
            #- midpoint -> Y'n+1' = Y'n' + h * F(X'n' + 0.5 * h, Y'n' + 0.5 * h * F(X'n', Y'n') )
            x = x + dt * dphi_dt_2
            t = t + dt

            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def func_dphi_dt(self, x, mask, mu, t, spks, cond, training=False, guidance_scale=0.0, fake_speaker=None, fake_content=None):
        print ("!!!! Inference with time ", t)
        dphi_dt = self.estimator(x, mask, mu, t, spks)

        if guidance_scale > 0.0:

            fake_speaker = fake_speaker.repeat(x.size(0), 1)
            fake_content = fake_content.repeat(x.size(0), 1, x.size(-1))
            dphi_avg = self.estimator(x, mask, fake_content, t, fake_speaker)
#            dphi_avg = self.estimator(x, mask, mu, t, fake_speaker)
            dphi_dt = dphi_dt + guidance_scale * (dphi_dt - dphi_avg)

        return dphi_dt

    def func_dphi_dt_1(self, t, x, mask, mu, spks, guidance_scale=0.0, fake_speaker=None, fake_content=None):
        print ("!!!! Inference with time ", t)
        dphi_dt = self.estimator(x, mask, mu, t, spks)

        if guidance_scale > 0.0:

            fake_speaker = fake_speaker.repeat(x.size(0), 1)
            fake_content = fake_content.repeat(x.size(0), 1, x.size(-1))
            dphi_avg = self.estimator(x, mask, fake_content, t, fake_speaker)
            dphi_dt = dphi_dt + guidance_scale * (dphi_dt - dphi_avg)

        return dphi_dt



















    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep with cutoff
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) * 0.98
        t = 1 - torch.cos(t * 0.5 * torch.pi)

#        if (t[0,0,0].item() < 0.01):
#            t[0,0,0] = 0.0
        print (t)

        # sample noise p(x_0)
        z = torch.randn_like(x1)
        self.sigma_min = 1e-2

        y = (1 - t) * z + t * x1
        u = x1 - z

        est = self.estimator(y, mask, mu, t.squeeze(), spks)

#        print (est)
#        print (u)

        loss_mse = F.mse_loss(est * mask, u * mask, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )

#        loss_l1 = F.l1_loss(est * mask, u * mask, reduction="sum") / (
#            torch.sum(mask) * u.shape[1]
#        )

#        loss_cos = (1 - F.cosine_similarity(est, u, dim=1)).mean()
#        print ("Loss MSE", loss_mse)
#        print ("Loss COS", loss_cos)
#        return loss_mse + loss_cos, y


        print ("Loss MSE", loss_mse)
        return loss_mse, y

#        print ("Loss L1", loss_l1)
#        return loss_l1, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

#        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
#        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)

        self.estimator = Decoder(noise_channels=80, cond_channels=256, hidden_channels=384, out_channels=80, filter_channels=768, dropout=0.1, n_layers=6, n_heads=4, kernel_size=3, gin_channels=spk_emb_dim, use_lsc=True)
