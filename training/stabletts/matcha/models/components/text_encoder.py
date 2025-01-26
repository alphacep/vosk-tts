""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

import matcha.utils as utils  # pylint: disable=consider-using-from-import
from matcha.utils.model import sequence_mask
from matcha.models.components.diffusion_transformer import DiTConVBlock







class Encoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        

        self.encoder = nn.ModuleList([DiTConVBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout, gin_channels) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for block in self.encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, x_mask: torch.Tensor):

        for layer in self.encoder:
            x = layer(x, c, x_mask)
        mu_x = self.proj(x) * x_mask

        return x, mu_x








class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_type,
        encoder_params,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.encoder = Encoder(
            out_channels = encoder_params.n_feats,
            hidden_channels = 256,
            filter_channels = 1024,
            n_heads = 4,
            n_layers = 2,
            kernel_size = 3,
            p_dropout = 0.1,
            gin_channels = spk_emb_dim,
        )

        self.dp_encoder = Encoder(
            out_channels = 50,
            hidden_channels = 256,
            filter_channels = 1024,
            n_heads = 4,
            n_layers = 2,
            kernel_size = 3,
            p_dropout = 0.1,
            gin_channels = spk_emb_dim,
        )

        # 256 transformer dim

        self.scale = 256 ** 0.5
        self.emb = nn.Embedding(n_vocab, 256)
        nn.init.normal_(self.emb.weight, 0.0, 256**-0.5)
        self.bert_proj = torch.nn.Conv1d(768, 256, 1)


    def forward(self, x, x_lengths, spks=None, bert=None):

        x = self.emb(x) * self.scale  # [b, t, h]
        x = x.transpose(1, -1)  # [b, h, t]
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        br = self.bert_proj(bert)

        x = x + br

        x_dp = torch.detach(x)
        x_dp, mu_dp = self.dp_encoder(x_dp, spks, x_mask)

        x, mu = self.encoder(x, spks, x_mask)

        return mu, x, mu_dp, x_dp, x_mask
