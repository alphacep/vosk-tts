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
            n_layers = 4,
            kernel_size = 3,
            p_dropout = 0.1,
            gin_channels = spk_emb_dim,
        )


        self.dp_encoder = Encoder(
            out_channels = 50,
            hidden_channels = 256,
            filter_channels = 1024,
            n_heads = 4,
            n_layers = 4,
            kernel_size = 3,
            p_dropout = 0.1,
            gin_channels = spk_emb_dim,
        )

        # 256 transformer dim

        self.scale = 160 ** 0.5
        self.emb = nn.Embedding(n_vocab, 160)
        nn.init.normal_(self.emb.weight, 0.0, 160**-0.5)

        self.punc_scale = 16 ** 0.5
        self.punc_emb = nn.Embedding(n_vocab, 16)
        nn.init.normal_(self.punc_emb.weight, 0.0, 16**-0.5)

        # We want to transform BERT into something useful, we also want to regularize bert vectors
        self.bert_proj = nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 32))


    def forward(self, x, x_lengths, spks=None, dur_spks=None, bert=None):

        x0 = self.emb(x[:,0,:]) * self.scale  # [b, t, h]
        x0 = x0.transpose(1, -1)  # [b, h, t]

        x1 = self.punc_emb(x[:,1,:]) * self.punc_scale  # [b, t, h]
        x1 = x1.transpose(1, -1)  # [b, h, t]

        x2 = self.punc_emb(x[:,2,:]) * self.punc_scale  # [b, t, h]
        x2 = x2.transpose(1, -1)  # [b, h, t]

        x3 = self.punc_emb(x[:,3,:]) * self.punc_scale  # [b, t, h]
        x3 = x3.transpose(1, -1)  # [b, h, t]

        x4 = self.punc_emb(x[:,4,:]) * self.punc_scale  # [b, t, h]
        x4 = x4.transpose(1, -1)  # [b, h, t]

        br = self.bert_proj(bert.transpose(1, -1)).transpose(1, -1)

#       Disable BERT here if needed
#        br = br.mean(dim=1, keepdim=True).expand_as(br)

        x = torch.cat([x0, x1, x2, x3, x4, br], dim=1)
#        print ("Input x", x.size())

        x_mask = sequence_mask(x_lengths, x.size(-1)).unsqueeze(1).to(x.dtype)

        x_mel, mu_mel = self.encoder(x, spks, x_mask)
        x_dp, mu_dp = self.dp_encoder(x, dur_spks, x_mask)

        return x, x_mel, mu_mel, x_dp, mu_dp, x_mask
