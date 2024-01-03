import os

import torch
from torch import nn
from torch.nn import functional as F
from utils.torchutils import *
from .mlp import MLP
from .dist import *


class Sampler(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = torch.device('cpu')
        self.nk = args.sample_k
        self.nz = args.nz
        self.share_eps = args.share_eps
        self.train_w_mean = args.train_w_mean

        self.pred_model_dim = args.tf_model_dim

        # Q net
        self.qnet_mlp = args.qnet_mlp
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, self.nk * self.nz)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, cvae,  mean=True, need_weights=False):
        agent_num = cvae.agent_num

        history_enc, agent_history = cvae.encode_history()

        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)
                eps = eps.repeat((agent_num * self.nk, 1))
            else:
                eps = torch.randn([agent_num, self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=0)

        qnet_h = self.q_mlp(agent_history)
        A = self.q_A(qnet_h).view(-1, self.nz)
        b = self.q_b(qnet_h).view(-1, self.nz)

        z = b if mean else A * eps + b
        logvar = (A ** 2 + 1e-8).log()
        sampler_dist = Normal(mu=b, logvar=logvar)

        vae_dist = cvae.get_prior(agent_history, sample_num=self.nk)
        dec_motion, attn_weights = cvae.decode_future(z, self.nk, history_enc, need_weights=need_weights)

        return dec_motion, sampler_dist, vae_dist, attn_weights

    def step_annealer(self):
        pass