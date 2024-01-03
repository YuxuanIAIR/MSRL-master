import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from .dist import Normal
from .multipathtransformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .mlp import MLP
from utils.torchutils import *
from utils.utils import initialize_weights
from .posenc import PositionalAgentEncoding

""" History Encoder """


class HistoryEncoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=4):
        super().__init__()

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.he_tf_layer

        self.agent_enc_shuffle = pos_enc['agent_enc_shuffle']

        self.pooling = args.pooling

        self.in_dim = in_dim
        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = TransformerEncoder(encoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, traj_in, agent_mask, agent_enc_shuffle=None):

        agent_num = traj_in.shape[1]

        # [N*8 1 model_dim] [N*8 1 model_dim] [8 N 1 model_dim]
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.obs_len, agent_num, 1, self.model_dim])

        # [N N]
        src_agent_mask = agent_mask.clone()

        # [8 N 1 model_dim] -> [8 N model_dim]
        history_enc = self.tf_encoder(tf_in_pos, mask=src_agent_mask, num_agent=agent_num)
        history_rs = history_enc.view(-1, agent_num, self.model_dim)
        # compute per agent context [N model_dim]
        if self.pooling == 'mean':
            agent_history = torch.mean(history_rs, dim=0)  # [N model_dim]
        else:
            agent_history = torch.max(history_rs, dim=0)[0]

        return history_enc, agent_history


""" Future Encoder """


class FutureEncoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=4):
        super().__init__()

        self.pred_len = args.pred_len

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.fe_tf_layer

        self.pooling = args.pooling
        self.cross_motion_only = args.cross_motion_only

        self.in_dim = in_dim

        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        decoder_layers = TransformerDecoderLayer(self.model_dim, self.nhead, self.ff_dim,
                                                 self.dropout, cross_motion_only=self.cross_motion_only)
        self.tf_decoder = TransformerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, traj_in, history_enc, agent_mask, agent_enc_shuffle=None):

        agent_num = traj_in.shape[1]

        # [N*12 1 model_dim] [N*12 1 model_dim]
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.pred_len, agent_num, 1, self.model_dim])

        # [N N] [N N]
        mem_agent_mask = agent_mask.clone()
        tgt_agent_mask = agent_mask.clone()

        # [12 N 1 model_dim] -> [12 N model_dim]
        tf_out, _ = self.tf_decoder(tf_in_pos, history_enc, memory_mask=mem_agent_mask,
                                    tgt_mask=tgt_agent_mask, num_agent=agent_num)
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        # [N d_model-256]
        if self.pooling == 'mean':
            agent_future = torch.mean(tf_out, dim=0)
        else:
            agent_future = torch.max(tf_out, dim=0)[0]

        return agent_future


""" Future Decoder """


class FutureDecoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=2):
        super().__init__()

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.pred_dim = args.pred_dim

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.fd_tf_layer

        self.cross_motion_only = args.cross_motion_only

        self.in_dim = in_dim + args.nz  # args.nz
        self.out_mlp_dim = args.fd_out_mlp_dim

        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        decoder_layers = TransformerDecoderLayer(self.model_dim, self.nhead, self.ff_dim,
                                                 self.dropout, cross_motion_only=self.cross_motion_only)
        self.tf_decoder = TransformerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, dec_in, z, sample_num, agent_num, agent_mask,
                history_enc, agent_enc_shuffle=None, need_weights=False):

        z_in = z.unsqueeze(0).repeat_interleave(self.pred_len, dim=0)  # [N*sn 32] -> [12 N*sample_num 32]
        z_in = z_in.view(self.pred_len, agent_num, sample_num, z.shape[-1])  # [12 N sample_num 32]

        in_arr = [dec_in, z_in]

        # [12 N sample_num 2] + [12 N sample_num 32] -> [12*N sample_num 34]  34 -> 64
        dec_in_z = torch.cat(in_arr, dim=-1).reshape([agent_num * 12, sample_num, -1])

        # [N*12 sample_num model_dim] [N*12 sample_num model_dim] [12 N sample_num model_dim] 256
        tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        #                              t_offset=self.obs_len - 1 if self.pos_offset else 0)
        tf_in_pos = tf_in_pos.reshape([self.pred_len, agent_num, sample_num, self.model_dim])

        # [N N] [N N]
        mem_agent_mask = agent_mask.clone()
        tgt_agent_mask = agent_mask.clone()

        # [T N sample_num model_dim] []
        tf_out, attn_weights = self.tf_decoder(tf_in_pos, history_enc, memory_mask=mem_agent_mask, tgt_mask=tgt_agent_mask,
                                               seq_mask=True, num_agent=agent_num, need_weights=need_weights)

        return tf_out, attn_weights


""" CVAE """


class CVAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = torch.device('cpu')

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        # position encoding
        self.pos_enc = {
            'pos_concat': args.pos_concat,
            'max_agent_len': 128,  # 128
            'use_agent_enc': False,  # False
            'agent_enc_learn': False,  # False
            'agent_enc_shuffle': False  # False
        }
        self.max_train_agent = args.max_train_agent
        self.rand_rot_scene = args.rand_rot_scene
        self.discrete_rot = args.discrete_rot

        self.model_dim = args.tf_model_dim
        self.pred_dim = args.pred_dim

        self.compute_sample = True

        num_dist_params = 2 * args.nz

        # p_z_net for prior
        self.he_out_mlp_dim = args.he_out_mlp_dim
        if self.he_out_mlp_dim is None:
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.he_out_mlp = MLP(self.model_dim, args.he_out_mlp_dim, 'relu')
            self.p_z_net = nn.Linear(self.he_out_mlp.out_dim, num_dist_params)
        initialize_weights(self.p_z_net.modules())

        # q_z_net for posterior
        self.fe_out_mlp_dim = args.fe_out_mlp_dim
        if self.fe_out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.fe_out_mlp = MLP(self.model_dim, args.fe_out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.fe_out_mlp.out_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())

        # output function for prediction
        self.fd_out_mlp_dim = args.fd_out_mlp_dim
        if self.fd_out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, self.pred_dim)
        else:
            self.fd_out_mlp = MLP(self.model_dim, args.fd_out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.fd_out_mlp.out_dim, self.pred_dim)
        initialize_weights(self.out_fc.modules())

        # models
        self.history_encoder = HistoryEncoder(args, self.pos_enc)
        self.future_encoder = FutureEncoder(args, self.pos_enc)
        self.future_decoder = FutureDecoder(args, self.pos_enc)

        # TCN
        self.num_tcn_layers = args.num_tcn_layers
        self.tcn_layers = nn.ModuleList()
        self.tcn_layers.append(nn.Sequential(
            nn.Conv1d(self.obs_len, self.pred_len, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.num_tcn_layers):
            self.tcn_layers.append(nn.Sequential(
                nn.Conv1d(self.pred_len, self.pred_len, 3, padding=1),
                nn.PReLU()
            ))

        self.param_annealers = nn.ModuleList()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def set_data(self, pre_motion, fut_motion, pre_motion_mask, fut_motion_mask):
        device = self.device

        fut_motion_orig = fut_motion.transpose(1, 2)  # [N T 2] for loss calculation

        # reshape first [N 2 T] -> [T N 2]
        pre_motion = pre_motion.permute(2, 0, 1)
        fut_motion = fut_motion.permute(2, 0, 1)

        if self.training and pre_motion.shape[1] > self.max_train_agent:
            ind = np.random.choice(pre_motion.shape[1], self.max_train_agent).tolist()
            ind = torch.tensor(ind).to(device)

            pre_motion = torch.index_select(pre_motion, 1, ind).contiguous()
            pre_motion_mask = torch.index_select(pre_motion_mask, 0, ind).contiguous()  # [N T]
            fut_motion = torch.index_select(fut_motion, 1, ind).contiguous()
            fut_motion_mask = torch.index_select(fut_motion_mask, 0, ind).contiguous()  # [N T]
            fut_motion_orig = torch.index_select(fut_motion_orig, 0, ind).contiguous()

        self.agent_num = pre_motion.shape[1]

        self.scene_orig = pre_motion[[-1]].view(-1, 2).mean(dim=0)  # [T N 2] -> [1 N 2]

        # rotate the scene
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2  # true branch
            pre_motion, pre_motion_scene_norm = rotation_2d_torch(pre_motion, theta, self.scene_orig)
            fut_motion, fut_motion_scene_norm = rotation_2d_torch(fut_motion, theta, self.scene_orig)
            fut_motion_orig, fut_motion_orig_scene_norm = rotation_2d_torch(fut_motion_orig, theta, self.scene_orig)
        else:
            theta = torch.zeros(1).to(device)
            pre_motion_scene_norm = pre_motion - self.scene_orig
            fut_motion_scene_norm = fut_motion - self.scene_orig

        pre_vel = pre_motion[1:] - pre_motion[:-1, :]
        pre_vel = torch.cat([pre_vel[[0]], pre_vel], dim=0)
        fut_vel = fut_motion - torch.cat([pre_motion[[-1]], fut_motion[:-1, :]])

        cur_motion = pre_motion[[-1]][0]
        mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        agent_mask = mask  # [N N]

        # assign values
        self.agent_mask = agent_mask

        self.pre_motion_mask = pre_motion_mask
        self.fut_motion_mask = fut_motion_mask

        self.pre_motion = pre_motion
        self.pre_vel = pre_vel
        self.pre_motion_scene_norm = pre_motion_scene_norm

        self.fut_motion = fut_motion
        self.fut_vel = fut_vel
        self.fut_motion_scene_norm = fut_motion_scene_norm

        self.fut_motion_orig = fut_motion_orig

    def temporal_convolution(self, pre_motion_scene_norm):
        dec_in = pre_motion_scene_norm.transpose(0, 1)  # [T N 2] -> [N T 2]
        dec_in = self.tcn_layers[0](dec_in)
        for k in range(1, self.num_tcn_layers):
            dec_in = self.tcn_layers[k](dec_in) + dec_in
        dec_in = dec_in.transpose(0, 1).unsqueeze(2)  # [T N sn 2] (sn=1)
        return dec_in

    def encode_history(self):
        he_in = torch.cat([self.pre_motion_scene_norm, self.pre_vel], dim=-1)
        history_enc, agent_history = self.history_encoder(he_in, self.agent_mask)
        return history_enc, agent_history

    def encode_future(self, history_enc):
        fe_in = torch.cat([self.fut_motion_scene_norm, self.fut_vel], dim=-1)
        agent_future = self.future_encoder(fe_in, history_enc, self.agent_mask)
        return agent_future

    def decode_future(self, z, sample_num, history_enc, dec_in=None, need_weights=False):
        if dec_in is None:
            dec_in = self.temporal_convolution(self.pre_motion_scene_norm)
        else:
            assert len(dec_in.shape) == 4  # [T N 1 2]  1 -> sample_num

        # align and fed into future decoder (repeat according to sample num)
        assert z.shape[0] == self.agent_num * sample_num  # z: [N*sample_num 32]
        dec_in = dec_in.repeat_interleave(sample_num, dim=2)  # repeat dec_in according to sample_num
        history_enc = history_enc.repeat_interleave(sample_num, dim=2)  # repeat history_enc according to sample_num

        dec_motion, attn_weights = self.future_decoder(dec_in, z, sample_num, self.agent_num,
                                                       self.agent_mask, history_enc, need_weights=need_weights)

        # output function
        out_tmp = dec_motion.view(-1, dec_motion.shape[-1])  # [N*12*sample_num model_dim] for MLP
        if self.fd_out_mlp_dim is not None:  # [512, 256] for eth
            cat_arr = [out_tmp]
            out_tmp = torch.cat(cat_arr, dim=-1)
            out_tmp = self.fd_out_mlp(out_tmp)
        seq_out = self.out_fc(out_tmp).view(dec_motion.shape[0], -1, self.pred_dim)  # [12 N*sample_num 2]

        # reshape and add scene_orig
        seq_out = seq_out.view(-1, self.agent_num * sample_num, seq_out.shape[-1])  # [T N*sample_num 2]
        dec_motion = seq_out + self.scene_orig
        dec_motion = dec_motion.transpose(0, 1).contiguous()  # [N*sample_num T 2]
        if sample_num > 1:
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])  # [N sample_num T 2]
        return dec_motion, attn_weights

    def get_prior(self, agent_history, sample_num=1):
        if self.he_out_mlp_dim is not None:  # None #
            agent_history = self.he_out_mlp(agent_history)
        h = agent_history.repeat_interleave(sample_num, dim=0)
        p_z_params = self.p_z_net(h)
        p_z_dist = Normal(params=p_z_params)
        return p_z_dist

    def get_posterior(self, agent_future):
        if self.fe_out_mlp_dim is not None:  # out mlp, [512, 256] for eth
            agent_future = self.fe_out_mlp(agent_future)
        q_z_params = self.q_z_net(agent_future)
        q_z_dist = Normal(params=q_z_params)
        return q_z_dist

    def inference(self, sample_num, need_weights=False):
        history_enc, agent_history = self.encode_history()
        p_z_dist = self.get_prior(agent_history, sample_num=sample_num)
        z = p_z_dist.sample()
        dec_motion, attn_weights = self.decode_future(z, sample_num, history_enc, need_weights=need_weights)
        if sample_num == 1:
            dec_motion = dec_motion.unsqueeze(1)
        return dec_motion, attn_weights  # .reshape([self.pred_len, self.agent_num, sample_num, self.pred_dim])

    def forward(self, sample_num):

        history_enc, agent_history = self.encode_history()
        agent_future = self.encode_future(history_enc)

        q_z_dist = self.get_posterior(agent_future)
        q_z_sample = q_z_dist.rsample()

        dec_in = self.temporal_convolution(self.pre_motion_scene_norm)

        # prior distribution for KL divergence
        p_z_dist = self.get_prior(agent_history, sample_num=1)

        # future decoder input for recon loss, sample_num=1, z is from q
        z = q_z_sample  # [N*1 32]
        recon_motion, _ = self.decode_future(z, 1, history_enc, dec_in=dec_in)

        # future decoder input for variety loss, sample_num=20, z is from p
        p_z_dist_var = self.get_prior(agent_history, sample_num=sample_num)
        z_var = p_z_dist_var.sample()  # dec_in = dec_in.repeat_interleave(sample_num, dim=2)
        var_motion, _ = self.decode_future(z_var, sample_num, history_enc, dec_in=dec_in)

        return self.fut_motion_orig, recon_motion, var_motion, q_z_dist, p_z_dist, self.fut_motion_mask

