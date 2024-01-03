import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import TrajectoryDataset
from model.CVAE import CVAE
from model.vaeloss import compute_vae_loss
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter

# maybe need to close
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--sdd_scale', type=float, default=50.0)

# model architecture
parser.add_argument('--pos_concat', type=bool, default=True)
parser.add_argument('--cross_motion_only', type=bool, default=True)

parser.add_argument('--tf_model_dim', type=int, default=256)
parser.add_argument('--tf_ff_dim', type=int, default=512)
parser.add_argument('--tf_nhead', type=int, default=8)
parser.add_argument('--tf_dropout', type=float, default=0.1)

parser.add_argument('--he_tf_layer', type=int, default=2)  # he = history encoder
parser.add_argument('--fe_tf_layer', type=int, default=2)  # fe = future encoder
parser.add_argument('--fd_tf_layer', type=int, default=2)  # fd = future decoder

# parser.add_argument('--cross_range', type=int, default=2)
# parser.add_argument('--num_conv_layer', type=int, default=7)

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=3)
parser.add_argument('--asconv_layer_num', type=int, default=3)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--sample_k', type=int, default=20)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)

# loss config
parser.add_argument('--mse_weight', type=float, default=1.0)
parser.add_argument('--kld_weight', type=float, default=1.0)
parser.add_argument('--kld_min_clamp', type=float, default=2.0)
parser.add_argument('--var_weight', type=float, default=1.0)
parser.add_argument('--var_k', type=int, default=20)

# training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default='step')

parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--lr_fix_epochs', type=int, default=10)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=20)


def print_log(dataset, epoch, total_epoch, index, total_samples, seq_name, frame, loss_str):
    # form a string and adjust format
    print_str = '{} | Epo: {:02d}/{:02d}, It: {:04d}/{:04d}, seq: {:s}, frame {:05d}, {}' \
        .format(dataset + ' vae', epoch, total_epoch, index, total_samples, str(seq_name), int(frame), loss_str)
    print(print_str)



def train(args, epoch, model, optimizer, scheduler, loader_train):
    train_loss_meter = {'mse': AverageMeter(), 'kld': AverageMeter(),
                        'sample': AverageMeter(), 'total_loss': AverageMeter()}
    data_index = 0
    for cnt, batch in enumerate(loader_train):
        seq_name = batch.pop()[0]
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

        model.set_data(obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)               # [T N or N*sn 2]
        fut_motion_orig, train_dec_motion, infer_dec_motion, q_z_dist, p_z_dist, fut_mask = model.forward(args.sample_k)

        optimizer.zero_grad()
        total_loss, loss_dict, loss_dict_uw = compute_vae_loss(args, fut_motion_orig, train_dec_motion,
                                                               infer_dec_motion, fut_mask, q_z_dist, p_z_dist)
        total_loss.backward()  # total loss is weighted
        optimizer.step()

        # save loss
        train_loss_meter['total_loss'].update(total_loss.item())
        for key in loss_dict_uw.keys():
            train_loss_meter[key].update(loss_dict_uw[key])  # printed loss item from loss_dict_uw

        # print loss
        if cnt - data_index == args.print_freq:
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            print_log(args.dataset, epoch, args.num_epochs, cnt, len(loader_train), seq_name, frame_idx, losses_str)
            data_index = cnt

    scheduler.step()
    model.step_annealer()


def main(args):

    data_set = './dataset/' + args.dataset + '/'

    prepare_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    traj_scale = 1.0
    if args.dataset == 'eth':
        args.max_train_agent = 32
        # traj_scale = 2.0
        # args.fe_out_mlp_dim = [512, 256]
        # args.fd_out_mlp_dim = [512, 256]

    if args.dataset == 'sdd':
        traj_scale = args.sdd_scale
        dset_train = SDD_Dataset(
            data_set + 'train/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)
    else:
        dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    ''' === set model === '''
    cvae = CVAE(args)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)
    scheduler_type = args.scheduler
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=args.lr_fix_epochs, nepoch=args.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=args.decay_step, decay_gamma=args.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    checkpoint_dir = './checkpoints/' + args.dataset + '/vae/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    cvae.set_device(device)
    cvae.train()
    for epoch in range(args.num_epochs):
        train(args, epoch, cvae, optimizer, scheduler, loader_train)
        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            cp_path = os.path.join(checkpoint_dir, 'model_%04d.p') % (epoch + 1)
            model_cp = cvae.state_dict()
            torch.save(model_cp, cp_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
