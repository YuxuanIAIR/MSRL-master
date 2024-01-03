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
from model.sampler import Sampler
from model.samplerloss import compute_sampler_loss
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter

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

# sampler architecture
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

# loss config
parser.add_argument('--kld_weight', type=float, default=0.1)
parser.add_argument('--kld_min_clamp', type=float, default=10)
parser.add_argument('--recon_weight', type=float, default=5.0)

# training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default='step')

parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--lr_fix_epochs', type=int, default=10)
parser.add_argument('--decay_step', type=int, default=5)
parser.add_argument('--decay_gamma', type=float, default=0.5)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=20)

parser.add_argument('--vae_epoch', type=int, default=70)


# assign diversity loss config
def get_diversity_config(dataset):
    if dataset == 'sdd':
        weight, scale = 5, 5.0 * 2
    elif dataset == 'eth':
        weight, scale = 20, 10
    elif dataset == 'univ':
        weight, scale = 10, 10
    else:
        weight, scale = 5, 5.0
    return {'weight': weight, 'scale': scale}


def print_log(dataset, epoch, total_epoch, index, total_samples, seq_name, frame, loss_str):
    # form a string and adjust format
    print_str = '{} | Epo: {:02d}/{:02d}, It: {:04d}/{:04d}, seq: {:s}, frame {:05d}, {}' \
        .format(dataset + ' sampler', epoch, total_epoch, index, total_samples, str(seq_name), int(frame), loss_str)
    print(print_str)


def train(args, epoch, cvae, sampler, optimizer, scheduler, loader_train, div_cfg):
    train_loss_meter = {'kld': AverageMeter(), 'diverse': AverageMeter(),
                        'recon': AverageMeter(), 'total_loss': AverageMeter()}
    data_index = 0
    for cnt, batch in enumerate(loader_train):
        seq_name = batch.pop()[0]
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask, frame_idx = batch

        cvae.set_data(obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)
        dec_motion, sampler_dist, vae_dist, _ = sampler.forward(cvae)  # [T N sn 2]

        fut_motion_orig = pred_traj_gt.transpose(1, 2)  # [N 2 T] -> [N T 2]

        optimizer.zero_grad()
        total_loss, loss_dict, loss_dict_uw = compute_sampler_loss(args, fut_motion_orig, dec_motion, pred_loss_mask,
                                                                   vae_dist, sampler_dist, div_cfg)

        total_loss.backward()
        optimizer.step()

        # save loss
        train_loss_meter['total_loss'].update(total_loss.item())
        for key in loss_dict_uw.keys():
            train_loss_meter[key].update(loss_dict_uw[key])

        # print loss
        if cnt - data_index == args.print_freq:
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            print_log(args.dataset, epoch, args.num_epochs, cnt, len(loader_train), seq_name, frame_idx, losses_str)
            data_index = cnt

    scheduler.step()
    sampler.step_annealer()


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
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)

    ''' === set model === '''

    cvae = CVAE(args)  # load CVAE

    # load cvae model
    vae_dir = './checkpoints/' + args.dataset + '/vae/'
    all_vae_models = os.listdir(vae_dir)
    if len(all_vae_models) == 0:
        print('VAE model not found!')
        return

    default_vae_model = 'model_%04d.p' % args.vae_epoch
    if default_vae_model not in all_vae_models:
        default_vae_model = all_vae_models[-1]
    cp_path = os.path.join(vae_dir, default_vae_model)
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = torch.load(cp_path, map_location='cpu')
    cvae.load_state_dict(model_cp)

    sampler = Sampler(args)
    optimizer = optim.Adam(sampler.parameters(), lr=args.lr)
    scheduler_type = args.scheduler
    if scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=args.lr_fix_epochs, nepoch=args.num_epochs)
    elif scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=args.decay_step, decay_gamma=args.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    checkpoint_dir = './checkpoints/' + args.dataset + '/sampler/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    div_cfg = get_diversity_config(args.dataset)

    cvae.set_device(device)
    cvae.eval()

    sampler.set_device(device)
    sampler.train()

    for epoch in range(args.num_epochs):
        train(args, epoch, cvae, sampler, optimizer, scheduler, loader_train, div_cfg)
        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            cp_path = os.path.join(checkpoint_dir, 'model_%04d.p') % (epoch + 1)  # need to add epoch num
            model_cp = sampler.state_dict()
            torch.save(model_cp, cp_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

