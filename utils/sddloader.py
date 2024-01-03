import os
import math
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset

class SDD_Dataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', traj_scale=1.0):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(SDD_Dataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip  # skip is 1
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        self.traj_scale = traj_scale

        all_files = os.listdir(self.data_dir)  # need to pick out
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        # for path in all_files:
        print()
        print('load data from following files:')
        print('=' * 30)

        with open(os.path.join(self.data_dir, all_files[0]), 'rb') as f:
            print(all_files[0])
            pec_data = pickle.load(f)

        print('=' * 30)

        self.pec_data = pec_data
        self.num_seq = len(pec_data)

        num_peds_in_seq = [traj_group.shape[0] for traj_group in pec_data]

        seq_list = np.concatenate(pec_data, axis=0) / traj_scale  # scale
        seq_list = seq_list.transpose(0, 2, 1)

        seq_list_rel = np.zeros(seq_list.shape)
        seq_list_rel[:, :, 1:] = seq_list[:, :, 1:] - seq_list[:, :, :-1]

        loss_mask_list = np.ones([seq_list.shape[0], seq_list.shape[2]])

        non_linear_ped = np.ones(np.sum(num_peds_in_seq))
        valid_ped = np.ones(np.sum(num_peds_in_seq))

        frame_idx = np.asarray([i + 1 for i in range(len(num_peds_in_seq))])

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.obs_loss_mask = torch.from_numpy(
            loss_mask_list[:, :self.obs_len]).type(torch.float)
        self.pred_loss_mask = torch.from_numpy(
            loss_mask_list[:, self.obs_len:]).type(torch.float)

        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.valid_ped = torch.from_numpy(valid_ped).type(torch.float)
        self.frame_idx = torch.from_numpy(frame_idx).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # need to check
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.valid_ped[start:end],
            self.obs_loss_mask[start:end, :], self.pred_loss_mask[start:end, :],
            self.frame_idx[index], 'sdd'  # add ped_id, seq_name
        ]

        return out
