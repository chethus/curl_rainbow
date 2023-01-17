import numpy as np
import os
import gym
import gzip

import torch

from os.path import expanduser
BASE_VIDEO_DIR = os.environ.get('ATARI_VIDEODATASET_DIR', '/nfs/kun2/users/chet/video/atari/preprocessing/datasets')

def get_video_dataset(game):
    path = os.path.join(BASE_VIDEO_DIR, f'{game.lower()}.npz')
    print('loading {}...'.format(path))
    data = np.load(path)
    observations = data['observations'].astype(np.uint8)
    terminals = data['terminals']
    observations = np.lib.stride_tricks.sliding_window_view(observations, 4, 0)

    data_dict = {
        'observations': observations,
        'terminals': terminals,
    }
    return data_dict

class VideoReplayMemory:
    def __init__(self, game, args):
        self.dataset = get_video_dataset(game)
        self.device = args.device
    
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.dataset['observations'].shape[0], batch_size)

        observations = self.dataset['observations'][idxs]
        next_observations = self.dataset['observations'][idxs + 1]
        terminals = self.dataset['terminals'][idxs+3]
        nonterminals = 1 - terminals
        
        torch_observations = torch.from_numpy(observations).to(self.device)
        torch_next_observations = torch.from_numpy(next_observations).to(self.device)
        torch_nonterminals = torch.from_numpy(nonterminals).to(self.device)

        torch_observations = torch.moveaxis(torch_observations, -1, 1) / 255.0
        torch_next_observations = torch.moveaxis(torch_next_observations, -1, 1) / 255.0

        return idxs, torch_observations, torch_next_observations, torch_nonterminals