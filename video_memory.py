import numpy as np
import os
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
        idxs = np.random.randint(0, self.dataset['observations'].shape[0]-1, batch_size)

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

    def sample_gz(self, batch_size, self_prop=0.1, prop_same_goal_intent=0.1):
        idxs = np.random.randint(0, self.dataset['observations'].shape[0]-1, batch_size)

        goal_dists = np.where(np.random.rand(batch_size) > 0.5,
            np.random.choice(100, batch_size),
            100+np.random.geometric(1/100, batch_size)
        )
        goal_idxs = np.clip(idxs + 1 + goal_dists, 0, len(self.dataset['observations']) - 1)
        same_goal = np.random.rand(batch_size) < self_prop
        goal_idxs = np.where(same_goal, idxs, goal_idxs)

        intent_goal_dists = np.where(np.random.rand(batch_size) > 0.5,
            np.random.choice(100, batch_size),
            100+np.random.geometric(1/100, batch_size)
        )
        intent_goal_idxs = np.clip(goal_idxs + intent_goal_dists, 0, len(self.dataset['observations']) - 1)
        same_intent_goal = np.random.rand(batch_size) < prop_same_goal_intent
        intent_goal_idxs = np.where(same_intent_goal, goal_idxs, intent_goal_idxs)

        observations = self.dataset['observations'][idxs]
        next_observations = self.dataset['observations'][idxs + 1]
        goals = self.dataset['observations'][goal_idxs]
        intents = self.dataset['observations'][intent_goal_idxs]

        terminals = same_goal.astype(np.float32)
        nonterminals = 1 - terminals
        rewards = same_goal.astype(np.float32) - 1
        intent_rewards = (idxs == intent_goal_idxs).astype(np.float32) - 1        
        intent_nonterminals = 1 - (idxs == intent_goal_idxs).astype(np.float32)
        
        def process(o):
            if len(o.shape) > 2:
                torch_o = torch.from_numpy(o).to(self.device)
                torch_o = torch.moveaxis(torch_o, -1, 1) / 255.0
                return torch_o
            else:
                return torch.from_numpy(o).to(self.device)
        
        outputs = [process(o) for o in [idxs, observations, next_observations, goals, intents, rewards, intent_rewards, nonterminals, intent_nonterminals]]
        return outputs
        # torch_observations = torch.from_numpy(observations).to(self.device)
        # torch_next_observations = torch.from_numpy(next_observations).to(self.device)
        # torch_nonterminals = torch.from_numpy(nonterminals).to(self.device)

        # torch_observations = torch.moveaxis(torch_observations, -1, 1) / 255.0
        # torch_next_observations = torch.moveaxis(torch_next_observations, -1, 1) / 255.0


        # return idxs, torch_observations, torch_next_observations, torch_nonterminals
        # return idxs, states, next_states, goals, intents, rewards, nonterminals
