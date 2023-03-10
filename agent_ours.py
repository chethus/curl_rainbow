# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import kornia.augmentation as aug
import torch.nn as nn
from model_ours import DQN

random_shift = nn.Sequential(aug.RandomCrop((80, 80)), nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
aug = random_shift

class Agent():
  def __init__(self, args, env):
    self.args = args
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

    self.online_net = DQN(args, self.action_space).to(device=args.device)

    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()
    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    self.video_optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      a, _ = self.online_net(state.unsqueeze(0))
      return (a * self.support).sum(2).argmax(1).item()

  # Acts with an ??-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ?? can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def video_loss(self, video_mem):
    idxs, states, next_states, goals, intents, rewards, intent_rewards, nonterminals, intent_nonterminals = video_mem.sample_gz(self.batch_size)
    aug_states = states # aug(states).to(device=self.args.device)
    aug_next_states = next_states #aug(next_states).to(device=self.args.device)
    aug_goals = goals #aug(goals).to(device=self.args.device)
        
    
    _, aux_s = self.online_net(aug_states, log=True)
    _, target_aux_ns = self.target_net(aug_next_states, log=True)
    _, aux_g = self.online_net(aug_goals, log=True)
    _, target_aux_g = self.target_net(aug_goals, log=True)

    og_phi = phi = aux_s['phi']
    psi = aux_g['psi']
    phi_target = target_aux_ns['phi']
    psi_target = target_aux_g['psi']

    if self.args.intent:
        _, target_aux_s = self.target_net(aug_states, log=True)
        aug_intents = intents #aug(intents).to(device=self.args.device)
        _, aux_z = self.online_net(aug_intents, log=True)
        _, target_aux_z = self.target_net(intents, log=True)
        phi = phi * aux_z['z']
        phi_target = phi_target * target_aux_z['z']
        
        psi = psi * aux_z['z']
        psi_target = psi_target * target_aux_z['z']

        phi_s_target = target_aux_s['phi'] * target_aux_z['z']
        psi_z_target = target_aux_z['psi'] * target_aux_z['z']
    

    v_sgz = -1 * torch.linalg.norm(phi - psi, dim=-1)
    v_nsgz = -1 * torch.linalg.norm(phi_target - psi_target, dim=-1)
    q_sgz = rewards + self.discount * nonterminals * v_nsgz

    if self.args.intent:
        v_szz = -1 * torch.linalg.norm(phi_s_target - psi_z_target, dim=-1)
        v_nszz = -1 * torch.linalg.norm(phi_target - psi_z_target, dim=-1)
        q_szz = intent_rewards + self.discount * intent_nonterminals * v_nszz

        adv = q_szz - v_szz

    else:
        v_szz = torch.zeros_like(v_sgz)
        q_szz = torch.zeros_like(q_sgz)
        adv = torch.zeros_like(v_sgz)

    def expectile_loss(adv, loss, expectile=0.5):
        weight = torch.where(adv >=0, expectile, 1 - expectile)
        return weight * loss
    def huber(x, delta=1.0):
        return torch.where(torch.abs(x) < delta, 0.5 * x ** 2, delta * (torch.abs(x) - 0.5 * delta))

    loss = huber(v_sgz - q_sgz)
    loss = expectile_loss(adv, loss, expectile=self.args.expectile)
    return loss.mean(), {
        'video_loss': loss.mean().item(),
        'v_sgz': v_sgz.mean().item(),
        'v_szz': v_szz.mean().item(),
        'q_sgz': q_sgz.mean().item(),
        'q_szz': q_szz.mean().item(),
        'adv_positive': (adv >= 0).float().mean().item(),
        'adv': adv.mean().item(),
        'adv max': adv.max().item(),
        'og_phi': torch.linalg.norm(og_phi, dim=-1).mean().item(),
        'phi': torch.linalg.norm(phi, dim=-1).mean().item(),
        'w_a std': self.online_net.fc_z_a.weight_sigma.mean().item(),
        'w_v std': self.online_net.fc_z_v.weight_sigma.mean().item(),
        'rewards': rewards.mean().item(),
        'median td_error': torch.median(q_sgz - v_sgz).item(),
        'mean td_error': torch.mean(q_sgz - v_sgz).item(),
    }
  def video_pretrain(self, video_mem):
    loss, info = self.video_loss(video_mem)
    self.online_net.zero_grad()
    loss.backward()
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.video_optimiser.step()
    return info

  def learn_with_video(self, mem, video_mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ??; ??online)
    video_loss, video_info = self.video_loss(video_mem)

    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; ??online)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ??; ??online)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ??; ??online))
      target_nq = dns.sum(2)
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; ??online))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ??; ??target)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; ??online))]; ??target)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (??^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / ??z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    td_loss = loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    original_loss = loss
    loss = loss + (video_loss * self.coeff)
    self.online_net.zero_grad()
    curl_loss = (weights * loss).mean()
    curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, original_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    return {
      'loss': curl_loss.mean().item(),
      'weight': weights.mean().item(),
      'max weight': weights.max().item(),
      'td_loss': (weights * td_loss).mean().item(),
      'true_rewards': returns.mean().item(),
      'true_nonterminals': nonterminals.mean().item(),
      'target_nq': target_nq.mean().item(),
      'target_nv': target_nq.max(1).values.mean().item(),
      'nq_std': target_nq.std(1).mean().item(),
      **video_info,
    }

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    aug_states_1 = aug(states).to(device=self.args.device)
    aug_states_2 = aug(states).to(device=self.args.device)
    # Calculate current state probabilities (online network noise already sampled)
    log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ??; ??online)
    _, z_anch = self.online_net(aug_states_1, log=True)
    _, z_target = self.momentum_net(aug_states_2, log=True)
    z_proj = torch.matmul(self.online_net.W, z_target.T)
    logits = torch.matmul(z_anch, z_proj)
    logits = (logits - torch.max(logits, 1)[0][:, None])
    logits = logits * 0.1
    labels = torch.arange(logits.shape[0]).long().to(device=self.args.device)
    moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.device)

    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; ??online)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ??; ??online)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ??; ??online))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; ??online))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ??; ??target)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; ??online))]; ??target)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (??^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / ??z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    td_loss = loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    loss = loss + (moco_loss * self.coeff)
    self.online_net.zero_grad()
    curl_loss = (weights * loss).mean()
    curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
    return {
      'loss': curl_loss.mean().item(),
      'moco_loss': (weights * moco_loss).mean().item(),
      'weight': weights.mean().item(),
      'td_loss': (weights * td_loss).mean().item(),

    }

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def initialize_momentum_net(self):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(param_q.data) # update
      param_k.requires_grad = False  # not update by gradient

  # Code for this function from https://github.com/facebookresearch/moco
  @torch.no_grad()
  def update_momentum_net(self, momentum=0.999):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(momentum * param_k.data + (1.- momentum) * param_q.data) # update

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      a, _ = self.online_net(state.unsqueeze(0))
      return (a * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
