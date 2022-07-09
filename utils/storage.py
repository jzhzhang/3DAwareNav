# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, map_obs_shape, points_entropy_obs_shape, points_goal_obs_shape, action_space,
                 rec_state_size):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        self.obs_map = torch.zeros(num_steps + 1, num_processes, *map_obs_shape)
        self.obs_entropy_points = torch.zeros(num_steps + 1, num_processes, *points_entropy_obs_shape)
        self.obs_goal_points = torch.zeros(num_steps + 1, num_processes, *points_goal_obs_shape)


        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs_map = self.obs_map.to(device)

        self.obs_entropy_points = self.obs_entropy_points.to(device)
        self.obs_goal_points = self.obs_goal_points.to(device)


        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs_map, obs_entropy_points, obs_goal_points, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        self.obs_map[self.step + 1].copy_(obs_map)

        self.obs_entropy_points[self.step + 1].copy_(obs_entropy_points)
        self.obs_goal_points[self.step + 1].copy_(obs_goal_points)

        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_map[0].copy_(self.obs_map[-1])

        self.obs_entropy_points[0].copy_(self.obs_entropy_points[-1])
        self.obs_goal_points[0].copy_(self.obs_goal_points[-1])

        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs_map': self.obs_map[:-1].view(-1, *self.obs_map.size()[2:])[indices],

                'obs_entropy_points': self.obs_entropy_points[:-1].view(-1, *self.obs_entropy_points.size()[2:])[indices],
                'obs_goal_points': self.obs_goal_points[:-1].view(-1, *self.obs_goal_points.size()[2:])[indices],

                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs_map = []
            # obs_points = []
            obs_entropy_points = []
            obs_goal_points = []

            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs_map.append(self.obs_map[:-1, ind])
                # obs_points.append(self.obs_points[:-1, ind])

                obs_entropy_points.append(self.obs_entropy_points[:-1, ind])
                obs_goal_points.append(self.obs_goal_points[:-1, ind])


                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs_map = torch.stack(obs_map, 1)
            # obs_points = torch.stack(obs_points, 1)

            obs_entropy_points = torch.stack(obs_entropy_points, 1)
            obs_goal_points = torch.stack(obs_goal_points, 1)


            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs_map': _flatten_helper(T, N, obs_map),
                # 'obs_points': _flatten_helper(T, N, obs_points),

                'obs_entropy_points': _flatten_helper(T, N, obs_entropy_points),
                'obs_goal_points': _flatten_helper(T, N, obs_goal_points),

                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_map_shape, obs_entropy_points_shape, obs_goal_points_shape, action_space,
                 rec_state_size, extras_size):
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_map_shape, obs_entropy_points_shape, obs_goal_points_shape, action_space, rec_state_size)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size

    def insert(self, obs_map, obs_entropy_points, obs_goal_points, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(
            obs_map, obs_entropy_points, obs_goal_points, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)
