from typing import Dict, List, Union, Tuple, Any, Optional
import attr

from mlagents.trainers.torch_utils import torch, nn, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OffPolicyHyperparamSettings,
    ScheduleType,
    NetworkSettings,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork, Actor
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.base_env import ActionSpec, ObservationSpec

from mlagents.trainers.torch_entities.networks import Critic
import numpy as np
from copy import deepcopy
import torch.nn.functional as F


@attr.s(auto_attribs=True)
class DDPGSettings(OffPolicyHyperparamSettings):
    tau: float = 0.005
    reward_signal_steps_per_update: float = attr.ib()
    shared_critic: bool = False
    gamma: float = 0.99

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class DDPGOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)

        params = list(self.policy.actor.parameters())
        self.hyperparameters: DDPGSettings = cast(
            DDPGSettings, trainer_settings.hyperparameters
        )
        self.tau = self.hyperparameters.tau
        self.batch_size = self.hyperparameters.batch_size

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        # Actor Network
        self.actor_optimizer = torch.optim.Adam(
            params,
            lr=self.trainer_settings.hyperparameters.learning_rate
        )

        # Critic Network
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                list(self.reward_signals.keys()),
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
        self._critic.to(default_device())
        params += list(self._critic.parameters())

        # Target Networks
        self.actor_target = deepcopy(self.policy.actor)
        self.critic_target = deepcopy(self._critic)

        # Initializing target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.policy.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        actions = AgentAction.from_buffer(batch)

        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        # Critic updation
        next_actions = self.actor_target(next_obs)
        Q_targets_next = self.critic_target(next_obs, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(current_obs, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Actor updation
        actions_pred = self.policy.actor(current_obs)
        actor_loss = -self.critic(current_obs, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.policy.actor, self.actor_target, self.tau)

        return {
            "Losses/Value Loss": critic_loss.item(),
            "Losses/Policy Loss": actor_loss.item(),
        }

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
