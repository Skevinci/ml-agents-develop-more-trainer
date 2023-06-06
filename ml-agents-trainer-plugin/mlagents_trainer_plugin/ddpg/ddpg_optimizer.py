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


@attr.s(auto_attribs=True)
class DDPGSettings(OffPolicyHyperparamSettings):
    tau: float = 0.005
    reward_signal_steps_per_update: float = attr.ib()

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
