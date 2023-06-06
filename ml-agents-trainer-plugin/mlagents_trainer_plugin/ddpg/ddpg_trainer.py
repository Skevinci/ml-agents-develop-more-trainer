"""Written by Arthur Sikai Li, based on other trainers by Unity Technologies."""
from typing import cast

import numpy as np
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings
from .ddpg_optimizer import DDPGOptimizer, DDPGSettings

logger = get_logger(__name__)
TRAINER_NAME = "ddpg"


class DDPGTrainer(OffPolicyTrainer):
    """The DDPGTrainer is an implementation of the DDPG algorithm."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training DDPG model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.policy: TorchPolicy = None  # type: ignore
        self.optimizer: TorchOptimizer = None  # type: ignore

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Evaluate all reward functions for reporting purposes
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )

            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        self._append_to_update_buffer(agent_buffer_trajectory)

        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        """
        Creates an Optimizer object
        """
        return DDPGOptimizer(
            cast(TorchPolicy, self.policy), self.trainer_settings
        )

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and DDPG hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            parsed_behavior_id=parsed_behavior_id,
        )
        self.maybe_load_replay_buffer()
        return policy

    @staticmethod
    def get_settings_type():
        return DDPGSettings

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME


def get_type_and_setting():
    return {DDPGTrainer.get_trainer_name(): DDPGTrainer}, {
        DDPGTrainer.get_trainer_name(): DDPGTrainer.get_settings_type()
    }
