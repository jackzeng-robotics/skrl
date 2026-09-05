from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .mappo_cfg import MAPPO_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param truncated: Signals to indicate that episodes have been truncated.
    :param values: Values obtained by the agent.
    :param last_values: Last values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.
    :param time_limit_bootstrap: Whether to use time-limit (truncation) bootstrapping.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class MAPPO(MultiAgent):
    def __init__(
        self,
        *,
        possible_agents: list[str],
        models: dict[str, dict[str, Model]],
        memories: dict[str, Memory] | None = None,
        observation_spaces: dict[str, gymnasium.Space] | None = None,
        state_spaces: dict[str, gymnasium.Space] | None = None,
        action_spaces: dict[str, gymnasium.Space] | None = None,
        device: str | torch.device | None = None,
        cfg: MAPPO_CFG | dict = {},
    ) -> None:
        """Multi-Agent Proximal Policy Optimization (MAPPO).

        https://arxiv.org/abs/2103.01955

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: MAPPO_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=MAPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        # checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        # broadcast models' parameters in distributed runs
        for uid in self.possible_agents:
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # validate parameter sharing: every agent must map to the same model instances
        if self.cfg.shared_across_agents:
            reference = self.possible_agents[0]
            for uid in self.possible_agents[1:]:
                if self.policies[uid] is not self.policies[reference] or self.values[uid] is not self.values[reference]:
                    raise ValueError(
                        "'shared_across_agents' is enabled but the agents do not share the same model instances. "
                        "Set 'models.shared_across_agents' in the runner configuration, or pass the same model "
                        "instances for every agent."
                    )

        # set up optimizer and learning rate scheduler
        self.optimizers = {}
        self.schedulers = {}
        # separate value optimizers/schedulers (None unless cfg.separate_optimizers and policy is not value)
        self.value_optimizers = {uid: None for uid in self.possible_agents}
        self.value_schedulers = {uid: None for uid in self.possible_agents}
        for uid in self.possible_agents:
            if self.policies[uid] is not None and self.values[uid] is not None:
                separate = self.cfg.separate_optimizers and self.policies[uid] is not self.values[uid]
                # - optimizers
                if self.policies[uid] is self.values[uid]:
                    self.optimizers[uid] = torch.optim.Adam(
                        self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                    )
                elif separate:
                    self.optimizers[uid] = torch.optim.Adam(
                        self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                    )
                    self.value_optimizers[uid] = torch.optim.Adam(
                        self.values[uid].parameters(), lr=self.cfg.learning_rate[uid][1]
                    )
                    self.checkpoint_modules[uid]["value_optimizer"] = self.value_optimizers[uid]
                else:
                    self.optimizers[uid] = torch.optim.Adam(
                        itertools.chain(self.policies[uid].parameters(), self.values[uid].parameters()),
                        lr=self.cfg.learning_rate[uid][0],
                    )
                self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]
                # - learning rate schedulers
                self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                if self.schedulers[uid] is not None:
                    self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                        self.optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                    )
                if separate and self.cfg.learning_rate_scheduler[uid][1] is not None:
                    self.value_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][1](
                        self.value_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][1]
                    )

        # set up preprocessors
        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        self._value_preprocessor = {}
        for uid in self.possible_agents:
            # - observations
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = self.cfg.observation_preprocessor[uid](
                    **self.cfg.observation_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = self._observation_preprocessor[uid]
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor
            # - states
            if self.cfg.state_preprocessor[uid]:
                self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                    **self.cfg.state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor
            # - values
            if self.cfg.value_preprocessor[uid]:
                self._value_preprocessor[uid] = self.cfg.value_preprocessor[uid](
                    **self.cfg.value_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="observations", size=self.observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = {}
        self._current_next_states = {}
        self._current_log_prob = {}
        self._current_values = {}
        self._rollout = 0

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        actions = {}
        log_prob = {}
        outputs = {}
        current_values = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # sample random actions
            # TODO, check for stochasticity
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")

            # sample stochastic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]

                # compute values
                if self.training:
                    values, _ = self.values[uid].act(inputs, role="value")
                    current_values[uid] = self._value_preprocessor[uid](values, inverse=True)

        self._current_log_prob = log_prob
        self._current_values = current_values
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        actions: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
        next_observations: dict[str, torch.Tensor],
        next_states: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        truncated: dict[str, torch.Tensor],
        infos: dict[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                # time-limit (truncation) bootstrapping
                if self.cfg.time_limit_bootstrap[uid] and truncated[uid].any():
                    with torch.no_grad():
                        inputs = {
                            "observations": self._observation_preprocessor[uid](next_observations[uid]),
                            "states": self._state_preprocessor[uid](next_states[uid]),
                        }
                        next_values, _ = self.values[uid].act(inputs, role="value")
                        next_values = self._value_preprocessor[uid](next_values, inverse=True)

                    rewards[uid] += self.cfg.discount_factor[uid] * next_values * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=self._current_values[uid],
                )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    if self.cfg.shared_across_agents:
                        # parameter sharing: pool every agent's transitions and update the shared
                        # parameters once, rather than once per agent
                        self.update_shared(timestep=timestep, timesteps=timesteps)
                    else:
                        for uid in self.possible_agents:
                            self.update(timestep=timestep, timesteps=timesteps, uid=uid)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        :param uid: Agent ID.
        """
        self._compute_returns_and_advantages(uid=uid)
        self._optimize(uid=uid, uids=[uid])

    def update_shared(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's update step for models shared across agents (parameter sharing).

        Returns and advantages are computed per agent (each agent has its own memory and its own
        value preprocessor statistics), then every agent's transitions are pooled into a single
        batch so the shared parameters take one optimization step per mini-batch.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        for uid in self.possible_agents:
            self._compute_returns_and_advantages(uid=uid)
        self._optimize(uid=self.possible_agents[0], uids=self.possible_agents)

    def _compute_returns_and_advantages(self, *, uid: str) -> None:
        """Bootstrap the last value and write returns/advantages into the agent's memory.

        :param uid: Agent ID.
        """
        value = self.values[uid]
        memory = self.memories[uid]

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=memory.get_tensor_by_name("terminated"),
            truncated=memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.gae_lambda[uid],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[uid],
        )

        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

    def _sample_mini_batches(self, *, uid: str, uids: list[str]) -> list[list[torch.Tensor]]:
        """Sample mini-batches, pooling across ``uids`` when more than one agent is given.

        :param uid: Agent ID whose mini-batch count is used.
        :param uids: Agent IDs whose memories are pooled.

        :return: Mini-batches of tensors ordered as :attr:`_tensors_names`.
        """
        mini_batches = self.cfg.mini_batches[uid]
        if len(uids) == 1:
            return self.memories[uid].sample(
                names=self._tensors_names, batch_size=len(self.memories[uid]), mini_batches=mini_batches
            )
        # pool agents: one concatenation per (mini-batch, tensor) rather than an incremental
        # accumulate-in-a-loop, which would copy O(len(uids)^2) data
        per_agent = [
            self.memories[u].sample(
                names=self._tensors_names, batch_size=len(self.memories[u]), mini_batches=mini_batches
            )
            for u in uids
        ]
        return [
            [torch.cat([batches[i][t] for batches in per_agent], dim=0) for t in range(len(self._tensors_names))]
            for i in range(mini_batches)
        ]

    def _optimize(self, *, uid: str, uids: list[str]) -> None:
        """Run the learning epochs for one set of shared models.

        :param uid: Agent ID owning the models, optimizers, preprocessors and hyperparameters.
        :param uids: Agent IDs whose memories feed the update (``[uid]`` unless parameter sharing).
        """
        policy = self.policies[uid]
        value = self.values[uid]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            # mini-batches loop (re-sampled every epoch for per-epoch shuffling)
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self._sample_mini_batches(uid=uid, uids=uids):

                # advantage filtering: keep the largest-magnitude advantages
                if self.cfg.advantage_filter_ratio[uid] > 0:
                    keep = int(sampled_advantages.shape[0] * (1.0 - self.cfg.advantage_filter_ratio[uid]))
                    if keep > 0:
                        index = torch.topk(sampled_advantages.abs().flatten(), keep, sorted=False).indices
                        (
                            sampled_observations,
                            sampled_states,
                            sampled_actions,
                            sampled_log_prob,
                            sampled_values,
                            sampled_returns,
                            sampled_advantages,
                        ) = (
                            sampled_observations[index],
                            sampled_states[index],
                            sampled_actions[index],
                            sampled_log_prob[index],
                            sampled_values[index],
                            sampled_returns[index],
                            sampled_advantages[index],
                        )

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }

                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = value.act(inputs, role="value")

                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                value_optimizer = self.value_optimizers[uid]
                self.optimizers[uid].zero_grad()
                if value_optimizer is not None:
                    value_optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    elif value_optimizer is not None:
                        self.scaler.unscale_(value_optimizer)
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                        nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[uid])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()), self.cfg.grad_norm_clip[uid]
                        )

                self.scaler.step(self.optimizers[uid])
                if value_optimizer is not None:
                    self.scaler.step(value_optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            kl = None
            for scheduler in (self.schedulers[uid], self.value_schedulers[uid]):
                if not scheduler:
                    continue
                if isinstance(scheduler, KLAdaptiveLR):
                    if kl is None:
                        kl = torch.tensor(kl_divergences, device=self.device).mean()
                        # reduce (collect from all workers/processes) KL in distributed runs
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                    scheduler.step(kl.item())
                else:
                    scheduler.step()

        # record data
        self.track_data(
            f"Loss / Policy loss ({uid})",
            cumulative_policy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        self.track_data(
            f"Loss / Value loss ({uid})",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(
                f"Loss / Entropy loss ({uid})",
                cumulative_entropy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
            )

        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())

        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
        if self.value_schedulers[uid]:
            self.track_data(f"Learning / Value learning rate ({uid})", self.value_schedulers[uid].get_last_lr()[0])
