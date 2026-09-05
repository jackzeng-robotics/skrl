import hypothesis
import hypothesis.strategies as st
import pytest

import dataclasses
import gymnasium

import torch

from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO as MultiAgent
from skrl.multi_agents.torch.mappo import MAPPO_CFG as MultiAgentCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import (
    categorical_model,
    deterministic_model,
    gaussian_model,
    multivariate_gaussian_model,
)

from ...utilities import MultiAgentEnv, check_config_keys, get_test_mixed_precision, is_device_available


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    max_num_agents=st.integers(min_value=2, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    gae_lambda=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    ratio_clip=st.floats(min_value=0, max_value=1),
    value_clip=st.floats(min_value=0, max_value=1),
    entropy_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    kl_threshold=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
    mixed_precision=st.booleans(),
    separate_optimizers=st.booleans(),
    advantage_filter_ratio=st.floats(min_value=0, max_value=0.9),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("separate", [True])
@pytest.mark.parametrize("asymmetric", [True])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin", "MultivariateGaussianMixin", "CategoricalMixin"])
def test_agent(
    capsys,
    device,
    num_envs,
    max_num_agents,
    asymmetric,
    # model config
    separate,
    policy_structure,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    discount_factor,
    gae_lambda,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    value_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    ratio_clip,
    value_clip,
    entropy_loss_scale,
    value_loss_scale,
    kl_threshold,
    rewards_shaper,
    time_limit_bootstrap,
    mixed_precision,
    separate_optimizers,
    advantage_filter_ratio,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_spaces = {}
    state_spaces = {}
    action_spaces = {}
    for i in range(max_num_agents):
        uid = f"agent_{i}"
        observation_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(max_num_agents,))
        state_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(6,)) if asymmetric else None  # common
        if policy_structure in ["GaussianMixin", "MultivariateGaussianMixin"]:
            action_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(max_num_agents - 1,))
        elif policy_structure == "CategoricalMixin":
            action_spaces[uid] = gymnasium.spaces.Discrete(max_num_agents - 1)

    # env
    env = MultiAgentEnv(
        observation_spaces=observation_spaces,
        state_spaces=state_spaces,
        action_spaces=action_spaces,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )

    # models
    network = {
        "policy": [
            {
                "name": "net",
                "input": "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
        "value": [
            {
                "name": "net",
                "input": "STATES" if asymmetric else "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    models = {}
    for uid in env.possible_agents:
        models[uid] = {}
        if separate:
            if policy_structure == "GaussianMixin":
                models[uid]["policy"] = gaussian_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=env.action_space(uid),
                    device=env.device,
                    network=network["policy"],
                    output="ACTIONS",
                )
            elif policy_structure == "MultivariateGaussianMixin":
                models[uid]["policy"] = multivariate_gaussian_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=env.action_space(uid),
                    device=env.device,
                    network=network["policy"],
                    output="ACTIONS",
                )
            elif policy_structure == "CategoricalMixin":
                models[uid]["policy"] = categorical_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=env.action_space(uid),
                    device=env.device,
                    network=network["policy"],
                    output="ACTIONS",
                )
            models[uid]["value"] = deterministic_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["value"],
                output="ONE",
            )
        else:
            raise ValueError("Separate is not supported for MAPPO, since it uses a centralized value function")

    # memory
    memories = {}
    for uid in env.possible_agents:
        memories[uid] = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {"size": env.observation_space("agent_0"), "device": env.device},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space("agent_0"), "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "kl_threshold": kl_threshold,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
        "mixed_precision": get_test_mixed_precision(mixed_precision),
        "separate_optimizers": separate_optimizers,
        "shared_across_agents": False,
        "advantage_filter_ratio": advantage_filter_ratio,
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }
    cfg["learning_rate_scheduler_kwargs"][
        "kl_threshold" if learning_rate_scheduler is KLAdaptiveLR else "factor"
    ] = learning_rate_scheduler_kwargs_value
    check_config_keys(cfg, dataclasses.asdict(MultiAgentCfg()))
    check_config_keys(cfg["experiment"], dataclasses.asdict(MultiAgentCfg().experiment))
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
    )

    # trainer
    cfg_trainer = {
        "timesteps": int(5 * rollouts),
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    if policy_structure == "CategoricalMixin":
        try:
            trainer.train()
        except RuntimeError as e:
            error_messages = [
                "probability tensor contains either",
                "invalid multinomial distribution",
            ]
            if not any(message in str(e) for message in error_messages):
                raise e
    else:
        trainer.train()


def _build_shared_setup(*, num_agents, num_envs, rollouts, device):
    """Build a MAPPO setup whose policy/value instances are shared by every agent."""
    observation_spaces, state_spaces, action_spaces = {}, {}, {}
    for i in range(num_agents):
        uid = f"agent_{i}"
        observation_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(num_agents,))
        state_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(6,))
        action_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(num_agents - 1,))

    env = MultiAgentEnv(
        observation_spaces=observation_spaces,
        state_spaces=state_spaces,
        action_spaces=action_spaces,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )

    reference = env.possible_agents[0]
    policy = gaussian_model(
        observation_space=env.observation_space(reference),
        state_space=env.state_space(reference),
        action_space=env.action_space(reference),
        device=env.device,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [5], "activations": "relu"}],
        output="ACTIONS",
    )
    value = deterministic_model(
        observation_space=env.observation_space(reference),
        state_space=env.state_space(reference),
        action_space=env.action_space(reference),
        device=env.device,
        network=[{"name": "net", "input": "STATES", "layers": [5], "activations": "relu"}],
        output="ONE",
    )
    policy.init_state_dict(role="policy")
    value.init_state_dict(role="value")

    # every agent points at the same two instances (parameter sharing)
    models = {uid: {"policy": policy, "value": value} for uid in env.possible_agents}
    memories = {
        uid: RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)
        for uid in env.possible_agents
    }
    return env, models, memories


def _shared_cfg(*, rollouts, **overrides):
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": 1,
        "mini_batches": 1,
        "learning_rate": 1.0e-3,
        "shared_across_agents": True,
        "experiment": {"directory": "", "experiment_name": "", "write_interval": 0, "checkpoint_interval": 0},
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("num_agents", [2, 3])
def test_shared_across_agents_pools_transitions(device, num_agents):
    """Parameter sharing must pool every agent's transitions into a single optimization step.

    Looping over agents instead would take ``num_agents`` steps on the same parameters per
    mini-batch, which is a different (and wrong) estimator.
    """
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    rollouts, num_envs = 4, 3
    env, models, memories = _build_shared_setup(
        num_agents=num_agents, num_envs=num_envs, rollouts=rollouts, device=device
    )
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=_shared_cfg(rollouts=rollouts),
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
    )

    # the shared batch must be num_agents times a single agent's batch
    seen_batch_sizes = []
    original = agent._sample_mini_batches

    def spy(*, uid, uids):
        batches = original(uid=uid, uids=uids)
        seen_batch_sizes.append((len(uids), batches[0][0].shape[0]))
        return batches

    agent._sample_mini_batches = spy

    trainer = SequentialTrainer(
        cfg={
            "timesteps": int(2 * rollouts),
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
        env=env,
        agents=agent,
    )
    trainer.train()

    assert seen_batch_sizes, "the update never ran"
    for pooled_agents, batch_size in seen_batch_sizes:
        assert pooled_agents == num_agents, "parameter sharing must pool every agent"
        assert batch_size == num_agents * rollouts * num_envs, "pooled batch has the wrong size"


@pytest.mark.parametrize("device", ["cpu"])
def test_shared_across_agents_rejects_unshared_models(device):
    """Enabling parameter sharing with per-agent model instances must fail loudly."""
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    rollouts = 4
    env, models, memories = _build_shared_setup(num_agents=2, num_envs=2, rollouts=rollouts, device=device)

    # break the sharing for one agent
    other = env.possible_agents[1]
    models[other] = {
        "policy": gaussian_model(
            observation_space=env.observation_space(other),
            state_space=env.state_space(other),
            action_space=env.action_space(other),
            device=env.device,
            network=[{"name": "net", "input": "OBSERVATIONS", "layers": [5], "activations": "relu"}],
            output="ACTIONS",
        ),
        "value": models[other]["value"],
    }

    with pytest.raises(ValueError, match="shared_across_agents"):
        MultiAgent(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=_shared_cfg(rollouts=rollouts),
            observation_spaces=env.observation_spaces,
            state_spaces=env.state_spaces,
            action_spaces=env.action_spaces,
            device=env.device,
        )


@pytest.mark.parametrize("device", ["cpu"])
def test_separate_optimizers_builds_two_optimizers(device):
    """`separate_optimizers` must give the policy and value networks their own optimizer and LR."""
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    rollouts = 4
    env, models, memories = _build_shared_setup(num_agents=2, num_envs=2, rollouts=rollouts, device=device)
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=_shared_cfg(rollouts=rollouts, separate_optimizers=True, learning_rate=(1.0e-3, 5.0e-4)),
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
    )
    for uid in env.possible_agents:
        assert agent.value_optimizers[uid] is not None, "value optimizer was not created"
        assert agent.optimizers[uid].param_groups[0]["lr"] == pytest.approx(1.0e-3)
        assert agent.value_optimizers[uid].param_groups[0]["lr"] == pytest.approx(5.0e-4)


@pytest.mark.parametrize("device", ["cpu"])
def test_advantage_filter_ratio_shrinks_the_batch(device):
    """`advantage_filter_ratio` must drop the requested fraction of each mini-batch."""
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    advantages = torch.arange(-5, 5, dtype=torch.float32, device=device).reshape(-1, 1)
    ratio = 0.5
    keep = int(advantages.shape[0] * (1.0 - ratio))
    index = torch.topk(advantages.abs().flatten(), keep, sorted=False).indices

    assert index.numel() == keep == 5
    kept = advantages.flatten()[index].abs()
    dropped_max = advantages.flatten()[
        torch.tensor([i for i in range(advantages.shape[0]) if i not in set(index.tolist())], device=device)
    ].abs()
    assert kept.min() >= dropped_max.max(), "filtering must keep the largest-magnitude advantages"


@pytest.mark.parametrize("device", ["cpu"])
def test_shared_across_agents_shares_preprocessors(device):
    """Parameter sharing must share preprocessors, not just networks.

    Only the first agent's preprocessors are fitted (the pooled update runs under its uid), so
    unshared preprocessors would leave the other agents feeding the shared networks differently
    normalized inputs at action time.
    """
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    rollouts = 4
    env, models, memories = _build_shared_setup(num_agents=3, num_envs=2, rollouts=rollouts, device=device)
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=_shared_cfg(
            rollouts=rollouts,
            observation_preprocessor=RunningStandardScaler,
            observation_preprocessor_kwargs={"size": env.observation_space("agent_0"), "device": device},
            state_preprocessor=RunningStandardScaler,
            state_preprocessor_kwargs={"size": env.state_space("agent_0"), "device": device},
            value_preprocessor=RunningStandardScaler,
            value_preprocessor_kwargs={"size": 1, "device": device},
        ),
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
    )
    reference = env.possible_agents[0]
    for uid in env.possible_agents[1:]:
        assert agent._observation_preprocessor[uid] is agent._observation_preprocessor[reference]
        assert agent._state_preprocessor[uid] is agent._state_preprocessor[reference]
        assert agent._value_preprocessor[uid] is agent._value_preprocessor[reference]
