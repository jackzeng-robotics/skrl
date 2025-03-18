from typing import Any, Mapping, Type, Union

import copy

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
from gymnasium.spaces import Box
import numpy as np

class Runner:
    def __init__(self, path: str) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        self._cfg = self.load_cfg_from_yaml(path)

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        if self._cfg["models"]["CTDE"]:
            self.possible_agents = ["falcon1", "falcon2", "falcon3"]
            self._models = self._generate_models_CTDE(copy.deepcopy(self._cfg))
        else:
            self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance"""
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance"""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file

        :param path: File path

        :return: Loaded configuration, or an empty dict if an error has occurred
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "gaussianmixin":
            from skrl.utils.model_instantiators.torch import gaussian_model as component
        elif name == "categoricalmixin":
            from skrl.utils.model_instantiators.torch import categorical_model as component
        elif name == "deterministicmixin":
            from skrl.utils.model_instantiators.torch import deterministic_model as component
        elif name == "multivariategaussianmixin":
            from skrl.utils.model_instantiators.torch import multivariate_gaussian_model as component
        elif name == "shared":
            from skrl.utils.model_instantiators.torch import shared_model as component
        # memory
        elif name == "randommemory":
            from skrl.memories.torch import RandomMemory as component
        # agent
        elif name in ["a2c", "a2c_default_config"]:
            from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG

            component = A2C_DEFAULT_CONFIG if "default_config" in name else A2C
        elif name in ["amp", "amp_default_config"]:
            from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG

            component = AMP_DEFAULT_CONFIG if "default_config" in name else AMP
        elif name in ["cem", "cem_default_config"]:
            from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG

            component = CEM_DEFAULT_CONFIG if "default_config" in name else CEM
        elif name in ["ddpg", "ddpg_default_config"]:
            from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

            component = DDPG_DEFAULT_CONFIG if "default_config" in name else DDPG
        elif name in ["ddqn", "ddqn_default_config"]:
            from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG

            component = DDQN_DEFAULT_CONFIG if "default_config" in name else DDQN
        elif name in ["dqn", "dqn_default_config"]:
            from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG

            component = DQN_DEFAULT_CONFIG if "default_config" in name else DQN
        elif name in ["ppo", "ppo_default_config"]:
            from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO
        elif name in ["rpo", "rpo_default_config"]:
            from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

            component = RPO_DEFAULT_CONFIG if "default_config" in name else RPO
        elif name in ["sac", "sac_default_config"]:
            from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

            component = SAC_DEFAULT_CONFIG if "default_config" in name else SAC
        elif name in ["td3", "td3_default_config"]:
            from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

            component = TD3_DEFAULT_CONFIG if "default_config" in name else TD3
        elif name in ["trpo", "trpo_default_config"]:
            from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG

            component = TRPO_DEFAULT_CONFIG if "default_config" in name else TRPO
        # multi-agent
        elif name in ["ippo", "ippo_default_config"]:
            from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG

            component = IPPO_DEFAULT_CONFIG if "default_config" in name else IPPO
        elif name in ["mappo", "mappo_default_config"]:
            from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

            component = MAPPO_DEFAULT_CONFIG if "default_config" in name else MAPPO
        # trainer
        elif name == "sequentialtrainer":
            from skrl.trainers.torch import SequentialTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(self, cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        # multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = "cpu"
        possible_agents = ["agent"]
        state_spaces = {"agent": None}
        observation_spaces = {"agent": Box(-np.inf, np.inf, (cfg["models"]["input_space"],), np.float32)}
        action_spaces = {"agent": Box(-np.inf, np.inf, (cfg["models"]["action_space"],), np.float32)}

        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            models[agent_id] = {}
            # non-shared models
            if _cfg["models"]["separate"]:
                # get instantiator function and remove 'class' field
                try:
                    model_class = self._class(_cfg["models"]["policy"]["class"])
                    del _cfg["models"]["policy"]["class"]
                except KeyError:
                    model_class = self._class("GaussianMixin")
                    logger.warning("No 'class' field defined in 'models:policy' cfg. 'GaussianMixin' will be used as default")
                # print model source
                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["policy"]),
                    return_source=True,
                )
                print("--------------------------------------------------\n")
                print(source)
                print("--------------------------------------------------")
                # instantiate model
                models[agent_id]["policy"] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["policy"]),
                )
                # get instantiator function and remove 'class' field
                try:
                    model_class = self._class(_cfg["models"]["value"]["class"])
                    del _cfg["models"]["value"]["class"]
                except KeyError:
                    model_class = self._class("DeterministicMixin")
                    logger.warning("No 'class' field defined in 'models:value' cfg. 'DeterministicMixin' will be used as default")
                # print model source
                source = model_class(
                    observation_space=(state_spaces if agent_class in [MAPPO] else observation_spaces)[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["value"]),
                    return_source=True,
                )
                print("--------------------------------------------------\n")
                print(source)
                print("--------------------------------------------------")
                # instantiate model
                models[agent_id]["value"] = model_class(
                    observation_space=(state_spaces if agent_class in [MAPPO] else observation_spaces)[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["value"]),
                )
            # shared models
            else:
                # remove 'class' field
                try:
                    del _cfg["models"]["policy"]["class"]
                except KeyError:
                    logger.warning("No 'class' field defined in 'models:policy' cfg. 'GaussianMixin' will be used as default")
                try:
                    del _cfg["models"]["value"]["class"]
                except KeyError:
                    logger.warning("No 'class' field defined in 'models:value' cfg. 'DeterministicMixin' will be used as default")
                model_class = self._class("Shared")
                # print model source
                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=None,
                    roles=["policy", "value"],
                    parameters=[
                        self._process_cfg(_cfg["models"]["policy"]),
                        self._process_cfg(_cfg["models"]["value"]),
                    ],
                    return_source=True,
                )
                print("--------------------------------------------------\n")
                print(source)
                print("--------------------------------------------------")
                # instantiate model
                models[agent_id]["policy"] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=None,
                    roles=["policy", "value"],
                    parameters=[
                        self._process_cfg(_cfg["models"]["policy"]),
                        self._process_cfg(_cfg["models"]["value"]),
                    ],
                )
                models[agent_id]["value"] = models[agent_id]["policy"]

        return models
    
    def _generate_models_CTDE(
        self, cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param cfg: A configuration dictionary

        :return: Model instances
        """
        # multi_agent = isinstance(env, MultiAgentEnvWrapper) assume this is always multi_agent
        device = "cpu"
        state_spaces = {agent: Box(-np.inf, np.inf, (cfg["models"]["state_space"],), np.float32) for agent in self.possible_agents}
        observation_spaces = {agent: Box(-np.inf, np.inf, (cfg["models"]["input_space"],), np.float32) for agent in self.possible_agents}
        action_spaces = {agent: Box(-np.inf, np.inf, (cfg["models"]["action_space"],), np.float32) for agent in self.possible_agents}

        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # instantiate models
        models = {}
        for agent_id in self.possible_agents:
            models[agent_id] = {}
        _cfg = copy.deepcopy(cfg)
        models_cfg = _cfg.get("models")
        if not models_cfg:
            raise ValueError("No 'models' are defined in cfg")
        # get separate (non-shared) configuration and remove 'separate' key
        try:
            separate = models_cfg["separate"]
            separate_actors = models_cfg["separate_actors"]
            separate_critics = models_cfg["separate_critics"]
            del models_cfg["separate"]
            del models_cfg["CTDE"]
            del models_cfg["separate_actors"]
            del models_cfg["separate_critics"]
            del models_cfg["state_space"]
            del models_cfg["input_space"]
            del models_cfg["action_space"]
        except KeyError:
            separate = True
            logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")

        # TODO: generalize this later
        if separate:
            if not separate_critics and separate_actors:
                for role in models_cfg:
                    if role =="policy":
                        # instantiate models
                        for agent_id in self.possible_agents:
                            # get instantiator function and remove 'class' key
                            model_class = models_cfg[role].get("class")
                            model_class = self._component(model_class)
                            # get specific spaces according to agent/model cfg
                            observation_space = observation_spaces[agent_id]
                            if agent_class == "mappo" and role == "value":
                                observation_space = state_spaces[agent_id]
                            # print model source
                            source = model_class(
                                observation_space=observation_space,
                                action_space=action_spaces[agent_id],
                                device=device,
                                **self._process_cfg(models_cfg[role]),
                                return_source=True,
                            )
                            print("==================================================")
                            print(f"Model (role): {role}")
                            print("==================================================\n")
                            print(source)
                            print("--------------------------------------------------")
                            # instantiate model
                            models[agent_id][role] = model_class(
                                observation_space=observation_space,
                                action_space=action_spaces[agent_id],
                                device=device,
                                **self._process_cfg(models_cfg[role]),
                            )
                    elif role == "value":
                        # get instantiator function and remove 'class' key
                        model_class = models_cfg[role].get("class")
                        if not model_class:
                            raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                        del models_cfg[role]["class"]
                        model_class = self._component(model_class)
                        # get specific spaces according to agent/model cfg
                        observation_space = observation_spaces[next(iter(self.possible_agents))] # assume the observation space is the same for all agents
                        if agent_class == "mappo" and role == "value":
                            observation_space = state_spaces[next(iter(self.possible_agents))] # assume the state space is the same for all agents
                        # print model source
                        source = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[next(iter(self.possible_agents))], # assume the action space is the same for all agents
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        print("==================================================")
                        print(f"Model (role): {role}")
                        print("==================================================\n")
                        print(source)
                        print("--------------------------------------------------")
                        # instantiate model
                        current_model = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[next(iter(self.possible_agents))], # assume the action space is the same for all agents
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                        )
                        current_model.init_state_dict(role)
                        
                        for agent_id in self.possible_agents:
                            models[agent_id][role] = current_model

            elif not separate_actors and not separate_critics:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    # get specific spaces according to agent/model cfg
                    observation_space = observation_spaces[next(iter(self.possible_agents))] # assume the observation space is the same for all agents
                    if agent_class == "mappo" and role == "value":
                        observation_space = state_spaces[next(iter(self.possible_agents))] # assume the state space is the same for all agents
                    # print model source
                    source = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[next(iter(self.possible_agents))], # assume the action space is the same for all agents
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                        return_source=True,
                    )
                    print("==================================================")
                    print(f"Model (role): {role}")
                    print("==================================================\n")
                    print(source)
                    print("--------------------------------------------------")
                    # instantiate model
                    current_model = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[next(iter(self.possible_agents))], # assume the action space is the same for all agents
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
                    current_model.init_state_dict(role)
                    
                    for agent_id in self.possible_agents:
                        models[agent_id][role] = current_model

        return models

    def _generate_agent(self, cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Model]]) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        # multi_agent = isinstance(env, MultiAgentEnvWrapper)
        num_envs = 1
        device = "cpu"
        possible_agents = ["agent"]
        state_spaces = {"agent": None}
        observation_spaces = {"agent": Box(-np.inf, np.inf, (cfg["models"]["input_space"],), np.float32)}
        action_spaces = {"agent": Box(-np.inf, np.inf, (cfg["models"]["action_space"],), np.float32)}

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # single-agent configuration and instantiation
        if agent_class in ["amp"]:
            agent_id = possible_agents[0]
            try:
                amp_observation_space = env.amp_observation_space
            except Exception as e:
                logger.warning(
                    "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                )
                amp_observation_space = observation_spaces[agent_id]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({"size": observation_spaces[agent_id], "device": device})
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_cfg["amp_state_preprocessor_kwargs"].update({"size": amp_observation_space, "device": device})

            motion_dataset = None
            if cfg.get("motion_dataset"):
                motion_dataset_class = cfg["motion_dataset"].get("class")
                if not motion_dataset_class:
                    raise ValueError(f"No 'class' field defined in 'motion_dataset' cfg")
                del cfg["motion_dataset"]["class"]
                motion_dataset = self._component(motion_dataset_class)(
                    device=device, **self._process_cfg(cfg.get("motion_dataset", {}))
                )
            reply_buffer = None
            if cfg.get("reply_buffer"):
                reply_buffer_class = cfg["reply_buffer"].get("class")
                if not reply_buffer_class:
                    raise ValueError(f"No 'class' field defined in 'reply_buffer' cfg")
                del cfg["reply_buffer"]["class"]
                reply_buffer = self._component(reply_buffer_class)(
                    device=device, **self._process_cfg(cfg.get("reply_buffer", {}))
                )

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
                "amp_observation_space": amp_observation_space,
                "motion_dataset": motion_dataset,
                "reply_buffer": reply_buffer,
                "collect_reference_motions": lambda num_samples: env.collect_reference_motions(num_samples),
            }
        elif agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in ["ippo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        elif agent_class in ["mappo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["shared_state_preprocessor_kwargs"].update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "shared_observation_spaces": state_spaces,
                "possible_agents": possible_agents,
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

    def run(self, mode: str = "train") -> None:
        """Run the training/evaluation

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation (default: ``"train"``)

        :raises ValueError: The specified running mode is not valid
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
