from experiments.experiment import Experiment
from rllib_integration.carla_env import CarlaEnv
from experiments.action_spaces import  discrete_actions, continuous_actions
from rllib_integration.carla_core import kill_all_servers
import yaml

from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents import sac

config_path = 'config.yaml'
rl_config_path = 'rl_config.yaml'

experiment = Experiment
experiment = discrete_actions(experiment, 4, 27)
# experiment = continuous_actions(experiment)

def parse_rl_config(path):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def parse_config(path):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['env'] = CarlaEnv
        config['env_config']['experiment']['type'] = experiment

    return config

def main():
    env_config = parse_config(config_path)
    rl_config = parse_rl_config(rl_config_path)
    default_config = sac.DEFAULT_CONFIG.copy()
    default_config["env_config"] = env_config['env_config']
    default_config['framework'] = "torch"
    default_config['env'] = CarlaEnv
    default_config['replay_buffer_config']['capacity'] = int(3e5)

    for key in rl_config.keys():
        default_config[key] = rl_config[key]
    trainer = SACTrainer(config=default_config)

    try:
        for i in range(10000):
                trainer.train()
    finally:
        kill_all_servers()
    kill_all_servers()

if __name__ == '__main__':
    main()
