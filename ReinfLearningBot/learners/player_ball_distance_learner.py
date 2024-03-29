from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward

from stable_baselines3 import PPO
from torch.nn import Tanh
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.state_setters import RandomState
from rlgym.utils.obs_builders import AdvancedObs

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from ReinfLearningBot.environment_config_objects.action_parser import CustomActionParser
from ReinfLearningBot.environment_config_objects.reward_function import CustomBallPlayerDistanceReward

# ----------------
# CONFIGURATION
CONFIG_NUMBER = 9
CONFIG_NAME = f"Config_TEST{CONFIG_NUMBER}"
LOAD_PREV_AGENT = True  # set true to train from a previous checkpoint

# CONSTANTS
NUM_PLAYERS = 1  # set for solo play
NUM_INSTANCES = 4  # concurrent game instances to run

NUM_ROLLOUT_STEPS = 32_768
NUM_ROLLOUT_STEPS_PER_AGENT = NUM_ROLLOUT_STEPS // (NUM_PLAYERS * NUM_INSTANCES)
GAMMA = 0.9
LEARNING_STEPS_TOTAL = int(6e10)

DEFAULT_TICK_SKIP = 8
PHYSICS_TICKS_PER_SECOND = 120
EP_LEN_SECONDS = 10

# ------------------
# Environment configuration objects
# TODO: Custom combined reward function for training 1v1
# reward_func = CombinedReward()
liu_distance = LiuDistancePlayerToBallReward()


# --------------------
# Methods

def get_match():
    return Match(
        reward_function=CustomBallPlayerDistanceReward(),
        terminal_conditions=[TimeoutCondition(500)],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=CustomActionParser(),
        spawn_opponents=False,
        game_speed=1
    )


if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=NUM_INSTANCES, wait_time=35)
    env = VecCheckNan(env)
    env = VecMonitor(env)  # enables logging for rollout phase (episode length, mean reward)
    env = VecNormalize(env, norm_obs=True, gamma=GAMMA)  # reward normalization

    # Save the model each n steps
    checkpoint_callback = CheckpointCallback(save_freq=round(5_000_000 / env.num_envs),
                                             save_path=f"./models/{CONFIG_NAME}",
                                             name_prefix="rl_model")

    if LOAD_PREV_AGENT:
        print("Loading from previous configuration.")
        model = PPO.load(path="./models/Config_9/rl_model_50000000_steps.zip",
                         env=env,
                         custom_objects=dict(n_envs=env.num_envs,
                                             _last_obs=None,
                                             learning_rate=2e-5,
                                             clip_range=0.1,
                                             batch_size=2048,
                                             n_epochs=10,
                                             ent_coef=0.0),
                         device="auto",
                         force_reset=True)
    else:
        print("Starting with a new configuration.")

        # Custom neural network architecture for the policy and value function
        policy_kwargs = dict(activation_fn=Tanh,
                             net_arch=dict(pi=[256, 256], vf=[256, 256]))

        model = PPO("MlpPolicy",
                    env=env,
                    n_epochs=20,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-5,
                    n_steps=4096,
                    batch_size=4096,
                    ent_coef=0.01,
                    verbose=2,
                    tensorboard_log="./rl_tensorboard_log",
                    device="auto")

    print("Learning will start.")
    model.learn(total_timesteps=LEARNING_STEPS_TOTAL,
                callback=[checkpoint_callback],
                tb_log_name=CONFIG_NAME,
                reset_num_timesteps=LOAD_PREV_AGENT)
