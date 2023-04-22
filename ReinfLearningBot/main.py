import rlgym
import matplotlib.pyplot as plt
import rlgym_tools

from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.state_setters import RandomState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from ObservationBuilder import CustomObsBuilder

# ----------------
# CONSTANTS
DEFAULT_TICK_SKIP = 8
PHYSICS_TICKS_PER_SECOND = 120
EP_LEN_SECONDS = 10
GAMMA = 0.9
TIME_STEPS = int(6e10)

# ------------------
# Variables
max_steps = int(round(EP_LEN_SECONDS * PHYSICS_TICKS_PER_SECOND / DEFAULT_TICK_SKIP))

liu_distance = LiuDistancePlayerToBallReward()
terminal_condition = TimeoutCondition(max_steps)


# --------------------
# Methods

def get_match():
    return Match(
        reward_function=liu_distance,
        terminal_conditions=[TimeoutCondition(225)],
        obs_builder=CustomObsBuilder(),
        state_setter=RandomState(),
        action_parser=KBMAction(),
        tick_skip=DEFAULT_TICK_SKIP,

        # game_speed=1
    )


if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=8, wait_time=20)
    env = VecMonitor(env)  # enables logging for rollout phase (episode length, mean reward)
    env = VecNormalize(env, norm_obs=False, gamma=0.9)  # reward normalization

    # TODO: Try using a custom neural network architecture for training the policy
    model = PPO("MlpPolicy",
                env=env,
                ent_coef=0.01,
                verbose=0,
                tensorboard_log="./rl_tensorboard_log",
                device="auto")

    # Use a checkpoint callback to save the model during training
    checkpoint_callback = CheckpointCallback(save_freq=round(5_000_000 / env.num_envs), save_path='./logs/',
                                             name_prefix="rl_model")

    model.learn(total_timesteps=TIME_STEPS, callback=[checkpoint_callback], tb_log_name="speed_objective")
