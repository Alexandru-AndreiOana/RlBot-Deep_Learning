import rlgym
import matplotlib.pyplot as plt
import rlgym_tools

from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.reward_functions.common_rewards import FaceBallReward
from rlgym_tools.extra_rewards.multiply_rewards import MultiplyRewards

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.state_setters import RandomState
from rlgym.utils.obs_builders import AdvancedObs

from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from ObservationBuilder import CustomObsBuilder
from ReinfLearningBot.RewardFunction import CustomReward

# ----------------
# CONSTANTS
LOAD_PREV_AGENT = True  # set true to train from a previous checkpoint
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
        reward_function=CustomReward(),
        # TODO: Reduce timeout condition (might speed up learning)
        terminal_conditions=[TimeoutCondition(400)],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=KBMAction()
    )


if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=4, wait_time=35)
    env = VecCheckNan(env)
    env = VecMonitor(env)  # enables logging for rollout phase (episode length, mean reward)
    env = VecNormalize(env, norm_obs=True, gamma=GAMMA)  # reward normalization

    # Save the model each n steps
    checkpoint_callback = CheckpointCallback(save_freq=round(5_000_000 / env.num_envs),
                                             save_path='./logs/dist_player_ball_cstm_rw',
                                             name_prefix="rl_model")

    # TODO: Try using a custom neural network architecture for training the policy
    if LOAD_PREV_AGENT:
        print("Loading from previous configuration.")
        model = PPO.load(path="./logs/dist_player_ball_2/rl_model_45000000_steps.zip",
                         env=env,
                         custom_objects=dict(n_envs=env.num_envs,
                                             _last_obs=None,
                                             learning_rate=3e-5,
                                             ent_coef=0.01),
                         device="auto",
                         force_reset=True)
    else:
        print("Starting with a new configuration.")
        model = PPO("MlpPolicy",
                    env=env,
                    n_epochs=30,
                    learning_rate=1e-4,
                    n_steps=4096,
                    batch_size=4096,
                    ent_coef=0.01,
                    verbose=2,
                    tensorboard_log="./rl_tensorboard_log",
                    device="auto")

    print("Learning will start.")
    model.learn(total_timesteps=TIME_STEPS,
                callback=[checkpoint_callback],
                tb_log_name="pl_ball_cstm_rw",
                reset_num_timesteps=LOAD_PREV_AGENT)
