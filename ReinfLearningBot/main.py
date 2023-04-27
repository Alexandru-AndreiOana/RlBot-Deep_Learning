import rlgym
import matplotlib.pyplot as plt
import rlgym_tools
from rlgym.utils.reward_functions import CombinedReward

from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.state_setters import RandomState
from rlgym.utils.obs_builders import AdvancedObs

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from ReinfLearningBot.RewardFunction import CustomReward

# ----------------
# CONSTANTS
LOAD_PREV_AGENT = True  # set true to train from a previous checkpoint
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
        reward_function=CustomReward(),
        # TODO: Reduce timeout condition (might speed up learning)
        terminal_conditions=[TimeoutCondition(500)],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=KBMAction(),
        spawn_opponents=False,
    )


if __name__ == "__main__":
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=NUM_INSTANCES, wait_time=35)
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
        model = PPO.load(path="./logs/dist_player_ball_cstm_rw/rl_model_25000000_steps.zip",
                         env=env,
                         custom_objects=dict(n_envs=env.num_envs,
                                             _last_obs=None,
                                             learning_rate=2e-5,
                                             batch_size=2048,
                                             n_epochs=4,
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
                    seed=1,
                    verbose=2,
                    tensorboard_log="./rl_tensorboard_log",
                    device="auto")

    print("Learning will start.")
    model.learn(total_timesteps=LEARNING_STEPS_TOTAL,
                callback=[checkpoint_callback],
                tb_log_name="plr_ball_cstm_rw_2",
                reset_num_timesteps=LOAD_PREV_AGENT)
