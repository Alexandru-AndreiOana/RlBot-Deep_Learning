import rlgym

from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.state_setters import RandomState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from ReinfLearningBot.environment_config_objects.observation_builder import CustomObsBuilder

# ----------------
# Variables
DEFAULT_TICK_SKIP = 8
PHYSICS_TICKS_PER_SECOND = 120
EP_LEN_SECONDS = 15

max_steps = int(round(EP_LEN_SECONDS * PHYSICS_TICKS_PER_SECOND / DEFAULT_TICK_SKIP))

liu_distance = LiuDistancePlayerToBallReward()
terminal_condition = TimeoutCondition(max_steps)

# --------------------
# Methods


if __name__ == "__main__":
    # TODO: Change back to random state setter (remove if multiple instances work)
    # env = rlgym.make(reward_fn=liu_distance,
    #                  obs_builder=CustomObsBuilder(),
    #                  terminal_conditions=[terminal_condition],
    #                  game_speed=1
    #                  )

    env = rlgym.make(reward_fn=liu_distance, terminal_conditions=[terminal_condition],
                     obs_builder=CustomObsBuilder(),
                     state_setter=RandomState(),
                     action_parser=DefaultAction())

    # TODO: Try using a custom neural network architecture for training the policy
    model = PPO("MlpPolicy",
                env=env,
                n_steps=8192,
                batch_size=256,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log="./rl_tensorboard_log")

    # Increase the number of training steps
    TIME_STEPS = int(6e10)

    # Use a checkpoint callback to save the model during training
    checkpoint_callback = CheckpointCallback(save_freq=5000000, save_path='models/',
                                             name_prefix='model')

    print("starts learning")
    model.learn(total_timesteps=TIME_STEPS, callback=[checkpoint_callback])
