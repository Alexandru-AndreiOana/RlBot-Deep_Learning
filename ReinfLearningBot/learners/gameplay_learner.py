from rlgym.envs import Match
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, RewardIfClosestToBall
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import RandomState, DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

from ReinfLearningBot.environment_config_objects.action_parser import CustomActionParser
from ReinfLearningBot.environment_config_objects.observation_builder import CustomObs
from ReinfLearningBot.environment_config_objects.reward_function import CustomBallPlayerDistanceReward

frame_skip = 8  # Number of ticks to repeat an action

fps = 120 // frame_skip
gamma = 0.999
agents_per_match = 2
num_instances = 2

target_steps = 750_000
steps = target_steps // (num_instances * agents_per_match)
batch_size = 50_000

training_interval = 20_000_000_000
mmr_save_frequency = 50_000_000


def get_match():  # Need to use a function so that each instance can call it and produce their own objects
    return Match(
        team_size=1,
        reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    goal=1000.0,
                    concede=-1000.0,
                    save=100.0,
                    shot=50.0,
                    demo=20.0,
                    touch=5,
                    boost_pickup=0.5
                ),
            ),
            (0.2, 0.4, 1.0)),
        # self_play=True,  in rlgym 1.2 'self_play' is depreciated. Uncomment line if using an earlier version and comment out spawn_opponents
        spawn_opponents=True,
        terminal_conditions=[TimeoutCondition(fps * 30), NoTouchTimeoutCondition(fps * 15), GoalScoredCondition()],
        obs_builder=CustomObs(),  # Not that advanced, good default
        state_setter=RandomState(),  # Resets to kickoff position
        action_parser=CustomActionParser()  # Discrete > Continuous don't @ me
    )


if __name__ == '__main__':

    def exit_save(model):
        model.save("./models/1s_config_1")


    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=80)  # Optional: add custom waiting time to load more instances
    env = VecCheckNan(env)
    env = VecMonitor(env)  # Useful for Tensorboard logging
    env = VecNormalize(env, norm_obs=True, gamma=gamma)

    try:
        model = PPO.load(
            "gibberish",
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs},
            # automatically adjusts to users changing instance count, may encounter shaping error otherwise
            # If you need to adjust parameters mid-training, you can use the below example as a guide
            # custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "n_epochs": 10, "learning_rate": 5e-5}
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh

        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=dict(pi=[192, 192], vf=[192, 192])
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=31,
            policy_kwargs=policy_kwargs,
            learning_rate=3.7e-4,
            ent_coef=0.05,  # From PPO Atari
            vf_coef=1.0,  # From PPO Atari
            gamma=gamma,
            verbose=3,
            batch_size=batch_size,
            n_steps=steps,
            clip_range=0.4,
            gae_lambda=0.8,
            tensorboard_log="./rl_tensorboard_log",
            device="auto"
        )

    callback = CheckpointCallback(round(5_000_000 / env.num_envs),
                                  save_path="./models/1s_config_1",
                                  name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            # may need to reset timesteps when you're running a diff number of instances than when you saved the model
            model.learn(training_interval,
                        callback=callback,
                        tb_log_name="1s_config_1",
                        reset_num_timesteps=False)  # can ignore callback if training_interval < callback target
            model.save("models/exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
