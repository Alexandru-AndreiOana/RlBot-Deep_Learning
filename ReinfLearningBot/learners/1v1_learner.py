from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

from ReinfLearningBot.environment_config_objects.action_parser import CustomActionParser
from ReinfLearningBot.environment_config_objects.reward_function import CustomBallPlayerDistanceReward

if __name__ == '__main__':
    frame_skip = 8  # Number of ticks to repeat an action

    fps = 120 // frame_skip
    gamma = 0.999
    agents_per_match = 2
    num_instances = 4

    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = target_steps // 10

    training_interval = 20_000_000_000
    mmr_save_frequency = 50_000_000


    def exit_save(model):
        model.save("./models/1s_config_1")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    CustomBallPlayerDistanceReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        goal=100.0,
                        concede=-100.0,
                        save=5.0,
                        demo=5.0,
                        shot=5.0,
                        touch=1,
                        boost_pickup=0.1
                    ),
                ),
                (0.1, 0.2, 1.0)),
            # self_play=True,  in rlgym 1.2 'self_play' is depreciated. Uncomment line if using an earlier version and comment out spawn_opponents
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(fps * 60), NoTouchTimeoutCondition(fps * 30), GoalScoredCondition()],
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=RandomState(),  # Resets to kickoff position
            action_parser=CustomActionParser()  # Discrete > Continuous don't @ me
        )


    env = SB3MultipleInstanceEnv(get_match, num_instances)  # Optional: add custom waiting time to load more instances
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
            net_arch=dict(pi=[512, 256, 256, 256], vf=[1024, 512, 256, 256, 256]),
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,
            ent_coef=0.01,  # From PPO Atari
            vf_coef=1.,  # From PPO Atari
            gamma=gamma,
            verbose=3,
            batch_size=batch_size,
            n_steps=steps,
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
