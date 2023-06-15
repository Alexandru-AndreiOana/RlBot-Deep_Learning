from rlgym.envs import Match
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward
from ReinfLearningBot.environment_config_objects.action_parser import CustomActionParser
from ReinfLearningBot.environment_config_objects.observation_builder import CustomObs
from enum import Enum

from ReinfLearningBot.constants import Constants

# COMPUTED CONSTANTS
fps = 120 // Constants.FRAME_SKIP.value
steps = Constants.TARGET_STEPS.value // (Constants.NUM_INSTANCES.value * Constants.AGENTS_PER_MATCH.value)


class LearningConfiguration(Enum):
    TRAINING = 1
    EVALUATION = 2


def get_match():  # Functia este apelata pentru fiecare instanta lansata
    return Match(
        team_size=1,  # setare pentru mod 1v1,
        # game_speed=2,
        reward_function=CombinedReward((
            VelocityReward(),
            VelocityPlayerToBallReward(use_scalar_projection=True),
            VelocityBallToGoalReward(use_scalar_projection=True),
            EventReward(
                goal=1000.0,
                concede=-1000.0,
                save=100.0,
                shot=50.0,
                demo=20.0,
                touch=10,
                boost_pickup=0.5
            )),
            (0.0001, 0.0002, 0.0008, 1.0)),
        spawn_opponents=True,  # antrenare prin self-play
        terminal_conditions=[TimeoutCondition(fps * 30), NoTouchTimeoutCondition(fps * 15)],
        obs_builder=CustomObs(),
        state_setter=RandomState(ball_rand_speed=True, cars_rand_speed=True),
        action_parser=CustomActionParser()
    )


if __name__ == '__main__':

    def exit_save(model):
        model.save(f"./models/{Constants.CONFIG_NAME.value}_exit")


    LEARNING_PHASE = LearningConfiguration.TRAINING

    env = SB3MultipleInstanceEnv(get_match, Constants.NUM_INSTANCES.value, wait_time=50)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, gamma=Constants.GAMMA.value)

    try:
        model = PPO.load(
            "models/Checkpoint3/rl_model_215000000_steps.zip",
            env=env,
            custom_objects=dict(n_envs=env.num_envs,
                                n_steps=steps,
                                batch_size=Constants.BATCH_SIZE.value,
                                clip_range=0.1,
                                learning_rate=7e-5,
                                ent_coef=0.01,
                                vf_coef=0.7,
                                gae_lambda=0.9,
                                _last_obs=None,
                                verbose=3,
                                tensorboard_log="./rl_tensorboard_log"),
            force_reset=True
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
            ent_coef=0.05,
            vf_coef=1.0,
            gamma=Constants.GAMMA.value,
            verbose=3,
            batch_size=Constants.BATCH_SIZE.value,
            n_steps=steps,
            clip_range=0.4,
            gae_lambda=0.8,
            tensorboard_log="./rl_tensorboard_log",
            device="cpu"
        )

    callback = CheckpointCallback(round(5_000_000 / env.num_envs),
                                  save_path=f"./models/{Constants.CONFIG_NAME.value}",
                                  name_prefix="rl_model")

    try:
        if LEARNING_PHASE == LearningConfiguration.TRAINING:
            print("Starting Training")
            print("Training on:", model.device)
            model.learn(total_timesteps=Constants.TRAINING_INTERVAL.value,
                        callback=callback,
                        tb_log_name=Constants.CONFIG_NAME.value)

        elif LEARNING_PHASE == LearningConfiguration.EVALUATION:
            print("Starting Evaluation")
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000, deterministic=True)
            print(mean_reward, std_reward)


    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
