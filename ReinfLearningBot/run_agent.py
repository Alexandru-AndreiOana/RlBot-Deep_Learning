import rlgym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv, SB3SingleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize

from ObservationBuilder import CustomObsBuilder
from ReinfLearningBot.RewardFunction import CustomReward
from ReinfLearningBot.main import DEFAULT_TICK_SKIP, get_match


def run_agent(path):
    env = rlgym.make(reward_fn=CustomReward(),
                     obs_builder=AdvancedObs(),
                     state_setter=RandomState(),
                     action_parser=KBMAction(),
                     terminal_conditions=[TimeoutCondition(400)],
                     game_speed=2
                     )
    # env = SB3SingleInstanceEnv(env)
    # env = VecCheckNan(env)
    # env = VecMonitor(env)  # enables logging for rollout phase (episode length, mean reward)
    # env = VecNormalize(env, norm_obs=True, gamma=0.9)  # reward normalization

    # Load the trained agent
    trained_model = PPO.load(
        path=path,
        env=env,
        custom_objects=dict(gamma=0.9, ent_coef=0.0),
        # # Need this to change number of agents
        device="auto",  # Need to set device again (if using a specific one)
        force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
    )

    # Run the agent for a specific number of episodes
    num_episodes = 100000

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = trained_model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


# Load and run the agent
# TODO: Add variable for folder and model iteration
if __name__ == "__main__":
    model_path = "./logs/dist_player_ball_3/rl_model_10000000_steps.zip"
    run_agent(model_path)
