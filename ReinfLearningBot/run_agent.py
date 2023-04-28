import rlgym
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.state_setters import RandomState
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.envs import Match
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from stable_baselines3 import PPO

from ReinfLearningBot.RewardFunction import CustomReward

# ----------------
# CONSTANTS
NUM_TEST_EPISODES = 10

# ------------------
# Environment configuration objects
liu_distance = LiuDistancePlayerToBallReward()


def get_match():
    return Match(
        reward_function=CustomReward(),
        terminal_conditions=[TimeoutCondition(500)],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=KBMAction(),
        spawn_opponents=False,
    )


def test_agent(model, num_episodes):
    env = get_match()

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total reward = {total_reward}")


if __name__ == "__main__":
    model_path = "logs/cstm_rew_1/rl_model_25000000_steps.zip"
    model = PPO.load(model_path)

    test_agent(model, NUM_TEST_EPISODES)

