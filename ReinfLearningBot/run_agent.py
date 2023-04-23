import rlgym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from stable_baselines3 import PPO
from ObservationBuilder import CustomObsBuilder
from ReinfLearningBot.main import DEFAULT_TICK_SKIP


def run_agent(path):
    # Load the trained agent
    trained_model = PPO.load(path)

    # Create the environment
    liu_distance = LiuDistancePlayerToBallReward()
    env = rlgym.make(
        reward_fn=LiuDistancePlayerToBallReward(),
        terminal_conditions=[TimeoutCondition(400)],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=KBMAction(),
        tick_skip=DEFAULT_TICK_SKIP,
        game_speed=2

        # game_speed=1
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
# FIXME: Not working with current configuration
model_path = "./logs/rl_model_20000000_steps.zip"
run_agent(model_path)
