import rlgym
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.state_setters import RandomState
from stable_baselines3 import PPO
from ObservationBuilder import CustomObsBuilder


def run_agent(path):
    # Load the trained agent
    trained_model = PPO.load(path)

    # Create the environment
    liu_distance = LiuDistancePlayerToBallReward()
    env = rlgym.make(reward_fn=liu_distance, state_setter=RandomState(), obs_builder=CustomObsBuilder())

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
model_path = "./logs/model_8188_steps"
run_agent(model_path)
