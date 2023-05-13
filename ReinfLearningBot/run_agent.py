import rlgym
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, \
    VelocityBallToGoalReward, EventReward

from stable_baselines3 import PPO
from rlgym.utils.state_setters import RandomState
from rlgym.utils.obs_builders import AdvancedObs

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, \
    GoalScoredCondition

from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

frame_skip = 8          # Number of ticks to repeat an action
half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

fps = 120 // frame_skip
agents_per_match = 2
num_instances = 4
target_steps = 1_000_000
steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
batch_size = target_steps//10 #getting the batch size down to something more manageable - 100k in this case
training_interval = 25_000_000
mmr_save_frequency = 50_000_000
# Constants
NUM_TEST_EPISODES = 1000


def test_agent(model, env, num_episodes):
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0

        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        while not done.any():
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total reward = {total_reward}")


if __name__ == "__main__":
    model_path = "models/BASELINE_1V1/rl_model_25000000_steps.zip"

    # Setting up the environment
    env = rlgym.make(
        game_speed=5,
        team_size=1,
        tick_skip=frame_skip,
        reward_fn=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0,
                ),
            ),
            (0.1, 1.0, 1.0)
        ),
        spawn_opponents=True,
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        obs_builder=AdvancedObs(),
        state_setter=RandomState(),
        action_parser=DiscreteAction()
    )

    env = SB3SingleInstanceEnv(env)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, gamma=0.9)

    model = PPO.load(model_path, env=env)

    test_agent(model, env, NUM_TEST_EPISODES)
