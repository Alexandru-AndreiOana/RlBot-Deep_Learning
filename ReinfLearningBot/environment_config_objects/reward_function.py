from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED


class CustomBallPlayerDistanceReward(RewardFunction):
    """
    Rewards agent proportional to the decrease in the player-ball
    distance, compared to the start of the episode.
    """
    def __init__(self):
        self.initial_ball_player_distances = {}

    # TODO: Incorporate all rewards into this reward function
    # TODO: Reward player for scoring a goal only if they touched the ball >= once in that episode.

    def reset(self, initial_state: GameState):
        ball_pos = initial_state.ball.position
        for player in initial_state.players:
            car_pos = player.car_data.position
            self.initial_ball_player_distances[player.car_id] = np.linalg.norm(car_pos - ball_pos) - BALL_RADIUS

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        current_dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_diff = self.initial_ball_player_distances[player.car_id] - current_dist
        return np.exp(0.5 * dist_diff / CAR_MAX_SPEED) / 10

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


