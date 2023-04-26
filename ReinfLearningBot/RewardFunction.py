from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED


class CustomReward(RewardFunction):
    def __init__(self):
        self.initial_ball_player_distance = None

    def reset(self, initial_state: GameState):
        ball_pos = initial_state.ball.position
        # FIXME: Careful, this impl works only if 1 player
        car_pos = initial_state.players[0].car_data.position
        self.initial_ball_player_distance = np.linalg.norm(car_pos - ball_pos) - BALL_RADIUS

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        current_dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_diff = self.initial_ball_player_distance - current_dist
        return np.exp(0.5 * dist_diff / CAR_MAX_SPEED) / 10

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
