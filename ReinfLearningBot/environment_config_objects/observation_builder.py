import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder


class CustomObs(ObsBuilder):
    # Termeni de normalizare
    POS_STD = 2300
    ANG_STD = math.pi
    # Pozitiile portilor
    BLUE_GOAL = np.array(common_values.BLUE_GOAL_BACK) / POS_STD
    ORANGE_GOAL = np.array(common_values.ORANGE_GOAL_BACK) / POS_STD

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # Observatia este reprezentata simetric
        # pentru ambele echipe
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
            goals_position = self.ORANGE_GOAL + self.BLUE_GOAL
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads
            goals_position = self.BLUE_GOAL + self.ORANGE_GOAL

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads,
               np.array(goals_position)]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies, enemies = [], []
        # Sunt adaugati si ceilalti jucatori in observatie
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)
            team_obs.extend([
                # Distanta dintre pozitia oponentului si a agentului
                (other_car.position - player_car.position) / self.POS_STD,
                # Distanta dintre viteza liniara a oponentului si a agentului
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def reset(self, initial_state: GameState):
        pass

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        # Reprezentare simetrica
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car
