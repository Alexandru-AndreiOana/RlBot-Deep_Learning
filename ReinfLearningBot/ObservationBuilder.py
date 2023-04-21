from typing import Any

from rlgym.utils import common_values
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState

import numpy as np


class CustomObsBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        # TODO: Add observation for small boost pads as well?
        obs = []

        # If this observation is being built for a player on the orange team,
        # we need to invert all the physics data we use.
        inverted = player.team_num == common_values.ORANGE_TEAM

        if inverted:
            obs += state.inverted_ball.serialize()
            # obs += state.inverted_boost_pads
        else:
            obs += state.ball.serialize()
            # obs += state.boost_pads

        for player in state.players:
            if inverted:
                obs += player.inverted_car_data.serialize()
            else:
                obs += player.car_data.serialize()

        return np.asarray(obs, dtype=np.float32)
