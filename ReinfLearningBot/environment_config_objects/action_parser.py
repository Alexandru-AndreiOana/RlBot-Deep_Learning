from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


class CustomActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self._available_actions = self._get_all_possible_actions()

    @staticmethod
    def _get_all_possible_actions():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, -0.5, 0, 0.5, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Can side flip with air roll only
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Power slide only improves landings, not used in aerial play
                            handbrake = 1
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._available_actions))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        parsed_actions = []

        for action in actions:
            parsed_actions.append(self._available_actions[action])

        return np.asarray(parsed_actions)


if __name__ == '__main__':
    ap = CustomActionParser()
    print(ap.get_action_space())
