import numpy as np


class HumanAgent:
    def __init__(self, player_id=1):
        self.player_id = player_id

    def take_action(self, state):
        act = input("Your turn: ").split()
        return act

    def get_action(self, *args, **kwargs):
        act = input("Your turn: ").split(",")
        acts = (int(act[0]), int(act[1]))
        return acts, None