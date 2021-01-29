import gym
import numpy as np
import random

from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement

from geom_agent import Agent

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    
int2action = {
    0: RockPaperScissors.ROCK,
    1: RockPaperScissors.PAPER,
    2: RockPaperScissors.SCISSORS
}

class GeomPolicy(Policy):
    """Play the move that would beat the last move of the opponent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()
        self.agent = Agent()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        def successor(x):
            if x[RockPaperScissors.ROCK] == 1:
                return int2action[self.agent(AttrDict(step=1, lastOpponentAction=0), AttrDict(signs=3))]
            elif x[RockPaperScissors.PAPER] == 1:
                return int2action[self.agent(AttrDict(step=1, lastOpponentAction=1), AttrDict(signs=3))]
            elif x[RockPaperScissors.SCISSORS] == 1:
                return int2action[self.agent(AttrDict(step=1, lastOpponentAction=2), AttrDict(signs=3))]

        return [successor(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
    
