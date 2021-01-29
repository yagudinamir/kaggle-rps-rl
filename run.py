"""A simple multi-agent env with two agents playing rock paper scissors.

This demonstrates running the following policies in competition:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PG policies
    (4) LSTM policy with custom entropy loss
"""

import argparse
from gym.spaces import Discrete
import os
import random

from ray import tune
from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.registry import get_agent_class
# from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
from env import RockPaperScissors
from ray.rllib.examples.policy.rock_paper_scissors_dummies import \
    BeatLastHeuristic, AlwaysSameHeuristic
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

from policy import GeomPolicy

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
os.environ["TWIX_PYTHON_ANONYMOUS_LOG_LEVEL"] = "INFO"

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=150)
parser.add_argument("--stop-reward", type=float, default=1000.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)



def run_heuristic_vs_learned(args, use_lstm=False, trainer="PG"):
    """Run heuristic policies vs a learned agent.

    The learned agent should eventually reach a reward of ~5 with
    use_lstm=False, and ~7 with use_lstm=True. The reason the LSTM policy
    can perform better is since it can distinguish between the always_same vs
    beat_last heuristics.
    """

    def select_policy(agent_id):
        if agent_id == "player1":
            return "learned"
        else:
            return random.choice(["geom"])

    config = {
        "env": RockPaperScissors,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1.,
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "multiagent": {
            "policies_to_train": ["learned"],
            "policies": {
#                 "always_same": (AlwaysSameHeuristic, Discrete(3), Discrete(3),
#                                 {}),
#                 "beat_last": (BeatLastHeuristic, Discrete(3), Discrete(3), {}),
                "geom": (GeomPolicy, Discrete(3), Discrete(3), {}),
                "learned": (None, Discrete(3), Discrete(3), {
                    "model": {
                        "use_lstm": use_lstm
                    },
                    "framework": "torch"
                }),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "torch"
    }
    
    cls = get_agent_class(trainer) if isinstance(trainer, str) else trainer
    tune.run(
        cls,
        config=config
    )
#     env = trainer_obj.workers.local_worker().env
#     for _ in range(args.stop_iters):
#         results = trainer_obj.train()
# #         print(results)
#         # Timesteps reached.
#         if results["timesteps_total"] > args.stop_timesteps:
#             break
#         # Reward (difference) reached -> all good, return.
#         elif env.player1_score - env.player2_score > args.stop_reward:
#             print("TRAINED")
#             return
#     # Reward (difference) not reached: Error if `as_test`.
#     if args.as_test:
#         raise ValueError(
#             "Desired reward difference ({}) not reached! Only got to {}.".
#             format(args.stop_reward, env.player1_score - env.player2_score))




if __name__ == "__main__":
    args = parser.parse_args()

    run_heuristic_vs_learned(args, use_lstm=True)