import numpy as np
import argparse
import os
import json
import time
import random
import tqdm

from rl_gomoku.envs import ArrayGomoku
from rl_gomoku.agents import MCTSZeroAgent, GreedyAgent, HumanAgent
from rl_gomoku.utils import start_play, collect_data, start_self_play, collect_self_play_data, create_model_from_args


cache_data = json.load(open("./cache_data/quick_list_set.json", "r"))
parser = argparse.ArgumentParser(description='Example Argparse Program')

parser.add_argument('--checkpoint', type=str, default=None, help='Input file')
args = parser.parse_args()


def create_args():
    from rl_gomoku.utils import default_args
    model_args, training_args = default_args()
    project_args = {
    }
    training_args.update(
        {
            "log_freq": 200,
            "n_playout": 1200,
        }
    )
    model_args.update(
        {
            "model_type": "res5",
            "learning_rate": 0.0004,
            "device": "cpu"
        }
    )
    return project_args, model_args, training_args


def main():
    project_args, model_args, training_args = create_args()
    model_args.update(
        {
            "device": "cpu",
            "model_type": "res5"
        }
    )
    env = ArrayGomoku(board_size=training_args["board_size"])

    # model = create_model_from_args(model_args)
    # if args.checkpoint:
    #     model.load_checkpoint(args.checkpoint)
    # else:
    #     raise ValueError("model not found!")
    
    model = create_model_from_args({"device": "cuda:1", "model_type": "res5", "in_channels": 7, "drop_out": 0.1, "learning_rate": 0.001,
                                         "use_attention":True, "weight_decay": 0.001})
    
    test_model = create_model_from_args({"device": "cuda:1", "model_type": "res3", "in_channels": 7, "drop_out": 0.1, "learning_rate": 0.001,
                                         "use_attention":True, "weight_decay": 0.001})
    # test_model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/pre-attention-botzone-gomuku-2023-12-22-21-31-iteration-24000.pth")
    model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/ResNet5-gomuku-2023-12-24-14-27-iteration-42000.pth")
    test_model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/ResNet3-gomuku-2023-12-24-17-26-iteration-9000.pth")

    mcts_player = MCTSZeroAgent(model.forward, player_id=1, c_puct=1, n_playout=2000)
    test_player = MCTSZeroAgent(test_model.forward, player_id=2, c_puct=1, n_playout=400)
    human_a = HumanAgent(1)
    human_b = HumanAgent(2)
    player_a = GreedyAgent(training_args["board_size"], player_id=1, cache_data=cache_data)
    player_b = GreedyAgent(training_args["board_size"], player_id=2, cache_data=cache_data)
    for i in tqdm.tqdm(range(100)):
        seed = int(time.time())
        player_a.rng = np.random.default_rng(seed)
        player_b.rng = np.random.default_rng(seed)
        winner, *_ = start_play(env, mcts_player, human_b, render=True)
        # winner, *_ = start_play(env, mcts_player, test_player, render=True)
        time.sleep(5)

if __name__ == "__main__":
    main()