import argparse
import os
import json
import time
import random
import tqdm
import datetime 
import wandb
import joblib
import numpy as np

from typing import Protocol
from collections import deque
from rl_gomoku.envs import ArrayGomoku
from rl_gomoku.agents import MCTSZeroAgent, GreedyAgent, HumanAgent
from rl_gomoku.common import AlphaZero2
from rl_gomoku.utils import (
    start_play, 
    replay_game_from_list,
    collect_data, 
    collect_self_play_data, 
    collect_botzone_data, 
    _targs_t,
    ModelInterface,
)


cache_data = json.load(open("./cache_data/quick_list_set.json", "r"))
botzone_data = joblib.load("./cache_data/botzone.joblib")


class _args_t(Protocol):
    checkpoint: str
    pretrain: bool
    render: bool
    log_to_wandb: bool
    use_botzone: bool

parser = argparse.ArgumentParser(description='Gomoku game')
parser.add_argument('--checkpoint', type=str, default=None, help='Input file')
parser.add_argument('--pretrain', action="store_true", help="pretrain the model")
parser.add_argument('--render', action="store_true")
parser.add_argument("--log_to_wandb", action="store_true")
parser.add_argument("--use_botzone", action="store_true")
args: _args_t = parser.parse_args()

if args.use_botzone and not args.pretrain:
    raise ValueError("botzone can only be allow in pretrain mode.")


def policy_update(model: ModelInterface, buffer: deque, configs: _targs_t):
    """update the policy-value net"""
    mini_batch = random.sample(buffer, configs["batch_size"])
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = model.policy_value(state_batch)

    for i in range(configs["update_times_per_episode"]):
        loss, entropy = model.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch
        )
        new_probs, new_v = model.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1))

    explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         np.var(np.array(winner_batch)))
    
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         np.var(np.array(winner_batch)))

    print(("kl:{:.5f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl, loss, entropy, explained_var_old, explained_var_new))
    return loss, entropy


def main():
    from rl_gomoku.utils import create_model_from_args
    all_args = {}
    pargs, model_args, train_args = create_args()
    all_args.update(pargs)
    all_args.update(model_args)
    all_args.update(train_args)
    
    try:
        env = ArrayGomoku(board_size=train_args['board_size'])
        buffer = deque(maxlen=train_args['buffer_size'])
        model = create_model_from_args(model_args)
        test_model = ModelInterface(model_type="res5", use_attention=True)
        test_model.load_checkpoint("/home/nymath/dev/rl/gomoku/model/ResNet5-gomuku-2023-12-24-14-27-iteration-12000.pth")
        
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
            
        mcts_player = MCTSZeroAgent(model.forward, player_id=1, n_playout=train_args["n_playout"])
        player_a = GreedyAgent(train_args["board_size"], player_id=1, cache_data=cache_data)
        player_b = GreedyAgent(train_args["board_size"], player_id=2, cache_data=cache_data) 
        
        if args.log_to_wandb:
            if pargs["project_name"] is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=pargs["project_name"],
                # entity='treaptofun',
                config=all_args,
                name=pargs["run_name"],
            )
            wandb.watch(model.model)
        
        for i in tqdm.tqdm(range(train_args['n_iterations'])):
            data_shape = (train_args['board_size'], train_args['board_size'])
            total_winners = np.array([0, 0, 0])

            if args.pretrain == True:
                # load states here.
                if args.use_botzone:
                    while True:
                        states = random.choice(botzone_data)["data"]
                        valid = True
                        try:
                            replay_game_from_list(env, states)
                        except ValueError:
                            valid = False
                        except IndexError:
                            valid = False
                            
                        if valid == True:
                            break
                        
                    episode_len = collect_botzone_data(
                        buffer=buffer,
                        env=env,
                        states=states,
                        data_shape=data_shape,
                        render=args.render,
                    )
                else:
                    episode_len = collect_data(
                        buffer, 
                        env, 
                        player_1=player_a, 
                        player_2=player_b,
                        n_games=1, 
                        data_shape=data_shape, 
                        render=args.render
                    )
                    seed = int(time.time())
                    player_a.reset(seed)
                    player_b.reset(seed)
            else:
                episode_len, winners = collect_self_play_data(
                    buffer, 
                    env, 
                    mcts_player, 
                    n_games=1, 
                    data_shape=data_shape, 
                    render=args.render
                )
                total_winners = total_winners + winners
            line = f"batch_idx:{i + 1}, episode_len:{episode_len}, tie: {total_winners[0]}, black: {total_winners[1]}, white: {total_winners[2]}, "

            if len(buffer) > train_args['batch_size']:
                loss, entropy = policy_update(
                    model,
                    buffer,
                    train_args,)
                line += f"loss: {loss:.4f}, entropy: {entropy:.4f}\n"
            else:
                line += f"loss: {0:.4f}, entropy: {0:.4f}\n"

            with open("log.txt", "a+") as f:
                f.write(line)
                
            if (i + 1) % train_args["log_freq"] == 0:
                valid_mcts_player = MCTSZeroAgent(model.forward, player_id=1, n_playout=1200)
                test_mcts_player = MCTSZeroAgent(test_model.forward, player_id=2, n_playout=1200)
                winners = [0, 0, 0]
                for _ in range(5):
                    winner, *_ = start_play(env, valid_mcts_player, test_mcts_player, render=True)
                    if winner == -1:
                        winner = 0
                    winners[winner] += 1
                if args.log_to_wandb:
                    wandb.log({
                        "tie": winners[0],
                        "win": winners[1],
                        "lose": winners[2],
                    }) 
                model_name = f"{pargs['project_name']}-{pargs['run_name']}-iteration-{i+1}.pth"
                model.save_checkpoint(os.path.join(pargs['save_dir'], model_name))
            
            if i + 1 > 3000:
                model.set_learning_rate(0.0001)
             
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
            
            
def create_args():
    from rl_gomoku.utils import default_args
    model_args, train_args = default_args()
    project_args = {
        "run_name": datetime.datetime.now().strftime("gomuku-%Y-%m-%d-%H-%M"),
        "project_name": "Lite",
        "save_dir": "./model",
    }
    train_args.update({
        "log_freq": 3000,
        "n_iterations": 50000,
    })
    model_args.update({
        "model_type": "lite",
        "learning_rate": 0.0005,
        "device": "cuda:0",
    })
    return project_args, model_args, train_args


if __name__ == "__main__":
    main()
