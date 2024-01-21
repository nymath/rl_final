import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .envs import ArrayGomoku
from .agents import MCTSZeroAgent, GreedyAgent
from .common import AlphaZeroBase, ModelInterface
from .protocols import _margs_t, _targs_t, _env_t, _agent_t
from typing import Optional, Union
from typing import TypedDict, Tuple, Protocol, Any


def default_args() -> Tuple[_margs_t, _targs_t]:
    model_args: _margs_t = {
        "model_type": "base",
        "in_channels": 7,
        "drop_out": 0.1,
        "device": "cuda:0",
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "use_attention": True,
    }
    
    training_args: _targs_t = {
        "n_iterations": 5000,
        "buffer_size": 8192,
        "n_playout": 400,
        "board_size": 15,
        "update_times_per_episode": 5,
        "log_freq": 200,
        "batch_size": 256,
    }
    return model_args, training_args


def create_model_from_args(args: _margs_t):
    model = ModelInterface(
        model_type=args["model_type"],
        in_channels=args['in_channels'],
        drop_out=args['drop_out'],
        device=args['device'],
        learning_rate=args['learning_rate'],
        weight_decay=args['weight_decay'],
        use_attention=args['use_attention']
    )
    return model


def ensure_idx(move, size=15):
    if not isinstance(move, tuple):
        return move
    else:
        return move[0] * size + move[1]
    
    
def start_play(
    env: _env_t, 
    player_1: Union[MCTSZeroAgent, GreedyAgent], 
    player_2: Union[MCTSZeroAgent, GreedyAgent], 
    render=False, 
    temp=1e-3
):
    env.reset()

    states, mcts_probs, current_players = [], [], []
    while True:
        if env.current_player == 1:
            move, move_probs = player_1.get_action(env, temp, 1)

        else:
            move, move_probs = player_2.get_action(env, temp, 1)

        move = ensure_idx(move)
        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
        # perform a move
        env.step(move)
        if render:
            env.render()
        terminated, winner = env.terminated()

        if terminated:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0

            try:
                player_1.reset()
                player_2.reset()
            except Exception:
                pass
            finally:
                break
    return winner, zip(states, mcts_probs, winners_z)


def replay_game_from_list(env: _env_t, position_list, render=False):
    env.reset()

    states, mcts_probs, current_players = [], [], []
    for i in range(len(position_list)):
        move = position_list[i]
        move_probs = np.zeros(shape=(225, ))
        move_probs[move] = 1
        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
 
        env.step(move)
        if render:
            env.render()
        terminated, winner = env.terminated()

        if terminated:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            break
        
    return winner, zip(states, mcts_probs, winners_z) 
    

def start_self_play(env: _env_t, player: MCTSZeroAgent, render=False, temp=1e-3):
    """ start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """
    env.reset()
    states, mcts_probs, current_players = [], [], []

    while True:
        move, move_probs = player.get_action(env,
                                             temp=temp,
                                             return_prob=1,
                                             is_selfplay=True)
        # store the data

        states.append(env.current_state())
        mcts_probs.append(move_probs)
        current_players.append(env.current_player)
        # perform a move
        env.step(move)
        if render:
            env.render()

        terminated, winner = env.terminated()
        if terminated:
            # winner from the perspective of the current player of each state
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            # reset MCTS root node
            player.reset()
            if render:
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            break

    return winner, zip(states, mcts_probs, winners_z)


def data_argumentation(play_data, data_shape=(15, 15)):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    # Data Augmentation
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_porb.reshape(*data_shape)), i)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,
                                np.flipud(equi_mcts_prob).flatten(),
                                winner))
    return extend_data


def collect_data(buffer, env: _env_t, player_1, player_2, n_games=1, data_shape=(15, 15), render=False):
    """collect self-play data for training"""
    episode_len = []
    for i in range(n_games):
        winner, play_data = start_play(env, player_1, player_2, render=render)  # TODO: control temp
        play_data = list(play_data)[:]
        episode_len.append(len(play_data))
        play_data = data_argumentation(play_data, data_shape)
        buffer.extend(play_data)

    return sum(episode_len) / n_games


def collect_botzone_data(buffer, env: _env_t, states, data_shape=(15, 15), render=False):
    winner, play_data = replay_game_from_list(env, states)
    play_data = list(play_data)[:]
    episode_len = len(play_data)
    play_data = data_argumentation(play_data, data_shape)
    buffer.extend(play_data)
    return episode_len
 

def collect_self_play_data(buffer, env: _env_t, player_1, n_games=1, data_shape=(15, 15), render=False):
    """collect self-play data for training"""
    episode_len = []
    winners = np.array([0, 0, 0])
    for i in range(n_games):
        winner, play_data = start_self_play(env, player_1, render=render)  # TODO: control temp
        play_data = list(play_data)[:]
        if winner == -1:
            winners[0] += 1
        else:
            winners[winner] += 1
            
        episode_len.append(len(play_data))
        play_data = data_argumentation(play_data, data_shape)
        buffer.extend(play_data)

    return sum(episode_len) / n_games, winners


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr