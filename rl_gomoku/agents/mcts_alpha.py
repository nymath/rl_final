import numpy as np
import copy
from typing import Callable, Tuple, Dict
from ..protocols import _env_t
from ..envs.array_gomoku import ArrayGomoku


def idx_to_position(idx, size=15):
    return idx // size, idx % size

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


PolicyValue = Callable[[ArrayGomoku], Tuple[np.ndarray, float]]


class Node(object):
    def __init__(self, parent: "Node", prior_p, action: int = None):
        self.parent: "Node" = parent
        self.children: Dict[int, "Node"]= {}  # a map from action to TreeNode
        self.N: int = 0
        self.Q: float = 0.
        self.P: float = prior_p
        self.action: int = action
        
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob, action)

    def select(self, c_puct) -> tuple[int, "Node"]:
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, reward):
        self.N += 1
        # (n+1)s_{n+1} = n * s_n + x_{n+1}
        self.Q += 1.0 * (reward - self.Q) / self.N

    def update_recursive(self, leaf_value):
        # If it is not root, this node's parent should be updated first.
        # 感觉更新顺序似乎没有影响
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        confident_bound = self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + c_puct * confident_bound

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
    
    def __repr__(self):
        return f"""Node(action: {idx_to_position(self.action)}, Q: {self.Q}, N: {self.N}, P: {self.P})"""


class MCTSZero(object):
    def __init__(
            self,
            policy_value_fn: PolicyValue,
            c_puct=5,
            n_playout=2000
    ):
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.checkpoint = {}

    def reset(self):
        self.update(-1)
        self.checkpoint = {}
        
    def _playout(self, env: ArrayGomoku):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            env.step(action)

        action_probs, leaf_value = self.policy(env)
        # Check for end of game.
        # if leaf_value > 0.99, then we
        lose_prob = leaf_value / 2 + 0.5 
        # if lose_prob < 0.01:
        #     self._do = action
            
        end, winner = env.terminated()  # end 一定是由node造成的, 但是current_layer 在 state.do_move时已经进行切换了

        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = -1
        # we do not rollout, insteat we just step one time and then use nn to summarize the reward afterwards.
        # Update value and visit count of nodes in this traversal.
        # here we input -reward, since the reward is obtained by the child of node;
        node.update_recursive(-leaf_value)

    def get_move_probs(self, env: ArrayGomoku, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        self._do = None
        for n in range(self.n_playout):
            # do plannings
            env.save_checkpoint()
            # 在playout时, node没有变化
            self._playout(env)
            env.load_checkpoint()
            
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.N)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        # 看看temp的影响
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update(self, action):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0, action)

    def get_action(self, env: ArrayGomoku, temp=1e-3, return_prob=0, is_selfplay=False):
        legal_moves = env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        if env.last_move != -1:
            self.update(env.last_move)
            
        move_probs = np.zeros(env.width * env.height)
        if len(legal_moves) > 0:
            acts, probs = self.get_move_probs(env, temp)
            move_probs[list(acts)] = probs

            if is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.update(move)
            else:
                move = np.random.choice(acts, p=probs)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")


class MCTSZeroAgent(object):
    def __init__(
            self,
            policy_value_function,
            player_id=1,
            c_puct=5,
            n_playout=2000,
    ):
        self.mcts = MCTSZero(policy_value_function, c_puct, n_playout)
        self.player_id = player_id

    def reset(self):
        self.mcts.update(-1)
        self.mcts.checkpoint = {}

    def get_action(self, env: ArrayGomoku, temp=1e-3, return_prob=0, is_selfplay=False):
        legal_moves = env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper

        # if env.last_move != -1:
        #     self.mcts.update(env.last_move)
            
        move_probs = np.zeros(env.width * env.height)
        if len(legal_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            move_probs[list(acts)] = probs
            if is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.95*probs + 0.05*np.random.dirichlet(0.1*np.ones(len(probs)))
                )
                self.mcts.update(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
