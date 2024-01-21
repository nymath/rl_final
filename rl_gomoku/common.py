import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = nn.BatchNorm2d(in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x
    
    
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            drop_out=0.1,
            use_attention=False,
    ):
        super().__init__()
        self.drop_out = drop_out
        self.activation = F.relu
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.norm_2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_connection = nn.Identity()

        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x):
        out = self.norm_1(self.conv_1(x))
        out = self.activation(out)
        out = self.norm_2(self.conv_2(out))
        out = self.activation(out + self.residual_connection(x))
        out = self.attention(out)
        return out


class AlphaZeroBase(nn.Module):
    def __init__(self, board_size, in_channels, dropout=0.1, use_attention=False):
        super().__init__()
        self.board_size = board_size

        # shared network
        self.shared_network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64, dropout, use_attention),
            ResidualBlock(64, 128, dropout, use_attention),        
        )

        # policy branch
        self.policy_branch = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*self.board_size*self.board_size, board_size*board_size),
            nn.LogSoftmax(dim=1),  # here we return the normalized logits
        )

        # state-value branch
        self.value_branch = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # bound the value to (-1, 1)
        )

    def forward(self, x):
        state = self.shared_network(x)
        act_logits = self.policy_branch(state)
        act_value = self.value_branch(state)
        return act_logits, act_value


class AlphaZeroLite(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(32, 32, **factory_kwargs),
            ResidualBlock(32, 32, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size**2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output
    

class AlphaZeroTiny(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(32, 32, **factory_kwargs),
            ResidualBlock(32, 32, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size**2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size**2, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output
    
    
class AlphaZero2(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size **2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size **2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output
    

class AlphaZero3(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size **2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size **2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output
    

class AlphaZero5(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() 
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
            ResidualBlock(64, 64, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size **2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size **2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output


class AlphaZero9(nn.Module):
    def __init__(
        self,
        board_size,
        in_channels,
        dropout=0.1,
        use_attention=False, 
    ):
        factory_kwargs = {"drop_out": dropout, "use_attention": use_attention}
        super().__init__()
        self.board_size = board_size
        
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU() 
        )
        
        self.shared_network = nn.ModuleList([
            conv_layer,
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
            ResidualBlock(128, 128, **factory_kwargs),
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size **2, board_size **2),
            nn.LogSoftmax(-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size **2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # bound the value to (-1, 1)
        )

    def forward(self, x):
        for module in self.shared_network:
            x = module(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)

        return policy_output, value_output
    
    
def mcts_train(net: nn.Module, optimizer: torch.optim.Optimizer, data, device="cuda"):
    states = torch.FloatTensor(np.array(data["states"])).to(device)          # (batch_size, 7, 15, 15)
    mcts_probs = torch.FloatTensor(np.array(data["mcts_probs"])).to(device)  # (batch_size, 225)
    winners = torch.FloatTensor(np.array(data["winners"])).to(device)        # (batch_size,)

    act_logits, state_value = net(states)
    value_loss = F.mse_loss(state_value.squeeze(), winners.squeeze())
    policy_loss = - torch.sum(mcts_probs*F.softmax(act_logits), 1)   # the cross entropy
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def actor_critic_train(net: nn.Module, optimizer: torch.optim.Optimizer, transition_dict, device="cuda"):
    # on policy
    pass


def ppo_train(net: nn.Module, optimizer: torch.optim.Optimizer, replay_buffer, device="cuda"):
    pass


class ModelInterface:
    """policy-value network """
    
    _models = {
        "base": AlphaZeroBase,
        "lite": AlphaZeroLite,
        "tiny": AlphaZeroTiny,
        "res2": AlphaZero2,
        "res3": AlphaZero3,
        "res5": AlphaZero5,
        "res9": AlphaZero9,
    }    

    def __init__(
            self,
            model_type: typing.Literal["base", "tiny", "res2", "res3", "res5", "res9"] = "base",
            board_width=15,
            board_height=15,
            device="cuda:0",
            in_channels=7,
            drop_out=0.1,
            learning_rate=0.0002,
            weight_decay=0.0001,
            use_attention=True,
    ):

        self.device = torch.device(device)
        self.factory_kwargs = {"dtype": torch.float32, "device": self.device}

        self.board_width = board_width
        self.board_height = board_height
        self.weight_decay = weight_decay
        
        model_cls = self._models[model_type]
        
        self.model = model_cls(
            board_size=board_height, 
            in_channels=in_channels, 
            dropout=drop_out,
            use_attention=use_attention,
        )
 
        self.model = self.model.to(**self.factory_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)

    @torch.no_grad()
    def policy_value(self, state_batch):
        self.eval()
        state_batch_tensor = torch.tensor(np.array(state_batch), **self.factory_kwargs)
        log_act_probs, value = self.model(state_batch_tensor)
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    @torch.no_grad()
    def forward(self, env):
        self.eval()
        legal_positions = env.availables
        current_state = env.current_state()
        
        current_state = torch.tensor(np.array(current_state)).to(**self.factory_kwargs)
        with torch.no_grad():
            log_act_probs, value = self.model(current_state.unsqueeze(0))

        act_probs = np.exp(log_act_probs.squeeze().cpu().numpy())
        act_probs = act_probs[legal_positions]
        act_probs = act_probs / act_probs.sum()
        act_probs = zip(legal_positions, act_probs)
        return act_probs, value.item()


    def train_step(self, state_batch, mcts_probs, winner_batch):
        self.train()

        state_batch = torch.tensor(np.array(state_batch)).to(**self.factory_kwargs)
        mcts_probs = torch.tensor(np.array(mcts_probs)).to(**self.factory_kwargs)
        winner_batch = torch.tensor(np.array(winner_batch)).to(**self.factory_kwargs)

        log_act_probs, value = self.model(state_batch)

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = - torch.mean(torch.sum(mcts_probs * log_act_probs, 1)) # kl divergence
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

        return loss.item(), entropy.item()

    def load_checkpoint(self, path):
        check_point = torch.load(path)
        self.model.load_state_dict(check_point["model_state_dict"])
        self.optimizer.load_state_dict(check_point["optim_state_dict"])

    def save_checkpoint(self, path):
        """ save model params to file """
        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        torch.save({
            "model_state_dict": model_state_dict,
            "optim_state_dict": optim_state_dict,
        }, path)

    def set_learning_rate(self, lr):
        """Sets the learning rate to the given value"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval() 