import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from rl_gomoku.common import AlphaZeroBase, AlphaZero5, AlphaZero9, AlphaZero3, AlphaZeroTiny, AlphaZeroLite

model_0 = AlphaZeroBase(15, 7, use_attention=True)
model_3 = AlphaZero3(15, 7, use_attention=True)
model_5 = AlphaZero5(15, 7, use_attention=True)
model_9 = AlphaZero9(15, 7, use_attention=True)
model_t = AlphaZeroLite(15, 7, use_attention=True)

summary(model_5)