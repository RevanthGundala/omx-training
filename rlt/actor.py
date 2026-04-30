import torch 
import torch.nn as nn
import torch.nn.functional as F

# Lightweight actor network 
class Actor(nn.Module):
    def __init__(
            self,
            act_dim: int,
            act_horizon: int,
            rl_token_dim: int, 
            config, 
    ):
        super(Actor, self).__init__()
        self.act_dim = act_dim 
        self.act_horizon = act_horizon
        self.rl_token_dim = rl_token_dim

        self.in_proj = nn.Linear()