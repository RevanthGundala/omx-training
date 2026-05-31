import math
from collections import deque

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights, resnet18

from .attention import TransformerBlock


def sinusoidal_positional_encoding(length, dim, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class ACT(nn.Module):
    def __init__(
            self,
            d_model,
            d_qpos, 
            d_z,
            chunk_size,
            device,
            num_cameras=2,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_heads=8,
            mlp_dim=2048,
            dropout=0.1,
            max_steps=1000,
    ):
        super(ACT, self).__init__()
        self.image_encoder = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
        self.device = device
        self.d_z = d_z
        self.num_cameras = num_cameras
        self.d_model = d_model

        self.image_proj = nn.Linear(512, d_model)
        self.camera_emb = nn.Parameter(torch.zeros(1, num_cameras, d_model))
        self.joint_proj = nn.Linear(d_qpos, d_model)
        self.z_proj = nn.Linear(d_z, d_model)

        self.cvae_encoder = Encoder(
            d_model,
            d_qpos,
            d_z,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.cvae_decoder = Decoder(
            d_model,
            d_qpos,
            chunk_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # eval/inference settings
        self.max_steps = max_steps
        self.buffer_dict = { t: deque(maxlen=chunk_size) for t in range(max_steps) }

    def forward(self, images, qpos, future_qpos = None, action_mask=None):
        # preprocess images 
        B, N, C, H, W = images.shape
        if N != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {N}")

        images = images.reshape(B * N, C, H, W)
        image_features = self.image_encoder(images)
        spatial_tokens = image_features.shape[2] * image_features.shape[3]
        image_features = image_features.flatten(2).permute(0, 2, 1)
        image_features = self.image_proj(image_features)
        image_features = image_features.reshape(B, N, spatial_tokens, self.d_model)
        pos_emb = sinusoidal_positional_encoding(spatial_tokens, self.d_model, image_features.device)
        image_features = image_features + pos_emb.view(1, 1, spatial_tokens, self.d_model) + self.camera_emb.unsqueeze(2)
        image_features = image_features.reshape(B, N * spatial_tokens, self.d_model)

        mu, log_var = None, None
        if future_qpos is not None: 
            z, mu, log_var = self.cvae_encoder(qpos, future_qpos, action_mask=action_mask)
        else:
            z = torch.zeros(B, self.d_z, device=qpos.device)

        joint_token = self.joint_proj(qpos).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)

        decoder_input = torch.cat([image_features, joint_token, z_token], dim=1)
        act_pred = self.cvae_decoder(decoder_input)
        return act_pred, mu, log_var
    
    @torch.inference_mode()
    def select_action(self, timestep, images, qpos):
        act_pred, _, _ = self.forward(images, qpos)
        self.buffer_dict[timestep].append(act_pred)
        candidates = list(self.buffer_dict[timestep])
        
        weights = torch.exp(-0.01 * torch.arange(len(candidates), device=act_pred.device, dtype=act_pred.dtype))
        weights = weights / weights.sum()
        stacked = torch.stack(candidates, dim=0)
        return (stacked * weights.view(-1, 1, 1)).sum(dim=0).squeeze(0)

class Encoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            d_qpos,
            d_z,
            num_layers=4,
                num_heads=8,
                mlp_dim=2048,
                dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.qpos_proj = nn.Linear(d_qpos, d_model)
        self.action_proj = nn.Linear(d_qpos, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, d_z * 2)

    def forward(self, qpos, future_qpos, action_mask=None):
        B = qpos.shape[0]
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)
        action_tokens = self.action_proj(future_qpos)
        pos_emb = sinusoidal_positional_encoding(action_tokens.shape[1], self.cls_token.shape[2], self.cls_token.device)
        action_tokens = action_tokens + pos_emb.unsqueeze(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, qpos_token, action_tokens], dim=1)

        future_mask = None
        if action_mask is not None:
            # joint token + cls mask + future action mask
            prefix_mask = torch.ones(B, 2, dtype=torch.bool, device=action_mask.device)
            future_mask = torch.cat([prefix_mask, action_mask.bool()], dim=1)
            future_mask = future_mask[:, None, None, :]

        for block in self.blocks:
            x = block(x, future_mask=future_mask)
        cls_output = x[:, 0]
        z_params = self.out_proj(cls_output)
        mu, log_var = z_params.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(
            self,
            d_model,
            d_qpos,
            chunk_size,
            num_layers=4,
                num_heads=8,
                mlp_dim=2048,
                dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.q_emb = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=True,
            ) for _ in range(num_layers)
        ])
        self.act_proj = nn.Linear(d_model, d_qpos)

    def forward(self, x):
        B = x.size(0)
        q_emb = self.q_emb.expand(B, -1, -1)
        for block in self.blocks:
            q_emb = block(q_emb, encoder_out=x)
        act_pred = self.act_proj(q_emb)
        return act_pred
        