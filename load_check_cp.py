from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import load_dataset
import os
import glob
import json
import argparse
from einops import rearrange, repeat, reduce, pack, unpack
import math
import random
import datetime
from omegaconf import OmegaConf
import sys
sys.path.append("./image_tokenization")
from image_tokenization.main import instantiate_from_config
from Network.test import VisionTransformerTime
from Network.dit_y_emb import DiT_models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_classes=None, flag_weight=0):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.n_classes = n_classes

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.betas = torch.linspace(betas[0], betas[1], n_T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.n_T = n_T
        self.device = device
        self.flag_weight = flag_weight
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        c_drop = []
        for conept in c:
            if random.random() < self.drop_prob:
                print(f"drop conept {conept}")
                c_drop.append(torch.zeros_like(conept))
            else:
                c_drop.append(conept)
        #print(f"noise shape: {noise.shape},x_t shape : {x_t.shape},")#original: {self.nn_model(x_t, c, _ts / self.n_T).shape}")
        return self.loss_mse(noise, self.nn_model(x_t, c_drop, _ts / self.n_T)) #, context_mask))


kwargs ={}
kwargs["discrete"] = False
kwargs["n_classes"] = [2, 3, 1]
loaded_model = DiT_models['concept_DIT_1'](class_dropout_prob= 0.0, **kwargs)
ddpm = DDPM(nn_model=loaded_model, 
                                    betas=(1e-4, 0.02), n_T=500, device='cpu', drop_prob=0.0, n_classes=[2,3,1])

print(
    f"ddpm model: {ddpm.nn_model},\n",
)
ddpm.load_state_dict(torch.load("results/output_all/single-body_2d_3classes/DiT/latent/continuous/XAI/H32-train1/25-04-14-10_5000_1.6_256_500_6000_0.0001_None_1500_2.0_1/checkpoints/ddpm_model_ep3799.pth", map_location='cpu'))

print("load model successfully")