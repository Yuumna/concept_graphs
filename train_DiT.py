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
from Network.transformer_pixels import VisionTransformer_Pix 
import sys

sys.path.append("./image_tokenization")
from image_tokenization.main import instantiate_from_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--lrate', default=1e-4, type=float)
parser.add_argument('--test_size', default=1.6, type=float)
parser.add_argument('--alpha', default=1500, type=int)
parser.add_argument('--beta', default=2.0, type=float)
parser.add_argument('--num_samples', default=5000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_T', default=500, type=int)
parser.add_argument('--n_feat', default=256, type=int)
parser.add_argument('--n_sample', default=64, type=int)
parser.add_argument('--n_epoch', default=100, type=int)
parser.add_argument('--experiment', default="H32-train1", type=str)
parser.add_argument("--label", default="None", type=str)
parser.add_argument('--remove_node', default="None", type=str)
parser.add_argument('--type_attention', default="", type=str)
parser.add_argument('--pixel_size', default=28, type=int)
parser.add_argument('--dataset', default="single-body_2d_3classes", type=str)
parser.add_argument('--our_labels', default= False, type=bool)
parser.add_argument('--scheduler', default="", type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--token_folder', type=str, default="")#"/work/dlclarge2/aliy-maskgit/maskgit/image_tokenization/vqgan_logs/2025-02-13T13-25-14_codebook_Third_synthetic_DLC13913381")#2025-02-13T18-09-06_codebook_10244_synthetic_DLC25267020")
parser.add_argument('--model', type=str, default="DiT", choices=["DiT", "U-Net"])




class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


def l2norm(t):
    return F.normalize(t, dim = -1)





# class EmbedFCID(nn.Module):
#     def __init__(self, input_dim, emb_dim):
#         super(EmbedFCID, self).__init__()
#         '''
#         generic one layer FC NN for embedding things  
#         '''
#         self.input_dim = input_dim
#         layers = [
#             nn.Embedding(input_dim, emb_dim),
#             nn.GELU(),
#             nn.Linear(emb_dim, emb_dim),
#         ]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)



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
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_classes=256, flag_weight=0):
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

        #self.feature_proj = nn.Linear(transformer_hidden_dim, self.n_classes)

    def forward(self, x, c):
        """
        Forward pass with discrete token diffusion.
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)


        #b- noise = torch.randint_like(x, low=0, high=self.n_classes).to(self.device)
        noise = torch.randn_like(x)
        #b- x_t = torch.where(torch.rand_like(x.float()) < 0.7, x, noise)  
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

        pred_noise =  self.nn_model(x_t, _ts / self.n_T, c)
        
        return self.loss_mse(noise, pred_noise)
        #return loss#self.loss_mse(pred_tokens.float(), x.float())  

    def sample(self, n_sample, c_gen, size, device, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        #x_i = F.pad(x_i, (1, 0, 1, 0), value=0)
        _c_gen = [tmpc_gen[:n_sample].to(device) for tmpc_gen in c_gen.values()] 

        #context_mask = torch.zeros_like(_c_gen[0]).to(device)

        x_i_store = [] 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, t_is, _c_gen) #, context_mask)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i%20==0:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def ddim_step(self, x_t, t, noise_pred):
        """
        DDIM step to predict the next state of the image.
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_t_1 = torch.where(t > 0, self.alphas_cumprod[t-1], torch.tensor(1.0).to(self.device))
        sigma_t = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_1))
        alpha_t = alpha_t.view(-1,1,1,1)
        sigma_t = sigma_t.view(-1,1,1,1)
        alpha_t_1 = alpha_t_1.view(-1,1,1,1)
    
        x_0_pred = (x_t - sigma_t * noise_pred) / torch.sqrt(alpha_t)
        x_t_1 = torch.sqrt(alpha_t_1) * x_0_pred + sigma_t * torch.randn_like(x_t)
        return x_t_1
    
    def sample_ddim(self, n_sample, c_gen, size, device):
        """
        Sample using the DDIM scheduler.
        """
        x_t = torch.randn(n_sample, *size).to(device)  # Initialize with noise
        
        _c_gen = {k: v.to(device) for k, v in c_gen.items()}

        x_i_store = [] 
        for i in reversed(range(0, self.n_T)):
            print(f'sampling timestep {i}',end='\r')
            t = torch.full((n_sample,), i, device=device, dtype=torch.long)
            noise_pred = self.nn_model(x_t, _c_gen, t.float() / self.n_T)
            x_t = self.ddim_step(x_t, t, noise_pred)

            if i%20==0:
                x_i_store.append(x_t.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_t, x_i_store

    def forward_diff(self, x, t):
        """
        this method is used in training, so samples t and noise randomly
        """
        #t is in range 0 to 1 convert it to 1 to n_T
        _ts = (t * self.n_T).long().to(self.device)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

def training(args):

    n_epoch = args.n_epoch 
    batch_size = args.batch_size 
    n_T = args.n_T 
    n_feat = args.n_feat 
    lrate = args.lrate 
    alpha = args.alpha
    beta = args.beta
    test_size = args.test_size
    dataset = args.dataset 
    our_labels = args.our_labels
    num_samples = args.num_samples 
    pixel_size = args.pixel_size
    experiment = args.experiment 
    label = args.label
    n_sample = args.n_sample 
    type_attention = args.type_attention 
    remove_node = args.remove_node 
    seed = args.seed
    scheduler = args.scheduler
    token_folder = args.token_folder
    model = args.model
    in_channels = 3 if "celeba" in dataset else 3


    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    """
    # loading tokenizer
    with torch.no_grad():
        vqgan_cfg_path = [p for p in os.listdir(os.path.join(token_folder, "configs")) if p.endswith("project.yaml")][0]
        vqgan_ckpt_path = os.path.join(token_folder, "checkpoints", f"last.ckpt")
        vqgan = instantiate_from_config(OmegaConf.load(os.path.join(token_folder, "configs", vqgan_cfg_path)).model).eval().cuda()
        vqgan.load_state_dict(torch.load(vqgan_ckpt_path)["state_dict"]) 
        vqgan = vqgan.eval()
        #set grads of vqgan to false
        for param in vqgan.parameters():
            param.requires_grad = False
    """
    with open("config_category.json", 'r') as f:
         configs = json.load(f)[experiment]


    experiment_classes = {
        "H42-train1": [2, 3, 1, 1],
        "H22-train1": [2, 2],
        "default": [2, 3, 1],
    }
    n_classes = experiment_classes.get(experiment, experiment_classes["default"]) if not our_labels else [2, 2, 2]
    if "celeba" in dataset:
        n_classes = [2,2,2]

    tf = transforms.Compose([transforms.Resize((pixel_size,pixel_size)), transforms.ToTensor()])

    # log the timestamp
    now = datetime.datetime.now().strftime("%d-%m-%H-%M")

    label = "discrete" if our_labels else "continuous"
    space = "latent" if token_folder else "pixel"
    
    save_dir = './output/'+dataset+'/'+model+ '/' + space + '/'+ label+ '/'+ experiment+'/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    
    save_dir = save_dir +str(now) + "_"+ str(num_samples) + "_" + str(test_size) + "_" + str(n_feat) + "_" + str(n_T) + "_" + str(n_epoch) \
                        + "_" + str(lrate) + "_" + remove_node + "_" + str(alpha) + "_" + str(beta) + "_" + str(seed) + "/" #+ str(type_attention) + "/"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    mask_transf = VisionTransformer_Pix(img_size=28 , nclass=len(n_classes), depth=7, heads=8, mlp_dim=1040, dropout=0.1, codebook_size=256, qk_norm=True, ignore_attn_to_source=False)
    ddpm = DDPM(nn_model=mask_transf, betas=(lrate, 0.02), n_T=n_T, device=device, drop_prob=0.1, n_classes=n_classes)
    ddpm.to(device)

    #mask_transf = MaskTransformer(img_size=28, codebook_size=(28*28*3)+1)
   
    
    train_dataset = load_dataset.my_dataset(tf, num_samples, dataset, configs=configs["train"], training=True, alpha=alpha, remove_node=remove_node, our_labels=our_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


    test_dataloaders = {}
    log_dict = {'train_loss_per_batch': [],
                'test_loss_per_batch': {key: [] for key in configs["test"]}}
    output_configs = list(set(configs["test"] + configs["train"])) 
    for config in output_configs: 
        test_dataset = load_dataset.my_dataset(tf, n_sample, dataset, configs=config, training=False, test_size=test_size, our_labels=our_labels)
        test_dataloaders[config] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_dataloader)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)

            #x = 2 * x - 1
            _c = [tmpc.to(device) for tmpc in c.values()]
            """
            with torch.no_grad():
                #normalize x to 0-1 
                #x = x/x.max()
                #emb, _, [_, _, code] = vqgan.encode(2*x-1)
                #emb = F.pad(emb, (1, 0, 1, 0), value=0)
                #code = code.reshape(x.size(0), args.num_tokens, args.num_tokens)
            """
            #ddpm.forward_diff(emb, torch.tensor([0.0]).to(device))
            loss = ddpm(x, _c)
            log_dict['train_loss_per_batch'].append(loss.item())
            loss.backward()
            loss_ema = loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        

        ddpm.eval()
        with torch.no_grad():

            for test_config in configs["test"]: 
                for test_x, test_c in test_dataloaders[test_config]:
                    test_x = test_x.to(device)
                    _test_c = [tmptest_c.to(device) for tmptest_c in test_c.values()]
                    #emb_test, _, [_, _, code] = vqgan.encode(test_x*2-1)
                    #emb_test = F.pad(emb_test, (1, 0, 1, 0), value=0)
                    #code_test = code_test.reshape(test_x.size(0), 7, 7)
                    test_loss = ddpm(test_x, _test_c)
                    log_dict['test_loss_per_batch'][test_config].append(test_loss.item())

            if (ep + 1) % 100 == 0 or ep >= (n_epoch - 5): 
                for test_config in output_configs: 
                    x_real, c_gen = next(iter(test_dataloaders[test_config]))
                    x_real = x_real[:n_sample].to(device)
                    if scheduler=="DDIM":
                        x_gen, x_gen_store = ddpm.sample_ddim(n_sample, c_gen, (in_channels ,pixel_size, pixel_size), device)
                        # how to remove by index the last element of x_gen F.pad(emb_test, (1, 0, 1, 0), value=0)
                    else:
                        #c_gen_[tmpc.to(device) for tmpc in c.values()]
                        x_gen, x_gen_store = ddpm.sample(n_sample, c_gen, (in_channels ,pixel_size, pixel_size), device, guide_w=0.0)

                    np.savez_compressed(save_dir + f"image_"+test_config+"_ep"+str(ep)+".npz", x_gen=x_gen.detach().cpu().numpy()) 
                    print('saved image at ' + save_dir + f"image_"+test_config+"_ep"+str(ep)+".png")

                    if ep + 1 == n_epoch: 
                        np.savez_compressed(save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz", x_gen_store=x_gen_store)
                        print('saved image file at ' + save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz")


            if (ep + 1) == n_epoch:
                with open(save_dir + f"training_log_"+str(ep)+".json", "w") as outfile:
                    json.dump(log_dict, outfile)



if __name__ == "__main__":
    args = parser.parse_args()
    training(args)


