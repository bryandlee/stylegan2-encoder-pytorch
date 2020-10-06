import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from tqdm import tqdm

from model import Encoder, Generator, Discriminator
from dataset import MultiResolutionDataset

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

    
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

    
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss 


def train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device):
    loader = sample_data(loader)

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    vgg_loss = VGGLoss(device=device)

    accum = 0.5 ** (32 / (10 * 1000))

    requires_grad(generator, False)
    
    truncation = 0.7
    trunc = generator.mean_latent(4096).detach()
    trunc.requires_grad = False
    
    if SummaryWriter and args.tensorboard:
        logger = SummaryWriter(logdir='./checkpoint')    
    
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        # D update
        requires_grad(encoder, False)
        requires_grad(discriminator, True)
        

        real_img = next(loader)
        real_img = real_img.to(device)
        
        latents = encoder(real_img)
        recon_img, _ = generator([latents],
                                 input_is_latent=True,
                                 truncation=truncation,
                                 truncation_latent=trunc,
                                 randomize_noise=False)

        recon_pred = discriminator(recon_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, recon_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # E update
        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        real_img = real_img.detach()
        real_img.requires_grad = False

        latents = encoder(real_img)
        recon_img, _ = generator([latents], 
                                 input_is_latent=True,
                                 truncation=truncation,
                                 truncation_latent=trunc,
                                 randomize_noise=False)

        recon_vgg_loss = vgg_loss(recon_img, real_img)
        loss_dict["vgg"] = recon_vgg_loss * args.vgg

        recon_l2_loss = F.mse_loss(recon_img, real_img)
        loss_dict["l2"] = recon_l2_loss * args.l2
        
        recon_pred = discriminator(recon_img)
        adv_loss = g_nonsaturating_loss(recon_pred) * args.adv
        loss_dict["adv"] = adv_loss

        e_loss = recon_vgg_loss + recon_l2_loss + adv_loss 
        loss_dict["e_loss"] = e_loss

        
        encoder.zero_grad()
        e_loss.backward()
        e_optim.step()

        e_loss_val = loss_dict["e_loss"].item()
        vgg_loss_val = loss_dict["vgg"].item()
        l2_loss_val = loss_dict["l2"].item()
        adv_loss_val = loss_dict["adv"].item()
        d_loss_val = loss_dict["d"].item()
        r1_val = loss_dict["r1"].item()

        pbar.set_description(
            (
                f"e: {e_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; l2: {l2_loss_val:.4f}; adv: {adv_loss_val:.4f}; d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
            )
        )

        if SummaryWriter and args.tensorboard:
            logger.add_scalar('E_loss/total', e_loss_val, i)
            logger.add_scalar('E_loss/vgg', vgg_loss_val, i)
            logger.add_scalar('E_loss/l2', l2_loss_val, i)
            logger.add_scalar('E_loss/adv', adv_loss_val, i)
            logger.add_scalar('D_loss/adv', d_loss_val, i)
            logger.add_scalar('D_loss/r1', r1_val, i)            
        
        if i % 100 == 0:
            with torch.no_grad():
                sample = torch.cat([real_img.detach(), recon_img.detach()])
                utils.save_image(
                    sample,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )

        if i % 10000 == 0:
            torch.save(
                {
                    "e": encoder.state_dict(),
                    "d": discriminator.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },
                f"checkpoint/encoder_{str(i).zfill(6)}.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--g_ckpt", type=str, default=None)
    parser.add_argument("--e_ckpt", type=str, default=None)

    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--vgg", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--adv", type=float, default=0.05)   
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)

    parser.add_argument("--tensorboard", action="store_true")
    
    args = parser.parse_args()

    device = args.device
    
    args.start_iter = 0

    print("load generator:", args.g_ckpt)
    g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
    g_args = g_ckpt['args']
    
    args.size = g_args.size
    args.latent = g_args.latent
    args.n_mlp = g_args.n_mlp
    args.channel_multiplier = g_args.channel_multiplier
    
    encoder = Encoder(args.size, args.latent).to(device)
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    
    generator.load_state_dict(g_ckpt["g_ema"])
    discriminator.load_state_dict(g_ckpt["d"])
    d_optim.load_state_dict(g_ckpt["d_optim"])
    
    if args.e_ckpt is not None:
        print("resume training:", args.e_ckpt)
        e_ckpt = torch.load(args.e_ckpt, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(e_ckpt["e"])
        e_optim.load_state_dict(e_ckpt["e_optim"])
        discriminator.load_state_dict(e_ckpt["d"])
        d_optim.load_state_dict(e_ckpt["d_optim"])
        
        try:
            ckpt_name = os.path.basename(args.e_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.split('_')[-1])[0])
        except ValueError:
            pass     

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.data, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True),
        drop_last=True,
    )

    train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device)
