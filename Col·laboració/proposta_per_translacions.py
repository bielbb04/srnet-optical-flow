# -*- coding: utf-8 -*-

import os
import torch
from einops import rearrange
from torch import nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from random import uniform
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import conv2d, interpolate
import math
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import skimage.registration
from torchvision.utils import flow_to_image
import torch.nn.functional as F
import imageio
import torchvision.transforms.functional as TF
import copy
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import argparse


from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

def warp_image(img: Tensor,
         flow: Tensor,
         padding_mode = "border",
         interpolation: str = "bilinear",
         align_corners: bool = False) -> Tensor:

    device = img.device
    if len(flow.shape) == 3:
        flow = flow.unsqueeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape

    xs = torch.arange(w, dtype=torch.float, device=flow.device).view(
        1, 1, 1, -1).expand(b, 1, h, w)
    ys = torch.arange(h, dtype=torch.float, device=flow.device).view(
        1, 1, -1, 1).expand(b, 1, h, w)
    grid = torch.cat((xs, ys), dim=1).permute(0, 2, 3, 1)

    #print("Average flow:", torch.mean(flow), flow.shape)
    if flow.shape[-1] != 2:
        grid += flow.permute(0, 2, 3, 1)
    else:
        grid += flow

    #print(grid.shape)
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / (w - 1) - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / (h - 1) - 1.0
    grid = grid.to(device)
    warped_img = F.grid_sample(img, grid, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return warped_img

def read_image(img_path):
    img_pil = Image.open(img_path)
    return TF.pil_to_tensor(img_pil).to(torch.float)/255.

def generate_random_shift(img):
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        b, c, h, w = img.shape
        flow_fake = torch.zeros(b, 2, h, w)

        flow_fake[:, 0, :, :] = uniform(0, 5)
        flow_fake[:, 1, :, :] = uniform(0, 5)

        # Aplicar warp
        I_1 = warp_image(img, flow_fake, padding_mode='zeros')
        I_1 = I_1.detach()

        return I_1, flow_fake

class Patches_translation(Dataset):
    def __init__( self, path, subset, sampling, patch_size = None):
        super().__init__()
        if patch_size is not None:
            if 200 % patch_size != 0:
                raise ValueError('Patch size should be a divisor of 200')
        if sampling not in [2,4,8]:
            raise ValueError('Sampling factor should be 2, 4 or 8')
        img_path_list = sorted([f'{path}/{subset}/{img}' for img in os.listdir(f'{path}/{subset}') if img.endswith('.png')])

        self.sampling = sampling
        self.patch_size = patch_size
        self.subset = subset
        self.gt, self.shifted, self.flow = self.generate_data(img_path_list)

        #print(f"Shape de gt: {self.gt.shape}")
        #print(f"Shape de shifted: {self.shifted.shape}")

    def __getitem__(self, index) :
        return self.gt[index, ...], self.shifted[index, ...], self.flow[index, ...]

    def __len__(self):
        return len(self.gt)


    def generate_data(self, img_path_list):
        gt_list = []
        shifted_list = []
        flow_list = []

        for i,img_path in enumerate(img_path_list):
            #print(f"\nProcessant imatge {i+1}/{len(img_path_list)}: {img_path}")
            gt = read_image(img_path)
            #print(f"Shape de gt: {gt.shape}")
            if self.patch_size is not None:
                gt = gt.unsqueeze(0)

                gt = nn.functional.unfold(gt, kernel_size=self.patch_size, stride=self.patch_size)

                #print(f'Shape despres de unfold: {gt.shape}')
                gt_fold = nn.functional.fold(gt, 200, kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)

                gt = rearrange(gt, 'b (c p1 p2) n -> (b n) c p1 p2', p1=self.patch_size, p2=self.patch_size)
                #print(f"Shape despres de rearrange: {gt.shape}")

            # 128 parelles
            for _ in range(16):
                gt_list.append(gt)
                shifted_small_list = []
                flow_small_list = []
                for idx in range(len(gt)):
                    shifted_i, flow_i = generate_random_shift(gt[idx])
                    shifted_small_list.append(shifted_i)
                    flow_small_list.append(flow_i)
                shifted = torch.cat(shifted_small_list, dim=0)
                flow = torch.cat(flow_small_list, dim = 0)
                flow_list.append(flow)
                if len(shifted.shape) == 3:
                    shifted = shifted.unsqueeze(0)
                if self.patch_size is not None:
                    shifted = nn.functional.unfold(shifted, kernel_size=self.patch_size, stride=self.patch_size)
                    shifted = rearrange(shifted, 'b (c p1 p2) n -> (b n) c p1 p2', p1=self.patch_size, p2=self.patch_size)
                shifted_list.append(shifted)

        return torch.cat(gt_list, dim=0), torch.cat(shifted_list, dim=0), torch.cat(flow_list, dim = 0)

class ResBlock(nn.Module):
    def __init__(self,  in_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return self.relu(res + x)
class SRNet(nn.Module):
    def __init__(self, sampling, features=64, kernel_size=3, blocks=3, channels=2):
        super(SRNet, self).__init__()
        self.add_features = nn.Conv2d(channels, features, kernel_size=kernel_size, padding=kernel_size//2)
        self.residual = nn.ModuleList([ResBlock(features, kernel_size=kernel_size) for _ in range(blocks)])
        self.obtain_channels = nn.Conv2d(features, channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, low):
        features = self.add_features(low)
        for res_block in self.residual:
            features = res_block(features)
        res = self.obtain_channels(features)
        return res

# Next two function were written by Tomeu Garau
def gradient(u, stack=False):
    # Input should have 4 dimensions: [B, C, H, W]
    if len(u.shape) == 3:
        u = u.unsqueeze(0)
    if len(u.shape) == 2:
        u = u.unsqueeze(0).unsqueeze(0)

    dx = torch.zeros_like(u)
    dy = torch.zeros_like(u)
    dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
    dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
    if stack:
        return torch.stack((dx, dy), dim=-1)
    else:
        return dx, dy

def warp_and_derivative(img, flow):
    grad = gradient(img, stack=True)
    grad_im1_1 = grad[..., 0]
    grad_im1_2 = grad[..., 1]

    grad_warp_im1_1 = warp_image(grad_im1_1, flow)
    grad_warp_im1_2 = warp_image(grad_im1_2, flow)
    # warp_I_1_grad = warp_image(grad, flow)
    warp_I_1 = warp_image(img, flow)
    #print('warp_I_1 shape:', warp_I_1.shape, 'grad_warp_im1_1 shape:', grad_warp_im1_1.shape, 'grad_warp_im1_2 shape:', grad_warp_im1_2.shape)
    warp_I_1_grad = torch.cat([grad_warp_im1_1, grad_warp_im1_2], dim=1) # Modified
    #print('warp_I_1_grad shape:', warp_I_1_grad.shape)
    return warp_I_1, warp_I_1_grad


def rho(I_0, warp_I_1, warp_I_1_grad, u, u_0):
    #print('shape I_0:', I_0.shape, 'shape warp_I_1:', warp_I_1.shape, 'shape warp_I_1_grad:', warp_I_1_grad.shape, 'shape u:', u.shape, 'shape u_0:', u_0.shape)
    prod_esc = warp_I_1_grad * (u - u_0)

    prod_esc = torch.sum(prod_esc, dim=1, keepdim=True)  # Modified to keep the channel dimension
    #print('prod_esc shape:', prod_esc.shape, 'warp_I_1 shape:', warp_I_1.shape, 'I_0 shape:', I_0.shape)
    rho = prod_esc + warp_I_1 - I_0

    return rho


def rho_derivative(I_0, warp_I_1, warp_I_1_grad,  flow_u_k, flow_init):
    rho_flow = rho(I_0, warp_I_1, warp_I_1_grad, flow_u_k, flow_init)

    diff_flow = flow_u_k - flow_init
    # We compute the gradient of rho
    #print('rho_flow shape:', rho_flow.shape, 'warp_I_1_grad shape:', warp_I_1_grad.shape, 'warp_I_1_grad_0 shape:', warp_I_1_grad[:, 0, ...].shape)
    gradient_rho_1 = rho_flow * warp_I_1_grad[:, 0, ...].unsqueeze(1) #* diff_flow[..., 0]
    gradient_rho_2 = rho_flow * warp_I_1_grad[:, 1, ...].unsqueeze(1) #* diff_flow[..., 1]
    #print('gradient_rho_1 shape:', gradient_rho_1.shape, 'gradient_rho_2 shape:', gradient_rho_2.shape)
    gradient_rho = torch.cat([gradient_rho_1, gradient_rho_2], dim=1)

    return rho_flow, gradient_rho

class Logger():
    def __init__(self, dataset, model_name, nickname, path_saving):
        day = datetime.now().strftime("%Y-%m-%d")
        self.model_name = model_name
        self.nickname = nickname
        self.dir_path = f"{path_saving}/checkpoints/{dataset}/{model_name}/{day}/{nickname}" if nickname is not None else f"{path_saving}/checkpoints/{dataset}/{model_name}/{day}"
        self.writer = SummaryWriter(self.dir_path)
        os.makedirs(f"{self.dir_path}/checkpoints", exist_ok=True)
        self.best_loss = float("inf")

    def log_loss(self, epoch, train_loss, validation_loss):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/validation", validation_loss, epoch)
        self.writer.add_scalars("Loss/comparison", {"train": train_loss, "validation": validation_loss}, epoch)

    def log_params(self, num_params):

        self.writer.add_text("Parameters", f'{self.model_name} = {num_params}')

    def save_checkpoints(self, model, epoch, validation_loss):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict()}
        torch.save(ckpt, f"{self.dir_path}/checkpoints/last.ckpt")
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            torch.save(ckpt, f"{self.dir_path}/checkpoints/best.ckpt")

    def plot_results(self, epoch, gt, low, high):
        sampling = high.size(2) // low.size(2)
        inter = nn.functional.interpolate(low, scale_factor=sampling, mode="bicubic")
        images = torch.cat([gt, inter, high], dim=3)
        grid = make_grid(images, nrow=1)
        self.writer.add_image("Image comparison: Ref ~ Bic ~ Pred", grid, epoch)

current_dir = os.getcwd()  # Directorio actual en Google Colab
path_biel  = '/content/drive/MyDrive/col·laboració/'

path_using = path_biel
os.environ['PROJECT_ROOT'] = path_using #'/content/drive/MyDrive/col·laboració/'


class Args(argparse.Namespace): 
    data = './data/penn'
    model = 'LSTM'
    emsize = 200
    nhid = 200
    parser = argparse.ArgumentParser(description="Train script")
    sampling = 2
    dataset_path = f"{path_using}"

    iter_warp = 5
    iter_prox = 15
    landa = 0.5
    eta = 0.25

    kernel_size = 3
    features = 128
    blocks = 3
    batch_size = 4
    epochs = 50
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    patch_size = 50
    nickname = None

def train(args):

    device = args.device
    # Definim el model
    model = SRNet(sampling=args.sampling,kernel_size=args.kernel_size, features=args.features, blocks=args.blocks).to(device)
    # Carragam els dataloader
    train_dataset = Patches_translation(sampling=args.sampling, subset="train", path=args.dataset_path, patch_size=args.patch_size)
    #train_dataset = Init(sampling=args.sampling, subset="", path=args.dataset_path, patch_size=args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=8)
    # Definim la funció de perdua i l'optimitzador
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    # Instanciam la classe que emplearem per monitoritzar els resultas i guardar els pesos
    logger = Logger("Init", "SRNet", args.nickname, path_using)

    # Començam l'entrenament

    max_epochs = args.epochs
    total_validataion_loss = float("inf")
    num_param = sum(p.numel() for p in model.parameters())
    logger.log_params(num_param)
    losses = []
    for epoch in range(max_epochs):
        # Fem una passada en tot el conjunt de train dividint les imatges en batches per a que hi capiga en memòria:
        model.train()
        total_train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Training]")
        #train_loader_tqdm.set_postfix({"Validation loss": total_validataion_loss})
        for idx, batch in enumerate(train_loader_tqdm):
            optimizer.zero_grad()
            img, shifted, flow = batch
            img = img.to(device)
            shifted = shifted.to(device)
            flow = flow.to(device)
            batch_len = img.size(0)

            u_0 = torch.zeros((batch_len, 2, args.patch_size, args.patch_size)).to(device)
            u_k = torch.zeros((batch_len, 2, args.patch_size, args.patch_size)).to(device)
            # Minimize flow using unfolding algorithm
            for num_warp in range(args.iter_warp):
                warped_I1, der_warped_I1 = warp_and_derivative(shifted, u_0)
                # Aplicam el model a la entrada del proximity
                rho_var, deriv = rho_derivative(img, warped_I1, der_warped_I1, u_k, u_0)
                for iter in range(args.iter_prox):
                    x = u_k + args.landa * args.eta * rho_var * deriv
                    u_k = model(x)

                u_0 = u_k.detach()
            # Calcular la loss entre el resultat i la nostra dada de referència
            # predicted = warp_image(img, u_k)
            loss = loss_function(flow, u_k)
            print(f"Loss: {loss.item():.6f}")
            #print(f"u_k min/max: {u_k.min().item():.3f}/{u_k.max().item():.3f}")
            # Feim el gradient amb back-propagation
            loss.backward()
            # Actualitzam els paràmetres amb un pas de l'optimizador
            optimizer.step()
            total_train_loss += batch_len*loss.item() / len(train_loader.dataset)

        # Al finalitzar cada epoch monitoritzam la loss i les imatges i també guardam els pesos
        logger.log_loss(epoch, total_train_loss, total_validataion_loss)
        logger.save_checkpoints(model, epoch, total_train_loss)

        losses.append(total_train_loss)

        print("mitja del fluxe real", flow.mean().item())
        print("mitja del fluxe resultant", u_k.mean().item())

        img_flow_resultant = flow_to_image(u_k.detach().cpu())[0]
        img_flow_resultant = img_flow_resultant.permute(1, 2, 0)

        print("shape del flow real", flow.shape)
        img_true_flow = flow_to_image(flow.detach().cpu())[0]
        img_true_flow = img_true_flow.permute(1,2,0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        im1 = ax1.imshow(img_flow_resultant)
        ax1.set_title("fluxe", fontsize=14, fontweight='bold', pad=20)
        ax1.axis('off')

        im2 = ax2.imshow(img_true_flow)
        ax2.set_title("fluxe Real", fontsize=14, fontweight='bold', pad=20)
        ax2.axis('off')

        plt.tight_layout()

        fig.suptitle('Comparació de fluxes', fontsize=16, fontweight='bold', y=1.02)

        plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument('--sampling', type=int, required=True, help='Sampling factor')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")

    parser.add_argument("--kernel_size", type=int,default=3,help="Kernel size for 2d convoltion")
    parser.add_argument("--features", type=int, default=128, help="Number of features for residual blocks")
    parser.add_argument("--blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch dimension to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train the model")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size to train the model")
    parser.add_argument("--nickname", type=str, default=None, help="Nickname for save in tensorboard")
    #args = parser.parse_args()
    args = Args()
    train(args)

img = read_image(f'{path_using}train/init.i1.png').unsqueeze(0)
model = SRNet(sampling=2,kernel_size=3, features=128, blocks=3)
model_path = f'{path_using}checkpoints/Init/SRNet/2025-10-06/checkpoints/best.ckpt'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

shifted, real_flow = generate_random_shift(img)

u_0 = torch.zeros((1, 2, 200, 200))
u_k = torch.zeros((1, 2, 200, 200))
# Minimize flow using unfolding algorithm
for num_warp in range(args.iter_warp):
    warped_I1, der_warped_I1 = warp_and_derivative(shifted, u_0)
    # Aplicam el model a la entrada del proximity
    rho_var, deriv = rho_derivative(img, warped_I1, der_warped_I1, u_k, u_0)
    for iter in range(args.iter_prox):
        x = u_k + args.landa * args.eta * rho_var * deriv
        u_k = model(x)

    u_0 = u_k

print("mitja del fluxe resultant:",u_k.mean())
print("mitja del fluxe real:",real_flow.mean())

# Convertimos el flujo predicho a imagen
flow_image_pred = flow_to_image(u_k.cpu())  # (H, W, 3)
flow_image_real = flow_to_image(real_flow.cpu())   # (H, W, 3)

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Flujo predicho
ax1.imshow(flow_image_pred.squeeze(0).permute(1, 2, 0))
ax1.set_title("Flow resultant")
ax1.axis('off')

# Flujo real
ax2.imshow(flow_image_real.squeeze(0).permute(1, 2, 0))
ax2.set_title("Flujo Real")
ax2.axis('off')

plt.tight_layout()
fig.suptitle('Comparació de fluxes aplicats a tota la imatge', fontsize=16, fontweight='bold', y=1.02)
plt.show()