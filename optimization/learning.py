import click 
import numpy as np 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.data import DataLoader 

from optimization.dataholder import DataHolder 
from optimization.constants import * 
from models.discriminator import MDiscriminators
from models.generator import Generator

from libraries.log import logger 
from libraries.strategies import * 

def save_images(XA, XA_B, path_to):
    XA = XA.cpu()
    XA_B = XA_B.cpu()
    merged_images = th.cat([XA, XA_B], dim=-1)
    merged_images = to_grid(merged_images, nb_rows=1)
    rescaled_merged_images = th2cv(merged_images) * 255
    cv2.imwrite(path_to, rescaled_merged_images)

@click.command()
@click.option('-s', '--source_path', help='path to source data')
@click.option('-e', '--nb_epochs', help='number of epochs', type=int)
@click.option('-b', '--bt_size', help='size of batch', type=int)
def train(source_path, nb_epochs, bt_size):
    device = th.device('cuda:0') if th.cuda.is_available() else 'cpu' 

    source = DataHolder(source_path, '*.jpg', mapper)
    loader = DataLoader(dataset=source, shuffle=True, batch_size=bt_size)

    D_X0 = MDiscriminators(i_dim, n_dim, n_down, n_models).to(device)
    D_X1 = MDiscriminators(i_dim, n_dim, n_down, n_models).to(device)
    G_X0 = Generator(i_dim, n_dim, n_down, s_dim, n_block_c, n_block_s, n_rblock, n_sampler, hidden_neurons).to(device)
    G_X1 = Generator(i_dim, n_dim, n_down, s_dim, n_block_c, n_block_s, n_rblock, n_sampler, hidden_neurons).to(device)
    print('generator and discriminator are ready')
    CT_X = 10  # constant scale for X reconstruction loss  
    CT_C = 1   # constant scale for C reconstruction loss 
    CT_S = 1   # constant scale for S reconstruction loss 

    RL, FL = 1, 0  # Least Square GAN LABELS   
    reconstruction_criterion = nn.L1Loss().to(device)

    G01_optimizer = optim.Adam(it.chain(G_X0.parameters(), G_X1.parameters()), lr=1e-4, betas=(0.5, 0.999))
    DX0_optimizer = optim.Adam(D_X0.parameters(), lr=1e-4, betas=(0.5, 0.999))
    DX1_optimizer = optim.Adam(D_X1.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    epoch_counter = 0 
    while epoch_counter < nb_epochs:
        for idx, (X0, X1) in enumerate(loader):
            X0 = X0.to(device)
            X1 = X1.to(device)

            S0_Z = th.randn(X0.shape[0], s_dim).to(device)
            S1_Z = th.randn(X0.shape[0], s_dim).to(device)

            G01_optimizer.zero_grad()

            X0_C, X0_S = G_X0.encode(X0)      # get content and style of X0 
            X1_C, X1_S = G_X1.encode(X1)      # get content and style of X1 

            X0_R = G_X0.decode(X0_C, X0_S)    # reconstruction of X0 
            X1_R = G_X1.decode(X1_C, X1_S)    # reconstruction of X1   

            X0_1 = G_X1.decode(X0_C, S1_Z)    # translation from domain X0 to domain X1
            X1_0 = G_X0.decode(X1_C, S0_Z)    # translation from domain X1 to domain X0
            
            X0_CR, X1_SR = G_X1.encode(X0_1)  # reconstruction of X0_C and X1_S 
            X1_CR, X0_SR = G_X0.encode(X1_0)  # reconstruction of X1_C and X0_S 
            
            LX0_rec = reconstruction_criterion(X0_R, X0)        # reconstruction loss of X0 
            LC0_rec = reconstruction_criterion(X0_CR, X0_C.detach())     # content reconstruction loss of X0  
            LS0_rec = reconstruction_criterion(X0_SR, S0_Z)     # style reconstruction loss of X0 
            LX0_gan = D_X0.mean_LS(D_X0(X1_0), RL)              # adversarial loss of X0 

            LX1_rec = reconstruction_criterion(X1_R, X1)        # reconstruction loss of X1 
            LC1_rec = reconstruction_criterion(X1_CR, X1_C.detach())     # content reconstruction loss of X1 
            LS1_rec = reconstruction_criterion(X1_SR, S1_Z)     # style reconstruction loss of X1 
            LX1_gan = D_X1.mean_LS(D_X1(X0_1), RL)              # adversarial loss of X1 
            L01_tot = (LX0_gan + LX1_gan) + CT_X * (LX0_rec + LX1_rec) + CT_C * (LC0_rec + LC1_rec) + CT_S * (LS0_rec + LS1_rec)

            L01_tot.backward()
            G01_optimizer.step()

            DX0_optimizer.zero_grad()
            LD0 = ( D_X0.mean_LS(D_X0(X0), RL) + D_X0.mean_LS(D_X0(X1_0.detach()), FL) )
            LD0.backward()
            DX0_optimizer.step()

            DX1_optimizer.zero_grad()
            LD1 = ( D_X1.mean_LS(D_X1(X1), RL) + D_X1.mean_LS(D_X1(X0_1.detach()), FL) )
            LD1.backward()
            DX1_optimizer.step()
            print(f'[{epoch_counter:03d}/{nb_epochs:03d}]:{idx:05d} | G_Error : {L01_tot.item():07.3f} | D_Error : {LD0.item():07.3f} {LD1.item():07.3f}')
            if idx % 200 == 0:
              save_images(X0, X0_1, f'dump/image{epoch_counter:03d}_{idx:03d}.jpg')
        # end for loop 
        
        epoch_counter = epoch_counter + 1
    # end while loop 


if __name__ == '__main__':
    train()