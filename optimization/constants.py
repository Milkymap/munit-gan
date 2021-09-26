import torch as th 
import torchvision.transforms as T 

# generator description 
i_dim            = 3
n_dim            = 64 
n_down           = 2
s_dim            = 64
n_block_c        = 3
n_block_s        = 2
n_rblock         = 3
n_sampler        = 2
hidden_neurons   = [256, 256]

# discriminator description 
n_models         = 3 

# image transformer 
img_size = 128
mapper = T.Compose([
    T.Resize((img_size, img_size)), 
    T.Normalize([0.5] * 3, [0.5] * 3)
])