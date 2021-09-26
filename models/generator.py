import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import operator as op 
import itertools as it, functools as ft 

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = None
        self.bias = None
        
        self.register_buffer("running_mean", th.zeros(num_features))
        self.register_buffer("running_var", th.ones(num_features))

    def forward(self, x):
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

class ResidualBlock(nn.Module):
    def __init__(self, filters, norm='in'):
        super(ResidualBlock, self).__init__()
        self.norm_mapper = {
            'bn': nn.BatchNorm2d,
            'in': nn.InstanceNorm2d,
            'an': AdaptiveInstanceNorm2d
        }
        self.body = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            self.norm_mapper[norm](filters),  # adin
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            self.norm_mapper[norm](filters)   # adin 
        )
    
    def forward(self, X):
        return X + self.body(X)

class ContentEncoder(nn.Module):
    #c7s1-64, d128, d256, R256, R256, R256, R256
    def __init__(self, i_dim, n_dim, n_down, n_block):
        super(ContentEncoder, self).__init__()
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(i_dim, n_dim, 7), nn.InstanceNorm2d(n_dim), nn.ReLU())
        self.down = nn.Sequential(
            *list(it.chain(*[ 
                [
                    nn.Conv2d(n_dim * 2 ** i, n_dim * 2 ** (i + 1), 4, 2, 1), 
                    nn.InstanceNorm2d(n_dim * 2 ** (i + 1)), 
                    nn.ReLU()
                ] 
                for i in range(n_down)
            ]))
        )
        self.body = nn.Sequential(*[ ResidualBlock(n_dim * 2 ** (n_down)) for _ in range(n_block)])

    def forward(self, X):
        return self.body(self.down(self.head(X)))


class StyleEncoder(nn.Module):
    #c7s1-64, d128, d256, d256, d256, GAP, fc8
    def __init__(self, i_dim, n_dim, n_down, n_block, s_dim):
        super(StyleEncoder, self).__init__()
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(i_dim, n_dim, 7), nn.ReLU())
        self.down = nn.Sequential(
            *list(it.chain(*[ 
                [
                    nn.Conv2d(n_dim * 2 ** i, n_dim * 2 ** (i + 1), 4, 2, 1), 
                    nn.ReLU()
                ] 
                for i in range(n_down)
            ]))
        )
        self.body = []
        for _ in range(n_block):
            self.body.append(nn.Conv2d(n_dim * 2 ** n_down, n_dim * 2 ** n_down, 4, 2, 1))
            self.body.append(nn.ReLU())
        self.body = nn.Sequential(*self.body)
        self.tail = nn.AdaptiveAvgPool2d(1)
        self.term = nn.Conv2d(n_dim * 2 ** n_down, s_dim, 1, 1, 0)

    def forward(self, X):
        return th.squeeze(self.term(self.tail(self.body(self.down(self.head(X))))))


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.shapes = list(zip(layers[:-1], layers[1:]))
        self.linears = nn.ModuleList([ nn.Linear(m, n) for m,n in self.shapes ])
        self.activations = nn.ModuleList([ nn.ReLU() for _ in range(len(self.shapes) - 1) ])
        self.activations.append(nn.Identity())
    
    def forward(self, X):
        return ft.reduce(
            lambda acc, crr: crr[1](crr[0](acc)), 
            zip(self.linears, self.activations), 
            X
        )

class Sampler(nn.Module):
    def __init__(self, n_times, i_dim, o_dim): 
        super(Sampler, self).__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=n_times), 
            nn.Conv2d(i_dim, o_dim, 5, 1, 2)
        )
    
    def forward(self, X):
        return self.body(X)

class Decoder(nn.Module):
    def __init__(self, filters, n_rblock, n_sampler, o_channel):
        super(Decoder, self).__init__()
        self.residuals = nn.Sequential(*[ ResidualBlock(filters, norm='an') for _ in range(n_rblock) ])
        self.samplers = nn.Sequential(*[ Sampler(2, filters // 2 ** i, filters // 2 ** (i + 1))  for i in range(n_sampler) ])
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3), 
            nn.Conv2d(filters // 2 ** n_sampler, o_channel, 7), 
            nn.Tanh()
        )
    
    def forward(self, X):
        return self.tail(self.samplers(self.residuals(X)))
    
    def assign_adain_params(self, adain_params):
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self):
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features  # mean and std 
        return num_adain_params

class Generator(nn.Module):
    def __init__(self, i_dim, n_dim, n_down, s_dim, n_block_c, n_block_s, n_rblock, n_sampler, hidden_neurons):
        super(Generator, self).__init__()
        self.content_encoder = ContentEncoder(i_dim, n_dim, n_down, n_block_c)
        self.style_encoder = StyleEncoder(i_dim, n_dim, n_down, n_block_s, s_dim)
        self.decoder = Decoder(n_dim * 2 ** n_down, n_rblock, n_sampler, i_dim)
        self.mlp = MLP([s_dim] + hidden_neurons + [self.decoder.get_num_adain_params()])
        
    def forward(self, images):
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        adain_params = self.mlp(style)
        self.decoder.assign_adain_params(adain_params)
        images = self.decoder(content)
        return images


if __name__ == '__main__':
    G = Generator(3, 64, 2, 64, 4, 2, 4, 2, [256, 256, 256])
    print(G)
    X = th.randn((3, 3, 128, 128))
    print(G(X).shape)