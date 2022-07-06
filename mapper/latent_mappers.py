import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

class Mapper(Module):

    def __init__(self, opts, latent_dim=512):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out

class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts
        # add fusing layer
        self.fc = EqualLinear(1024, 512, lr_mul=0.01, activation='fused_lrelu')
        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x):
        x = self.fc(x)
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

class FullTextMapper(Module):

    def __init__(self, opts, n_latent):
        super(FullTextMapper, self).__init__()
        
        self.opts = opts
        self.n_latent = n_latent
        for c, c_dim in enumerate([512 for _ in range(self.n_latent)]):
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c in range(self.n_latent):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x[:,[c],:])
            out.append(x_c_res)
        out = torch.cat(out, dim=1)
        return out
