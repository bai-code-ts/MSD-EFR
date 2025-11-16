import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .linear_pattern_extractor import Linear_extractor as expert
from .distributional_router_encoder import encoder
from ..layers.RevIN import RevIN
from einops import rearrange



class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, d_model):
        super(MultiScaleSeasonMixing, self).__init__()

        self.d_model = d_model

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.d_model,
                        self.d_model,
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.d_model,
                        self.d_model,
                    ),

                )
                for _ in range(4)
            ]
        )
        print('down',len(self.down_sampling_layers))

    def forward(self, season, B, C):

        # season torch.Size([32, 28, 256])
        #######
        season = rearrange(season, 'b (c l) d -> l b c d', c=C)
        # print(season.shape)     # torch.Size([4, 32, 7, 256])

        outs = []

        out = self.down_sampling_layers[0](season[0])  # 第一步
        outs.append(out)

        for i in range(1, 4):
            # print(i)
            out = self.down_sampling_layers[i](out + season[i])  # 累加再进函数
            outs.append(out)

        return torch.stack(outs, dim=0)

