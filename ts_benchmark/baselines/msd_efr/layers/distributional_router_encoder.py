import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        input_size = config.seq_len
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out



class encoder_fre(nn.Module):
    def __init__(self, config):
        super(encoder_fre, self).__init__()
        input_size = config.seq_len
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size        #### change 256 to 128

        fre_size = int(input_size//2) + 1

        # self.distribution_fit = nn.Sequential(linear_frequency(fre_size, encoder_hidden_size), nn.ReLU(),
        #                                       linear_frequency(encoder_hidden_size, num_experts))

        self.linear_fre1 = linear_frequency(fre_size, encoder_hidden_size)
        self.linear_fre2 = linear_frequency(encoder_hidden_size, num_experts)
        self.gelu = nn.GELU()


    def forward(self, x):
        # print(x.shape)      torch.Size([896, 512, 1])
        # mean = torch.mean(x, dim=-1)
        x = x.squeeze(-1)
        out = self.distribution_fit(x)
        return torch.abs(out)

    def distribution_fit(self, x) :

        x_fre1 = self.linear_fre1(x)

        real = self.gelu(x_fre1.real)
        imag = self.gelu(x_fre1.imag)

        x_fre2 = torch.complex(real, imag)

        x_fre2 = self.linear_fre2(x_fre2)

        return x_fre2



class linear_frequency(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear_frequency, self).__init__()

        self.real = nn.Linear(in_features, out_features, bias=False)
        self.imag = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # print('X',x.shape)      # torch.Size([896, 512, 1])
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)

        x = torch.complex(real, imag)

        return x