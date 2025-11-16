import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp


class Linear_extractor(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear_extractor, self).__init__()



        self.seq_len = int(configs.seq_len//2)+1

        self.pred_len = int(configs.d_model//2)+1
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            # self.Linear_Cat = nn.Linear(self.seq_len*2, self.pred_len*2)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (0) * torch.ones([self.pred_len, self.seq_len]))
            # self.Linear_Cat.weight = nn.Parameter(
            #     (1 / self.seq_len*2) * torch.ones([self.pred_len*2, self.seq_len*2]))

        # self.Wr = nn.Conv1d(F, F, kernel_size=1, bias=True)
        # self.Wi = nn.Conv1d(F, F, kernel_size=1, bias=True)
        # self.reset_parameters(scale)
        # 初始化为 scale * 单位矩阵
        # nn.init.zeros_(self.Linear_Seasonal.weight)
        # nn.init.zeros_(self.Linear_Trend.weight)
        # nn.init.zeros_(self.Linear_Seasonal.bias)
        # nn.init.zeros_(self.Linear_Trend.bias)

        # with torch.no_grad():
        #     for i in range(self.F):
        #         self.Wr.weight[i, i, 0] = scale
        #         self.Wi.weight[i, i, 0] = 0.0  # 纯实数 scale
        # # 如果想让 scale 是复数，可以修改 Wi.weight[i,i,0]


    def encoder(self, x):

        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXxxxx',x.shape)        torch.Size([498, 257, 1])
        # x = x.permute(0, 2, 1)

        ### Linear
        # real = self.Linear_Seasonal(x.real)
        # imag = self.Linear_Trend(x.imag)

        ### Concat
        # x_cat = torch.cat([x.real, x.imag], dim=-1)
        # out_cat = self.Linear_Cat(x_cat)
        # real, imag = torch.split(out_cat, self.pred_len, dim=-1)

        ### FreMLP
        xr, xi = x.real.permute(0,2,1), x.imag.permute(0,2,1)
        real = self.Linear_Seasonal(xr) - self.Linear_Trend(xi)
        imag = self.Linear_Seasonal(xi) + self.Linear_Trend(xr)

        x = torch.complex(real, imag)

        # seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init, trend_init = seasonal_init.permute(
        #     0, 2, 1), trend_init.permute(0, 2, 1)
        # if self.individual:
        #     seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
        #                                   dtype=seasonal_init.dtype).to(seasonal_init.device)
        #     trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
        #                                dtype=trend_init.dtype).to(trend_init.device)
        #     for i in range(self.channels):
        #         seasonal_output[:, i, :] = self.Linear_Seasonal[i](
        #             seasonal_init[:, i, :])
        #         trend_output[:, i, :] = self.Linear_Trend[i](
        #             trend_init[:, i, :])
        # else:
        #     seasonal_output = self.Linear_Seasonal(seasonal_init)
        #     trend_output = self.Linear_Trend(trend_init)
        # x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)


    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# class ComplexLinear(nn.Module):
#     def __init__(self, seq_len, pred_len):
#         super().__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#
#         self.W_real = nn.Linear(self.seq_len, self.pred_len)
#         self.W_imag = nn.Linear(self.seq_len, self.pred_len)
#
#         self.Linear_Seasonal.weight = nn.Parameter(
#             (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
#         self.Linear_Trend.weight = nn.Parameter(
#             (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
#
#     def forward(self, x):
#         xr, xi = x.real, x.imag
#         real = self.W_real(xr) - self.W_imag(xi)
#         imag = self.W_real(xi) + self.W_imag(xr)
#         return torch.complex(real, imag)