# from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster import Linear_extractor_cluster
from openpyxl.styles.builtins import output
from sympy import false

from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster_NoDe import Linear_extractor_cluster
from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster_NoDe_fre import Linear_extractor_cluster_fre


import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.msd_efr.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch
from ts_benchmark.baselines.msd_efr.layers.Autoformer_EncDec import series_decomp
from ts_benchmark.baselines.msd_efr.layers.RevIN import RevIN
from ts_benchmark.baselines.msd_efr.layers.Embed import DataEmbedding_wo_pos
import torch.nn.functional as F


class Decomp_MS_Change4(nn.Module):
    def __init__(self, config):
        super(Decomp_MS_Change4, self).__init__()
        # self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,           ### change
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )


        self.linear_head = nn.Linear(config.d_model, config.pred_len)
        self.drop_layer = nn.Dropout(config.fc_dropout)

        ######################################              change                        #####################################
        self.kernel_list = [24, 12, 6, 3]
        self.decomp_list = [
            series_decomp(kernel_size=kernel) for kernel in self.kernel_list
        ]
        self.cluster_s = Linear_extractor_cluster(config)
        self.cluster_t = Linear_extractor_cluster_fre(config)

        self.linear_concat = nn.Linear(len(self.kernel_list),1)
        self.revin = RevIN(self.n_vars)

        # self.linear_trans = nn.Sequential(nn.Linear(config.enc_in*len(self.kernel_list), config.enc_in), nn.Dropout(config.fc_dropout))

        # self.weights_s = nn.Parameter(torch.ones(len(self.kernel_list)))
        # self.weights_t = nn.Parameter(torch.ones(len(self.kernel_list)))

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        B, L, C = input.shape
        num_parts = len(self.kernel_list)
        # input_mixing = input
        #######################################            norm                             #############
        input = self.revin(input, "norm")
        #######################################            norm                             #############

        #######################################            season & trend List             #####################################
        season_list = []
        trend_list = []
        for i in range(len(self.kernel_list)):
            season, trend = self.decomp_list[i](input)
            season_list.append(season)
            trend_list.append(trend)
        # print([season.shape for season in season_list])
        # [torch.Size([32, 512, 7]), torch.Size([32, 512, 7]), torch.Size([32, 512, 7]), torch.Size([32, 512, 7])]
        season_list = [rearrange(s, 'b l c -> (b c) l 1') for s in season_list]
        trend_list = [rearrange(t, 'b l c -> (b c) l 1') for t in trend_list]
        # print([season.shape for season in season_list])
        # [torch.Size([224, 512, 1]), torch.Size([224, 512, 1]), torch.Size([224, 512, 1]), torch.Size([224, 512, 1])]
        #######################################            season & trend List             #####################################

        ####################          moe input         #################
        s_concat = torch.cat(season_list, dim=0)
        t_concat = torch.cat(trend_list, dim=0)
        # print(s_concat.shape)       # torch.Size([896, 512, 1])
        ####################          moe input         #################


        s_moe_embed, L_importance_s = self.cluster_s(s_concat, C)
        t_moe_embed, L_importance_t = self.cluster_t(t_concat, C)
        # print(s_moe_embed.shape)    # torch.Size([896, 256, 1])

        s_chunks = torch.chunk(s_moe_embed, num_parts, dim=0)
        t_chunks = torch.chunk(t_moe_embed, num_parts, dim=0)
        s_list = [rearrange(s, '(b c) l 1 -> b l c', c=C) for s in s_chunks]
        t_list = [rearrange(t, '(b c) l 1 -> b l c', c=C) for t in t_chunks]
        # print([season.shape for season in s_list])
        # [torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7])]

        s_denorm_list = [layer(s, mode="denorm") for s, layer in zip(s_list, self.cluster_s.revin_layers)]
        t_denorm_list = [layer(t, mode="denorm") for t, layer in zip(t_list, self.cluster_t.revin_layers)]

        ##############################  weight denorm  #################################
        # w_s = F.softmax(self.weights_s, dim=0)
        # w_t = F.softmax(self.weights_t, dim=0)
        # s_weight_list = [ws * si for ws, si in zip(w_s, s_denorm_list)]
        # t_weight_list = [wt * ti for wt, ti in zip(w_t, t_denorm_list)]
        ##############################  weight denorm  #################################

        decomp_out_list = [s + t for s, t in zip(s_denorm_list, t_denorm_list)]
        # print([decomp_out.shape for decomp_out in decomp_out_list])
        # [torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7])]
        #################################       OUT         ############################################

        temporal_feature = torch.stack(decomp_out_list, dim=-1).sum(-1)

        # print(decomp_out.shape)     #   torch.Size([32, 256, 7])




        # temporal_feature = rearrange(decomp_out, 'b l c n -> b (c n) l')
        # print(temporal_feature.shape)       # torch.Size([32, 28, 256])

        #--------------------------------------------------------------------------------------------------------------#



        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            ################    w/o CCM    ###############

            # changed_input = rearrange(input_mixing, 'b l n -> b n l')
            # channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=None)
            ################    w/o CCM    ###############

            # print('OOUT',output.shape)      # OOUT torch.Size([32, 7, 256])
            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = self.drop_layer(output)

        output = rearrange(output, 'b n d -> b d n')
        # print('O',output.shape)     # torch.Size([32, 96, 7])
        ######################
        output = self.revin(output, "denorm")
        return output, L_importance_s+L_importance_t
