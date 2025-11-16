from fs.path import parts
from openpyxl.styles.builtins import output

# from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster import Linear_extractor_cluster
from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster_NoDe import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.msd_efr.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch
from ts_benchmark.baselines.msd_efr.layers.Autoformer_EncDec import series_decomp
from ts_benchmark.baselines.msd_efr.layers.RevIN import RevIN
from ts_benchmark.baselines.msd_efr.layers.DecompostionMixer import MultiScaleSeasonMixing

class Decomp_MS_FUS(nn.Module):
    def __init__(self, config):
        super(Decomp_MS_FUS, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
        self.Season_Channel_transformer = Encoder(
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
        self.Trend_Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,  ### change
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

        self.linear_head_s = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))
        self.linear_head_t = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

        ######################################              change                        #####################################
        self.kernel_list = [24, 12, 6, 3]
        self.decomp_list = [
            series_decomp(kernel_size=kernel) for kernel in self.kernel_list
        ]
        self.cluster_s = Linear_extractor_cluster(config)
        self.cluster_t = Linear_extractor_cluster(config)

        self.linear_concat = nn.Linear(len(self.kernel_list),1)
        self.revin = RevIN(self.n_vars)

        self.multimixing_season = MultiScaleSeasonMixing(config.d_model)
        self.multimixing_trend = MultiScaleSeasonMixing(config.d_model)


    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        B, L, C = input.shape
        num_parts = len(self.kernel_list)

        input = self.revin(input, "norm")

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
        ####################          moe input         #################
        s_concat = torch.cat(season_list, dim=0)
        t_concat = torch.cat(trend_list, dim=0)
        # print(s_concat.shape)       # torch.Size([896, 512, 1])
        ####################          moe input         #################
        #######################################            season & trend List             #####################################



        s_moe_embed, L_importance_s = self.cluster_s(s_concat, C)
        t_moe_embed, L_importance_t = self.cluster_t(t_concat, C)
        # print(s_moe_embed.shape)    # torch.Size([896, 256, 1])

        '''
        s_chunks = torch.chunk(s_moe_embed, num_parts, dim=0)
        t_chunks = torch.chunk(t_moe_embed, num_parts, dim=0)
        s_list = [rearrange(s, '(b c) l 1 -> b l c', c=C) for s in s_chunks]
        t_list = [rearrange(t, '(b c) l 1 -> b l c', c=C) for t in t_chunks]
        # print([season.shape for season in s_list])
        # [torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7])]

        decomp_out_list = [s + t for s, t in zip(s_list, t_list)]
        decomp_out = torch.stack(decomp_out_list, dim=-1)
        # print(decomp_out.shape)     #torch.Size([32, 256, 7, 4])
        temporal_feature = self.linear_concat(decomp_out).squeeze(-1)
        # print(temporal_feature.shape)     # torch.Size([32, 256, 7])
        #--------------------------------------------------------------------------------------------------------------#
        '''
        s_chunks = torch.chunk(s_moe_embed, num_parts, dim=0)
        t_chunks = torch.chunk(t_moe_embed, num_parts, dim=0)
        # print([s.shape for s in s_chunks])
        # [torch.Size([224, 256, 1]), torch.Size([224, 256, 1]), torch.Size([224, 256, 1]), torch.Size([224, 256, 1])]
        s_list = [rearrange(s, '(b c) l 1 -> b c l', c=C) for s in s_chunks]
        t_list = [rearrange(t, '(b c) l 1 -> b c l', c=C) for t in t_chunks]
        s_feature = torch.concat(s_list, dim=1)
        t_feature = torch.concat(t_list, dim=1)
        # print(s_feature.shape)      #torch.Size([32, 7x4, 256])




        ################     CCM    ###############
        # changed_input = rearrange(input, 'b l n -> b n l')
        # channel_mask = self.mask_generator(changed_input)
        s_group_feature, attention = self.Season_Channel_transformer(x=s_feature, attn_mask=None)
        # print([att.shape for att in attention])         ### [torch.Size([32, 1, 28, 28])]
        t_group_feature, attention = self.Trend_Channel_transformer(x=t_feature, attn_mask=None)
        # print(s_group_feature.shape)        # torch.Size([32, 28, 256])
        ################     CCM    ###############

        ################    T/S-Mix      ##############
        # print(s_group_feature.shape)        # # torch.Size([32, 28, 256])
        # s_mixing_feature = self.multimixing_season(s_group_feature,B,C).reshape(B, 4 * C, -1)
        # t_mixing_feature = self.multimixing_trend(t_group_feature,B,C).reshape(B, 4 * C, -1)
        # print(s_mixing_feature.shape)       # torch.Size([32, 28, 256])
        ################    T/S-Mix      ##############



        output_s = self.linear_head_s(s_group_feature)
        output_t = self.linear_head_t(t_group_feature)
        output_s = rearrange(output_s, 'b c p -> b p c')
        output_t = rearrange(output_t, 'b c p -> b p c')

        ############### denorm  ####################
        s_chunks_denorm = torch.chunk(output_s, num_parts, dim=-1)
        t_chunks_denorm = torch.chunk(output_t, num_parts, dim=-1)
        # print('a',[s.shape for s in s_chunks_denorm])
        # a [torch.Size([32, 192, 7]), torch.Size([32, 192, 7]), torch.Size([32, 192, 7]), torch.Size([32, 192, 7])]
        s_denorm_list = [layer(s, mode="denorm") for s, layer in zip(s_chunks_denorm, self.cluster_s.revin_layers)]
        t_denorm_list = [layer(t, mode="denorm") for t, layer in zip(t_chunks_denorm, self.cluster_t.revin_layers)]


        # output_s = self.cluster_s.revin(output_s, "denorm")
        # output_t = self.cluster_t.revin(output_t, "denorm")

        # output = output_s + output_t
        out_list = [s + t for s, t in zip(s_denorm_list, t_denorm_list)]

        # out_list = torch.chunk(output, num_parts, dim=-1)
        # print([out.shape for out in out_list])
        # [torch.Size([32, 192, 7]), torch.Size([32, 192, 7]), torch.Size([32, 192, 7]), torch.Size([32, 192, 7])]
        # out = torch.stack(out_list, dim=-1)
        # print(out.shape)        # torch.Size([32, 192, 7, 4])
        # out = self.linear_concat(out).reshape(B,-1,C)

        out = torch.stack(out_list, dim=0).sum(dim=0)

        # print('OUT Season',output_s.shape)  # OUT Season torch.Size([32, 192, 28])
        # output = self.linear_concat(output.reshape(B,-1,C,4)).reshape(B,-1,C)
        out = self.revin(out, "denorm")

        return out, L_importance_s+L_importance_t
