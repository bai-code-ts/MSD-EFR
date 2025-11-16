# from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.msd_efr.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer,EncoderLayer_c3, Encoder_c3
import torch
from ts_benchmark.baselines.msd_efr.layers.Autoformer_EncDec import series_decomp
from ts_benchmark.baselines.msd_efr.layers.RevIN import RevIN
from ts_benchmark.baselines.msd_efr.layers.linear_extractor_cluster_NoDe import Linear_extractor_cluster


class Decomp_MS_Change1(nn.Module):
    def __init__(self, config):
        super(Decomp_MS_Change1, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator1 = Mahalanobis_mask(config.d_model)
        self.mask_generator2 = Mahalanobis_mask(config.d_model)
        # self.Channel_transformer = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(
        #                     False,           ### change
        #                     config.factor,
        #                     attention_dropout=config.dropout,
        #                     output_attention=config.output_attention,
        #                 ),
        #                 config.d_model,
        #                 config.n_heads,
        #             ),
        #             config.d_model,
        #             config.d_ff,
        #             dropout=config.dropout,
        #             activation=config.activation,
        #         )
        #         for _ in range(config.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(config.d_model)
        # )

        self.Channel_transformer_dual = Encoder_c3(
            [
                EncoderLayer_c3(
                    AttentionLayer(
                        FullAttention(
                            True,  ### change
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            True,  ### change
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


        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

        ######################################              change                        #####################################
        self.kernel_list = [24, 12, 6, 3]
        self.decomp_list = [
            series_decomp(kernel_size=kernel) for kernel in self.kernel_list
        ]
        self.cluster_s = Linear_extractor_cluster(config)
        self.cluster_t = Linear_extractor_cluster(config)

        self.linear_concat = nn.Linear(len(self.kernel_list),1)
        self.revin = RevIN(self.n_vars)

        self.revin_attention_layers = nn.ModuleList([RevIN(self.n_vars) for _ in range(4)])


    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        B, L, C = input.shape
        num_parts = len(self.kernel_list)

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

        ################################            MoE             ##########################################
        s_moe_embed, L_importance_s = self.cluster_s(s_concat, C)
        t_moe_embed, L_importance_t = self.cluster_t(t_concat, C)
        # print(s_moe_embed.shape)    # torch.Size([896, 256, 1])
        ################################            MoE             ##########################################

        s_chunks = torch.chunk(s_moe_embed, num_parts, dim=0)
        t_chunks = torch.chunk(t_moe_embed, num_parts, dim=0)
        s_list = [rearrange(s, '(b c) l 1 -> b l c', c=C) for s in s_chunks]
        t_list = [rearrange(t, '(b c) l 1 -> b l c', c=C) for t in t_chunks]
        # print([season.shape for season in s_list])
        # [torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7])]

        ################         denorm decompostion         #######################
        s_denorm_list = [layer(s, mode="denorm") for s, layer in zip(s_list, self.cluster_s.revin_layers)]
        t_denorm_list = [layer(t, mode="denorm") for t, layer in zip(t_list, self.cluster_t.revin_layers)]
        ################         denorm decompostion         #######################


        decomp_out_list = [s + t for s, t in zip(s_denorm_list, t_denorm_list)]
        # print([decomp_out.shape for decomp_out in decomp_out_list])
        # [torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7]), torch.Size([32, 256, 7])]

        #################        norm ?         ##########
        decomp_out_norm_list = [layer(out, mode="norm") for out, layer in zip(decomp_out_list, self.revin_attention_layers)]
        #################        norm ?         ##########


        #--------------------------------------------------------------------------------------------------------------#
        # 处理 进入 attention 的 输入
        # decomp_out = torch.stack(decomp_out_list, dim=-1)
        # print(decomp_out.shape)     #torch.Size([32, 256, 7, 4])

        temporal_feature = torch.cat(decomp_out_norm_list, dim=-1)

        #################       concat linear           ###################
        # temporal_feature = self.linear_concat(decomp_out).squeeze(-1)
        # print(temporal_feature.shape)     # torch.Size([32, 256, 7])
        #################       concat linear           ###################
        #--------------------------------------------------------------------------------------------------------------#



        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d c -> b c d')
        if self.n_vars > 1:

            ################    w/o CCM    ###############
            #### change input   ####
            changed_input = torch.cat(decomp_out_norm_list, dim=0)
            # print(changed_input.shape)  # torch.Size([128, 256, 7])
            changed_input1 = rearrange(changed_input, 'b l n -> b n l')

            changed_input2 = changed_input1.reshape(B*C, len(self.kernel_list), -1)

            #### change input   ####

            channel_mask1 = self.mask_generator1(changed_input1)
            channel_mask2 = self.mask_generator2(changed_input2)

            channel_group_feature, attention = self.Channel_transformer_dual(x=temporal_feature, attn_mask1=channel_mask1, attn_mask2=channel_mask2)
            ################    w/o CCM    ###############


            ################        denorm attention output        ################
            out_chunks = torch.chunk(channel_group_feature, num_parts, dim=1)
            out_trans = [rearrange(out, 'b c d -> b d c') for out in out_chunks]
            out_denorm_list = [layer(out, mode="denorm") for out, layer in zip(out_trans, self.revin_attention_layers)]
            out_list = [rearrange(out, 'b d c -> b c d') for out in out_denorm_list]
            # print('OOO',[o.shape for o in out_list])
            channel_group_feature = torch.stack(out_list, dim=-1).sum(-1)
            # print(channel_group_feature.shape)
            ################        denorm attention output        ################


            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        # print('O',output.shape)     # torch.Size([32, 96, 7])
        ######################
        output = self.revin(output, "denorm")
        return output, L_importance_s+L_importance_t
