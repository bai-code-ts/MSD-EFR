import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers.models.auto.image_processing_auto import model_type
import matplotlib.pyplot as plt
import os


from ts_benchmark.baselines.msd_efr.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict, Optional, Tuple
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)

from ...models.model_base import ModelBase, BatchMaker

from ts_benchmark.baselines.msd_efr.layers.Autoformer_EncDec import series_decomp


from ts_benchmark.baselines.msd_efr.models.decomp_ms import Decomp_MS
# from ts_benchmark.baselines.msd_efr.models.decomp_ms_fusion import Decomp_MS_FUS
from ts_benchmark.baselines.msd_efr.models.decomp_ms_c3_2 import Decomp_MS_Change
from ts_benchmark.baselines.msd_efr.models.decomp_ms_c3_1 import Decomp_MS_Change1
from ts_benchmark.baselines.msd_efr.models.decomp_ms_c4 import Decomp_MS_Change4

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "parallel_strategy": "DP",
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.parallel_strategy not in [None, "DP"]:
            raise ValueError(
                "Invalid value for parallel_strategy. Supported values are 'DP' and None."
            )

    @property
    def pred_len(self):
        return self.horizon


class MSD_EFR(ModelBase):
    def __init__(self, **kwargs):
        super(MSD_EFR, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len

    @property
    def model_name(self):
        return "MSD_EFR"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )

                output, _ = self.model(input)

                target = target[:, -config.horizon :, :]
                output = output[:, -config.horizon :, :]
                loss = criterion(output, target).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time data data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)



        self.model = Decomp_MS_Change4(self.config)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        # print(device_ids)
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        self.scaler.fit(train_data.values)

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )

        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input

                output, loss_importance = self.model(input)

                target = target[:, -config.horizon :, :]
                output = output[:, -config.horizon :, :]

                #################################         LOSS                  ############################################
                # ########## MSE  ##############
                # loss = criterion(output, target)
                # ########## MSE  ##############

                series_de = series_decomp(kernel_size=3)
                ####################        change         ###########################
                rec_lambda = 1.0
                auxi_lambda = 0.0
                moe_lambda = 1.0
                self.patch_len_threshold = 24
                self.kl_loss = nn.KLDivLoss(reduction='none')


                output_s, output_t = series_de(output)
                target_s, target_t = series_de(target)

                ############################          mse loss        ###############################
                loss = criterion(target, output)
                ############################          mse loss        ###############################

                # loss_auxi = criterion(target_s, output_s)
                ############################          fre loss        ###############################
                loss_auxi = torch.fft.rfft(target, dim=1) - torch.fft.rfft(output, dim=1)
                loss_auxi = (loss_auxi.abs()).mean()
                ############################          fre loss        ###############################

                ############################          PS loss        ###############################
                # loss_ps = self.ps_loss(target, output)
                # loss_auxi = loss_ps
                ############################          PS loss        ###############################


                #####################        dynamic weights           #####################
                # rec_lambda, auxi_lambda = self.gradient_based_dynamic_weighting_double(target, output, loss, loss_auxi)
                #####################        dynamic weights           #####################


                total_loss = (rec_lambda * loss + moe_lambda * loss_importance + auxi_lambda * loss_auxi)
                total_loss.backward()

                # if i % 100 == 0 and i > 0:
                    # print(i, loss, loss_importance, loss_auxi)

                optimizer.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, criterion)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)



    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param testdata: Time data data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.config.norm:
            train = pd.DataFrame(
                self.scaler.transform(train.values),
                columns=train.columns,
                index=train.index,
            )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        train, test = split_before(train, len(train) - config.seq_len)

        # Additional timestamp marks required to generate transformer class methods
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()


        print('test~')
        with torch.no_grad():
            answer = None

            all_preds = []
            all_trues = []

            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    output, _ = self.model(input)

                    all_preds.append(output.detach().cpu().numpy())
                    all_trues.append(target.detach().cpu().numpy())

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon :]




                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    if self.config.norm:
                        all_preds = np.concatenate(all_preds, axis=0)
                        all_trues = np.concatenate(all_trues, axis=0)
                        all_preds = self.scaler.inverse_transform(
                            all_preds.reshape(-1, all_preds.shape[-1])
                        )
                        all_trues = self.scaler.inverse_transform(
                            all_trues.reshape(-1, all_trues.shape[-1])
                        )
                    else:
                        all_preds = np.concatenate(all_preds, axis=0).reshape(-1, column_num)
                        all_trues = np.concatenate(all_trues, axis=0).reshape(-1, column_num)


                    feature_id = 0  # 要画的特征索引
                    plt.figure(figsize=(10, 4))
                    plt.plot(all_trues[:, feature_id], label='Ground Truth', linewidth=2)
                    plt.plot(all_preds[:, feature_id], label='Prediction', linestyle='--', linewidth=2)
                    plt.legend()
                    plt.title(f'Forecast vs Ground Truth (feature {feature_id})')
                    plt.xlabel('Time step')
                    plt.ylabel('Value')
                    plt.tight_layout()

                    # import datetime
                    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    plt.savefig(f'./test/forecast_compare.png', dpi=300)

                    # plt.show()

                    return answer[-horizon:]

                output = output.cpu().numpy()[:, -config.horizon :, :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon :]
                test = self.padding_data_for_forecast(test)




                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )



    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                output, _ = self.model(input)
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]

                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)

        # # ---- 新增的部分：可视化最终的预测和真实值对比 ----
        # final_pred = answers[:, -horizon:, :]  # 取最后预测的部分
        # final_true = target_np[:, -horizon:, :]  # 真实值的最后部分
        #
        # # 选择维度（假设我们只关心第一列数据）
        # final_pred_values = final_pred[0, :, 0]  # 取第一个样本的预测结果
        # final_true_values = final_true[0, :, 0]  # 取第一个样本的真实值
        #
        # # 绘制最终对比图
        # plt.figure(figsize=(12, 4))
        # plt.plot(final_true_values, label='True Values', linewidth=2)
        # plt.plot(final_pred_values, label='Predicted Values', linestyle='--', linewidth=2)
        # plt.legend()
        # plt.title('Final Prediction vs True (Last Horizon Steps)')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        # plt.tight_layout()
        #
        # # 保存最终对比图
        # folder_path = './results/final_predictions/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # 
        # plt.savefig(os.path.join(folder_path, 'final_prediction_vs_true.png'), dpi=300)
        # plt.close()

        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np






    def gradient_based_dynamic_weighting_double(self, true, pred, corr_loss, var_loss):

        # true = true.permute(0, 2, 1)
        # pred = pred.permute(0, 2, 1)
        # true_mean = torch.mean(true, dim=-1, keepdim=True)
        # pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        # true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        # pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        # true_std = torch.sqrt(true_var)
        # pred_std = torch.sqrt(pred_var)
        # true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        #
        # linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        # linear_sim = (1.0 + linear_sim) * 0.5
        #
        # var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        # mean_gradient = torch.autograd.grad(mean_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient ) / 2.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta = gradiant_avg.norm().detach() / var_gradient.norm().detach()
        # gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        # gamma = gamma * torch.mean(linear_sim * var_sim).detach()

        return aplha, beta

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):

        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)

        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5

        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.model.linear_head.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient ) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta = gradiant_avg.norm().detach() / var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()

        return aplha, beta, gamma
    
    def ps_loss(self, true, pred):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)
        
        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return ps_loss

    def create_patches(self, x, patch_len, stride):
        # print(x.shape, patch_len, stride)       torch.Size([32, 96, 21]) 24 12
        x = x.permute(0, 2, 1)  # [B, C, L]
        B, C, L = x.shape

        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        # print(patches.shape)        # torch.Size([32, 21, 7, 24])

        return patches

    def fouriour_based_adaptive_patching(self, true, pred):

        # print(true.shape)   # 32 96 7       torch.Size([32, 96, 21])

        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        # print(true_fft.shape)     torch.Size([32, 49, 7])

        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        # print(frequency_list.shape)     49
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        # print(top_index)    4

        period = (true.shape[1] // top_index)  ###  L // f  ===> p
        patch_len = min(period // 2, self.patch_len_threshold)  # patch = p // 2
        stride = patch_len // 2

        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        # print(true_patch.shape)         # batch channel patch_num 15 patch_len 12       stride = 6
        return true_patch, pred_patch

    def patch_wise_structural_loss(self, true_patch, pred_patch):

        # print('TRUE',true_patch.shape)      TRUE torch.Size([32, 21, 7, 24])
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        # print(true_patch_mean.shape)        torch.Size([32, 21, 7, 1])

        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)

        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1,
                                         keepdim=True)

        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()

        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

        return linear_corr_loss, var_loss, mean_loss