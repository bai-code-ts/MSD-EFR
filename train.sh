#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" \
#    --data-name-list "Weather.csv"\
#    --strategy-args '{"horizon": 720}' --model-name "msd_efr.MSD_EFR" --model-hyper-params '{"CI": 1, "batch_size": 32,
#    "d_ff": 256, "d_model": 256, "e_layers": 2, "factor": 3,
#    "horizon": 720, "k": 2, "loss": "MAE", "lr": 0.0005,
#    "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5,
#    "seq_len": 96}' --deterministic "full" \
#    --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/MSD_EFR"

#
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh2.csv" \
    --strategy-args '{"horizon": 96}' --model-name "msd_efr.MSD_EFR" --model-hyper-params '{"CI": 1, "batch_size": 32,
    "d_ff": 512, "d_model": 256, "dropout": 0.15, "e_layers": 1, "factor": 3,
    "horizon": 96, "k": 2, "loss": "MAE", "lr": 0.0001,
    "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5,
    "seq_len": 512}' --deterministic "full" \
    --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/MSD_EFR"


