🌟 Introduction

Long-term time series forecasting remains challenging due to mixed frequency components and distribution shift across time.
To address this, MSD-EFR introduces a two-stage design:

1. MKSTD Decomposition

We explicitly decompose the input sequence into:

Seasonal component (S) — short-term, high-frequency dynamics

Trend component (T) — long-term, low-frequency variations

This separation enables the model to select different experts for each component.
If both components always select the same experts, the decomposition would lose its meaning — therefore, routing differences validate the usefulness of MKSTD.

2. MoTE & MoFE Routing

MoTE (Mixture of Temporal Experts) handles the seasonal part in the time domain, where similar samples exhibit similar routing weights.

MoFE (Mixture of Frequency Experts) models the trend in the frequency domain, distinguishing samples even when trend curves appear similar in the time domain.

Together, they form the MSD-EFR framework, enhancing prediction stability and robustness.

🚀 Quick Start
Install dependencies
pip install -r requirements.txt

Prepare datasets

Place ETTh1, Weather, etc. into ./data/.

Train a model
python experiments/train.py --config configs/ETTh1.yaml

Evaluate
python experiments/test.py --model_path saved_models/msd_efr.pth

Visualization (samples + gate selector weights)
python experiments/visualize.py
