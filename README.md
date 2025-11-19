MSD-EFR: Multi-Scale Decomposition Enhanced Frequency–Temporal Routing for Long-Term Time Series Forecasting

MSD-EFR is a multi-scale forecasting framework that explicitly decomposes time series into seasonal and trend components, and models them through Mixture of Temporal Experts (MoTE) and Mixture of Frequency Experts (MoFE). The method effectively mitigates distribution drift across heterogeneous temporal patterns.

🌟 Introduction

Long-term time series forecasting remains challenging due to mixed frequency components and temporal distribution drift.
To address this, MSD-EFR introduces a two-stage design:

1. MKSTD Decomposition

The input series is decomposed as:

Seasonal component (S) — high-frequency, short-term variations

Trend component (T) — low-frequency, long-term variations

This decomposition enables different expert modules to be selected for each component.
If both components choose the same experts, the decomposition would lose its utility — therefore, different routing behaviors validate the value of MKSTD.

2. MoTE & MoFE Routing

MoTE (Mixture of Temporal Experts): models seasonal patterns in the time domain. Similar seasonal patterns → similar routing weights.

MoFE (Mixture of Frequency Experts): models trend patterns in the frequency domain, capable of capturing small frequency variations even when the trend curves look similar in the time domain.

Together, they form the MSD-EFR architecture and enhance prediction stability.

🚀 Quick Start
Install dependencies
pip install -r requirements.txt

Prepare datasets

Place ETTh1, Weather, and other datasets under:

./data/

Train a model
python experiments/train.py --config configs/ETTh1.yaml

Evaluate
python experiments/test.py --model_path saved_models/msd_efr.pth

Visualization (samples and gate weights)
python experiments/visualize.py
