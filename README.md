# MSD-EFR: Multi-Scale Decomposition Enhanced Frequency–Temporal Routing for Long-Term Time Series Forecasting

**MSD-EFR** is a multi-scale forecasting framework that explicitly decomposes time series into seasonal and trend components, and models them through **Mixture of Temporal Experts (MoTE)** and **Mixture of Frequency Experts (MoFE)**.  
The method effectively mitigates *distribution drift* across heterogeneous time series patterns.

---

## 🌟 Introduction

Long-term time series forecasting remains challenging due to mixed frequency components and distribution drift across time.  
To address this, **MSD-EFR** introduces a two-stage design:

### **1. MKSTD Decomposition**
We explicitly decompose the input sequence into:

- **Seasonal component (S)** — short-term, high-frequency dynamics  
- **Trend component (T)** — long-term, low-frequency variations  

This separation enables the model to select *different experts* for each component.  
If both components always select the same experts, the decomposition would lose its meaning — therefore, routing differences validate the usefulness of MKSTD.

### **2. MoTE & MoFE Routing**

- **MoTE (Mixture of Temporal Experts)** handles the seasonal part in the **time domain**, where similar samples exhibit similar routing weights.
- **MoFE (Mixture of Frequency Experts)** models the trend in the **frequency domain**, distinguishing samples even when trend curves appear similar in the time domain.

Together, they form the **MSD-EFR framework**, enhancing prediction stability and robustness.

---

## 🧩 Framework Overview

### **Framework Architecture**

![framework](./pic/framwork_3.pdf)



## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare datasets

Place ETTh1, Weather, and other datasets into:
```bash
./data/
```

### Train the model
```bash
python experiments/train.py --config configs/ETTh1.yaml
```

### Evaluate the model
```bash
python experiments/test.py --model_path saved_models/msd_efr.pth
```

### Visualize samples + gate routing
```bash
python experiments/visualize.py
```

## 📊 Experimental Results

[Visualization Example (PDF)](pic/e5.pdf)








